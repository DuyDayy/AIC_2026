"""
Tầng truy vấn trên Modal — ①②③⑤a④ chạy từ xa, ⑤c⑦ chạy tại máy
==================================================================

Vì sao tách ở ĐÚNG chỗ này. Ba tầng đầu là thứ tốn máy: BM25 trên 609.476 khung,
7 lượt chấm mỗi probe, và một `emb.npy` 1,2 GB phải nằm trong RAM. Tại máy nó mất
~35 giây mỗi truy vấn. Ngược lại ⑤c rerank là gọi API bên ngoài và ⑦ chỉ ghi file —
cả hai không tốn máy, mà lại cần khoá API và ổ đĩa của người dùng.

Gói trả về CỐ Ý NHỎ: mỗi truy vấn chỉ mang rổ (≤300 khung) kèm `S[:, rổ]`, tức
`N_mốc × 300` số thực. Trả cả ma trận 609.476 cột thì mỗi truy vấn là 2,4 MB và
24 truy vấn thành 58 MB — không có lý do gì, vì mọi tầng sau chỉ nhìn vào rổ.

Kèm theo là văn bản OCR/ASR của `RERANK_TOP` khung đầu, để ⑤c tại máy không phải
giữ bản sao 2,9 GB `ocr.jsonl`.

Volume `aic-query-index` phải có: emb.npy · ids.npy · frame_idx.npy · ranges.json ·
ocr_min.jsonl · asr.json — dựng bằng `data/_modal_bundle` (1,4 GB, gọn từ 4,3 GB).
"""

from __future__ import annotations

import modal

MODEL = "jinaai/jina-clip-v2"
MODEL_SHA = "e10d47f5691d0454a0fb5d13f46f2199b74cb436"
CODE_SHA = "39e6a55ae971b59bea6e44675d237c99762e7ee2"

app = modal.App("aic-query")
# `create_if_missing`: ⑥ `qa_answer` KHÔNG mount volume này, nên deploy chỉ để dùng
# ⑥ không được chết vì thiếu 1,4 GB chỉ mục. Các hàm cần chỉ mục sẽ tự hỏng khi gọi,
# ở đó lỗi nói đúng nguyên nhân — còn chết lúc deploy thì nói sai.
vol = modal.Volume.from_name("aic-query-index", create_if_missing=True)
hf = modal.Volume.from_name("hf-cache", create_if_missing=True)

# `add_local_*` phải là bước CUỐI của một ảnh — Modal từ chối build step sau nó. Nên
# nền tách riêng, và mỗi ảnh dẫn xuất tự thêm phần của mình rồi mới gắn file local.
_base = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.1.3", "torch==2.5.1", "transformers==4.48.0",
                 "einops", "timm", "pillow==11.1.0", "safetensors")
    .env({"HF_HOME": "/hf"})
)

# Mã của tầng truy vấn: CÙNG module với bản chạy tại máy, không viết lại bản sao.
image = _base.add_local_dir("src", "/root/src")

# ⑥ không cần `src/` — nó chỉ nhận ảnh + chữ và trả chuỗi. NHƯNG nó cần transformers
# MỚI HƠN: `Qwen2_5_VLForConditionalGeneration` chỉ có từ 4.49, còn `_base` ghim 4.48
# cho jina-clip. `run.py` đã tách đúng hai ảnh vì lý do này (4.48 và 4.51.3); tôi chép
# nhầm ghim sang đây và nó nổ ngay lần gọi đầu.
qa_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "transformers==4.51.3", "pillow==11.1.0",
                 "accelerate", "qwen-vl-utils", "safetensors")
    .env({"HF_HOME": "/hf"})
)


@app.function(image=image, gpu="A10G", volumes={"/hf": hf}, timeout=1800)
def encode_text(texts: list[str]) -> list[list[float]]:
    """Mã hoá truy vấn. GHIM revision — vector khung dùng đúng cặp sha này."""
    import numpy as np
    import torch
    from transformers import AutoModel

    m = AutoModel.from_pretrained(MODEL, trust_remote_code=True,
                                  revision=MODEL_SHA, code_revision=CODE_SHA)
    m = m.to("cuda").eval()
    with torch.inference_mode():
        v = np.asarray(m.encode_text(texts, batch_size=32), dtype=np.float32)
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    return v.tolist()


@app.function(image=image, volumes={"/idx": vol}, cpu=8.0, memory=32768, timeout=3600)
def score(payload: dict) -> dict:
    """
    ②③⑤a④ cho MỘT mẻ truy vấn.

    `payload` = {queries: [{id, kind, probes: [str], exp: {ocr: [...], asr: [...]},
                            qv: {text: vector}}],
                 alpha, rrf_k, pool_cap, rerank_top, n_mom_gop: bool}
    """
    import json
    import sys
    import time

    import numpy as np
    sys.path.insert(0, "/root")

    from src.retrieval.pool import fused_pool
    from src.retrieval.score_matrix import hierarchical_rrf
    from src.retrieval.sources import AsrSource, TextSource, VisualSource

    t0 = time.time()
    # Gói tải lên có ĐÚNG layout của `data/embed`, nên dùng thẳng bộ nạp thật thay vì
    # giả lập một đối tượng chỉ mục. Giả lập là chỗ dễ lệch nhất: `VisualSource` đọc
    # `.emb` chứ không phải `.vectors`, và `n_frames`/`dim` là property suy từ shape.
    from src.ingestion.vector_index import load_flat_index

    dim = payload.get("dim", 512)
    ix = load_flat_index("/idx/_modal_bundle", dim=dim)
    ids = list(ix.ids)
    FI = np.asarray(ix.frame_idx)

    ocr_map, fms = {}, {}
    with open("/idx/_modal_bundle/ocr_min.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            k = (d["v"], d["n"])
            if d["t"]:
                ocr_map[k] = d["t"]
            fms[k] = d["f"] / d["fps"] * 1000.0
    asr = {k: [(a, b, t) for a, b, t in v]
           for k, v in json.load(open("/idx/_modal_bundle/asr.json")).items()}
    vis = VisualSource(ix)
    tsrc = [TextSource("ocr", ids, ocr_map), AsrSource(ids, fms, asr)]
    print(f"nạp {time.time() - t0:.0f}s · {ix.n_frames:,} khung · {len(ix.ranges)} video "
          f"· OCR {len(ocr_map):,} · ASR {len(asr)}", flush=True)

    ALPHA = payload["alpha"]
    RRF_K = payload["rrf_k"]
    # Rổ đặt theo TỪNG truy vấn, không một giá trị chung cả mẻ: KIS/QA chỉ cần 1 khung
    # mỗi dòng nên rổ 100 (= ngân sách nộp) là đủ, còn ④ TRAKE cần ≥N khung CÙNG một
    # video mới dựng được đường — rổ hẹp trải trên nhiều video thì rất ít video đủ.
    CAP = payload["pool_cap"]
    RTOP = payload["rerank_top"]
    out = []
    for i, q in enumerate(payload["queries"]):
        rows = []
        for t in q["probes"]:
            qv = np.asarray(q["qv"][t], dtype=np.float32)[:dim]
            qv /= max(float(np.linalg.norm(qv)), 1e-12)
            runs = {"visual": [vis.score(qv)]}
            for s in tsrc:
                mr = q["exp"].get(s.name) or []
                runs[s.name] = [s.score(t)] + [s.score(x) for x in mr]
            beta = {m: (1.0 / len(r),) * len(r) for m, r in runs.items()}
            rows.append(hierarchical_rrf(runs, alpha=ALPHA, beta=beta, k=RRF_K))
        S = np.vstack(rows).astype(np.float32)
        if q["gop"] and S.shape[0] > 1:
            S = S.max(axis=0, keepdims=True)
        pool = fused_pool(S, int(q.get("cap") or CAP))
        base = S.max(axis=0)
        top = [int(r) for r in pool[np.argsort(-base[pool])][:RTOP]]
        ctx = {}
        for r in top:
            v, n = ids[r]
            ms = fms.get((v, n), -1.0)
            a = " ".join(t for x, y, t in asr.get(v, [])
                         if ms >= 0 and x - 5000 <= ms <= y + 5000)[:400]
            ctx[str(r)] = {"ocr": ocr_map.get((v, n), ""), "asr": a}
        out.append({
            "id": q["id"], "pool": pool.tolist(),
            "S_pool": S[:, pool].tolist(),
            "vid": [ids[r][0] for r in pool.tolist()],
            "fi": [int(FI[r]) for r in pool.tolist()],
            "rerank_rows": top, "ctx": ctx,
        })
        print(f"  ②③⑤a {i + 1}/{len(payload['queries'])} {q['id']} "
              f"({time.time() - t0:.0f}s)", flush=True)
    return {"queries": out, "giay": round(time.time() - t0, 1),
            "n_frames": len(ids)}


QA_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


@app.function(image=qa_image, gpu="A10G", volumes={"/hf": hf}, timeout=3600)
def qa_answer(jobs: list[dict]) -> list[str]:
    """
    ⑥ — đọc khung + chứng cứ chữ rồi sinh `answer` cho đề Q&A.

    Không có tầng này thì câu Q&A **được 0 điểm dù tìm đúng khung**: thể lệ đòi
    `aᵢ = GTₐ` ngoài `vᵢ = GTᵥ` và `idᵢ ∈ [s,e]`.

    Nhận `[{image_b64, question, ocr, asr}]`, trả `[answer]` cùng thứ tự. Đưa kèm OCR
    và lời nói vì nhiều câu hỏi — tên riêng, con số, ngày tháng — **chỉ đọc được từ
    chữ**, không nhìn ra từ ảnh.

    ẢNH LẤY TỪ VIDEO GỐC, không lấy từ kho keyframe. Kho `aic-keyframes` đánh số theo
    `n` của bộ trích xuất CŨ (601 khung cho `L21_V001`, chỉ mục hiện tại có 1.778 và
    `n` tới 3.711) — tra theo `n` ở đó sẽ ra ẢNH KHÁC mà không có gì báo. Bên gọi cắt
    thẳng `frame_idx` khỏi `.mp4` bằng ffmpeg nên không thể lệch.
    """
    import base64 as b64
    import io as _io

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    proc = AutoProcessor.from_pretrained(QA_MODEL)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QA_MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    out = []
    for i, j in enumerate(jobs):
        im = Image.open(_io.BytesIO(b64.b64decode(j["image_b64"]))).convert("RGB")
        ctx = ""
        if j.get("ocr"):
            ctx += f"\nChữ hiện trên khung: {j['ocr'][:300]}"
        if j.get("asr"):
            ctx += f"\nLời nói quanh khung: {j['asr'][:400]}"
        msg = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text":
             f"{j['question']}{ctx}\n\nTrả lời NGẮN GỌN bằng tiếng Việt, chỉ nêu đáp "
             f"án, không giải thích. Nếu là số thì viết bằng chữ số."}]}]
        text = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[text], images=[im], return_tensors="pt").to("cuda")
        with torch.inference_mode():
            g = model.generate(**inp, max_new_tokens=48, do_sample=False)
        ans = proc.batch_decode(g[:, inp.input_ids.shape[1]:],
                                skip_special_tokens=True)[0].strip()
        out.append(" ".join(ans.split())[:100])
        print(f"  ⑥ {i + 1}/{len(jobs)} {out[-1][:60]!r}", flush=True)
    return out


@app.function(image=image, volumes={"/idx": vol}, cpu=8.0, memory=65536, timeout=7200)
def rocchio_sweep(payload: dict) -> dict:
    """
    Quét lưới Rocchio RF trên bộ đề có ground truth — vòng 0 so vòng 1.

    VÌ SAO QUÉT Ở ĐÂY chứ không ở máy: phần đắt là ②③ BM25 trên 609.476 khung, và nó
    KHÔNG phụ thuộc Rocchio — Rocchio chỉ đổi run thị giác. Nên chấm BM25 một lần mỗi
    truy vấn rồi tái dùng cho MỌI cấu hình; mỗi cấu hình chỉ tốn thêm một phép
    `emb @ q′` (609.476×512, cỡ mili-giây) và một lần hợp RRF.

    HAI CHẾ ĐỘ, và khoảng cách giữa chúng chính là GIÁ TRỊ CỦA MỘT CÚ CLICK:

      `oracle` — positive lấy từ khung trong top-K THẬT SỰ gần ground truth. Đây là
                 mô phỏng operator của tài liệu §11.1, và là TRẦN của RF.
      `prf`    — positive lấy thẳng `m` khung đầu bảng, không nhìn ground truth. Đây
                 là bản DÙNG ĐƯỢC ngày thi, vì ngày thi không có đáp án.

    Truy vấn không có khung liên quan trong top-K bị đánh dấu `mat_recall` và LOẠI
    khỏi phép so oracle — tài liệu §11.1: đó là lỗi tầng truy hồi thứ nhất, không
    phải lỗi Rocchio.
    """
    import json
    import sys
    import time

    import numpy as np
    sys.path.insert(0, "/root")

    from src.feedback.rocchio import RocchioConfig, l2, update
    from src.ingestion.vector_index import load_flat_index
    from src.retrieval.score_matrix import (
        SourceScores, hierarchical_rrf, rrf_normalize,
    )
    from src.retrieval.sources import AsrSource, TextSource, VisualSource

    t0 = time.time()
    dim = payload.get("dim", 512)
    ix = load_flat_index("/idx/_modal_bundle", dim=dim)
    ids = list(ix.ids)
    ocr_map, fms = {}, {}
    with open("/idx/_modal_bundle/ocr_min.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            k = (d["v"], d["n"])
            if d["t"]:
                ocr_map[k] = d["t"]
            fms[k] = d["f"] / d["fps"] * 1000.0
    asr = {k: [(a, b, t) for a, b, t in v]
           for k, v in json.load(open("/idx/_modal_bundle/asr.json")).items()}
    vis = VisualSource(ix)
    tsrc = [TextSource("ocr", ids, ocr_map), AsrSource(ids, fms, asr)]
    TMS = np.array([fms.get((v, n), -1.0) for v, n in ids], dtype=np.float64) / 1000.0
    VID = np.array([v for v, _ in ids])
    print(f"nạp {time.time() - t0:.0f}s · {ix.n_frames:,} khung", flush=True)

    ALPHA, RRF_K = payload["alpha"], payload["rrf_k"]
    TAU = payload.get("tau", 1.0)
    KS = tuple(payload.get("ks", (1, 5, 10, 20, 50, 100)))
    grid = payload["grid"]        # [{mode, beta, radius_sec, m, max_frames}]
    out = {json.dumps(g, sort_keys=True): {"hit": [], "rr": [], "drank": [], "drift": 0}
           for g in grid}
    base = {"hit": [], "rr": []}
    n_mat_recall = 0

    for qi, q in enumerate(payload["queries"]):
        # ── vòng 0: ②③ đầy đủ, BM25 chấm MỘT lần ────────────────────────────
        qv0 = l2(np.asarray(q["qv"], dtype=np.float32)[:dim])
        v0 = vis.score(qv0)
        text_runs = {}
        for s in tsrc:
            mr = q["exp"].get(s.name) or []
            text_runs[s.name] = [s.score(q["text"])] + [s.score(x) for x in mr]
        text_rrf = {m: sum((1.0 / len(r)) * rrf_normalize(x.scores, x.covered, RRF_K)
                           for x in r) for m, r in text_runs.items()}

        def hop(vrun: SourceScores) -> np.ndarray:
            """③ với run thị giác thay được — OCR/ASR giữ NGUYÊN (tài liệu §4.3)."""
            s = ALPHA["visual"] * rrf_normalize(vrun.scores, vrun.covered, RRF_K)
            for m, r in text_rrf.items():
                s = s + ALPHA[m] * r
            return s.astype(np.float32)

        S0 = hop(v0)
        ord0 = np.argsort(-S0, kind="stable")
        dung = (VID == q["video"]) & (TMS >= 0) & (np.abs(TMS - q["t"]) <= TAU)
        r0 = int(np.flatnonzero(dung[ord0])[0]) + 1 if dung.any() else 10**9
        base["hit"].append({k: r0 <= k for k in KS})
        base["rr"].append(1.0 / r0 if r0 < 10**9 else 0.0)

        top = ord0[:max(int(g.get("m", 5)) for g in grid) * 4]
        oracle_pool = [int(r) for r in top[:100] if dung[r]]
        if not oracle_pool:
            n_mat_recall += 1

        for g in grid:
            key = json.dumps(g, sort_keys=True)
            m = int(g.get("m", 5))
            neo = (oracle_pool[:1] if g["mode"] == "oracle"
                   else [int(r) for r in ord0[:m]])
            if not neo:
                out[key]["hit"].append({k: False for k in KS})
                out[key]["rr"].append(0.0)
                out[key]["drank"].append(0)
                continue
            # ── prototype: cửa sổ thời gian quanh mỗi neo, cùng video ────────
            rows, grp = [], []
            for gi, r in enumerate(neo):
                v, _n = ids[r]
                t = TMS[r]
                sel = np.flatnonzero((VID == v) & (np.abs(TMS - t) <= g["radius_sec"]))
                if sel.size > g.get("max_frames", 7):
                    sel = sel[np.argsort(np.abs(TMS[sel] - t))[:g["max_frames"]]]
                rows.extend(sel.tolist())
                grp.extend([gi] * len(sel))
            P = ix.emb[np.asarray(rows)]
            cfg = RocchioConfig(beta=float(g["beta"]), gamma=0.0,
                                radius_sec=float(g["radius_sec"]),
                                max_frames=int(g.get("max_frames", 7)))
            q1 = update(qv0, P, cfg=cfg, pos_groups=np.asarray(grp))
            if float(np.dot(qv0, q1)) < cfg.drift_threshold:
                out[key]["drift"] += 1
            S1 = hop(vis.score(q1))
            ord1 = np.argsort(-S1, kind="stable")
            r1 = int(np.flatnonzero(dung[ord1])[0]) + 1 if dung.any() else 10**9
            out[key]["hit"].append({k: r1 <= k for k in KS})
            out[key]["rr"].append(1.0 / r1 if r1 < 10**9 else 0.0)
            out[key]["drank"].append((r0 - r1) if r0 < 10**9 else 0)
        if (qi + 1) % 20 == 0:
            print(f"  {qi + 1}/{len(payload['queries'])} ({time.time() - t0:.0f}s)",
                  flush=True)

    def tom(d):
        """
        Trả kèm `rr` TỪNG CÂU, không chỉ trung bình.

        Không có giá trị từng câu thì không bắt cặp được, không bootstrap được, và
        không có khoảng tin cậy — tức không chứng minh được gì, chỉ báo được một con
        số. Bắt cặp là bắt buộc ở đây vì vòng 0 và vòng 1 chạy trên CÙNG truy vấn,
        nên phương sai giữa các truy vấn (rất lớn) triệt tiêu khi lấy hiệu.
        """
        n = len(d["rr"])
        return {"MRR": float(np.mean(d["rr"])),
                **{f"R@{k}": float(np.mean([h[k] for h in d["hit"]])) for k in KS},
                **({"dRank_median": float(np.median(d["drank"])),
                    "dRank_win": float(np.mean([x > 0 for x in d["drank"]])),
                    "drift_rate": d["drift"] / max(n, 1)} if "drank" in d else {}),
                "n": n,
                "rr_moi_cau": [round(float(x), 6) for x in d["rr"]],
                "hit1_moi_cau": [bool(h[1]) for h in d["hit"]]}

    return {"base": tom(base), "configs": {k: tom(v) for k, v in out.items()},
            "query_ids": [q["id"] for q in payload["queries"]],
            "mat_recall": n_mat_recall, "giay": round(time.time() - t0, 1)}


@app.function(image=image, volumes={"/idx": vol}, cpu=8.0, memory=65536, timeout=7200)
def chan_doan(payload: dict) -> dict:
    """
    Vì sao 74% truy vấn KHÔNG có khung đúng trong top-100 — hỏng ở đâu?

    Trả về, cho TỪNG truy vấn, hạng của khung đúng trong:
      · từng nguồn RIÊNG (thị giác · OCR · ASR), mỗi nguồn xếp hạng độc lập
      · điểm ĐÃ HỢP của ③

    Ba kết luận khác nhau, ba hướng sửa khác nhau:

      (a) một nguồn có hạng TỐT mà điểm hợp lại xấu  ⟹ lỗi DUNG HỢP, sửa alpha
      (b) mọi nguồn đều hạng rất sâu                 ⟹ lỗi TRUY HỒI, sửa encoder /
                                                        mở rộng truy vấn / chỉ mục
      (c) khung đúng không tồn tại trong chỉ mục     ⟹ lỗi CẮT KHUNG

    Không đo cái này thì mọi đề xuất "cải thiện top" đều là đoán.
    """
    import json
    import sys
    import time

    import numpy as np
    sys.path.insert(0, "/root")

    from src.ingestion.vector_index import load_flat_index
    from src.retrieval.score_matrix import rrf_normalize
    from src.retrieval.sources import AsrSource, TextSource, VisualSource

    t0 = time.time()
    dim = payload.get("dim", 512)
    ix = load_flat_index("/idx/_modal_bundle", dim=dim)
    ids = list(ix.ids)
    ocr_map, fms = {}, {}
    with open("/idx/_modal_bundle/ocr_min.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            k = (d["v"], d["n"])
            if d["t"]:
                ocr_map[k] = d["t"]
            fms[k] = d["f"] / d["fps"] * 1000.0
    asr = {k: [(a, b, t) for a, b, t in v]
           for k, v in json.load(open("/idx/_modal_bundle/asr.json")).items()}
    vis = VisualSource(ix)
    tsrc = [TextSource("ocr", ids, ocr_map), AsrSource(ids, fms, asr)]
    TMS = np.array([fms.get((v, n), -1.0) for v, n in ids], dtype=np.float64) / 1000.0
    VID = np.array([v for v, _ in ids])
    ALPHA, RRF_K, TAU = payload["alpha"], payload["rrf_k"], payload.get("tau", 1.0)
    out = []

    def hang(sc: np.ndarray, dung: np.ndarray) -> int:
        if not dung.any():
            return -1
        o = np.argsort(-sc, kind="stable")
        return int(np.flatnonzero(dung[o])[0]) + 1

    for i, q in enumerate(payload["queries"]):
        dung = (VID == q["video"]) & (TMS >= 0) & (np.abs(TMS - q["t"]) <= TAU)
        qv = np.asarray(q["qv"], dtype=np.float32)[:dim]
        qv /= max(float(np.linalg.norm(qv)), 1e-12)
        v = vis.score(qv)
        rec = {"id": q["id"], "co_trong_chi_muc": bool(dung.any()),
               "n_khung_dung": int(dung.sum())}
        hop = ALPHA["visual"] * rrf_normalize(v.scores, v.covered, RRF_K)
        rec["visual"] = hang(v.scores, dung)
        for s in tsrc:
            mr = q["exp"].get(s.name) or []
            runs = [s.score(q["text"])] + [s.score(x) for x in mr]
            acc = sum(rrf_normalize(r.scores, r.covered, RRF_K) / len(runs)
                      for r in runs)
            hop = hop + ALPHA[s.name] * acc
            rec[s.name] = hang(acc, dung)
        rec["hop"] = hang(hop, dung)
        out.append(rec)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(payload['queries'])} ({time.time() - t0:.0f}s)",
                  flush=True)
    return {"rows": out, "giay": round(time.time() - t0, 1)}


@app.function(image=image, volumes={"/idx": vol}, cpu=8.0, memory=65536, timeout=7200)
def tune_alpha(payload: dict) -> dict:
    """
    Quét `alpha` (③ tầng modality) và `beta` (③ tầng expansion) trên bộ có đáp án.

    MẸO LÀM CHO PHÉP QUÉT GẦN NHƯ MIỄN PHÍ. `hierarchical_rrf` là
    `Σ_m α_m·(Σ_j β_mj·rrf_j)`, và `rrf_j` **không phụ thuộc α, β**. Nên chấm BM25
    một lần mỗi truy vấn, giữ lại vector `rrf` của TỪNG RUN, rồi mọi ô lưới chỉ là
    vài phép nhân-cộng. Không có mẹo này thì mỗi ô lưới phải chấm lại 609.476 khung.

    Trả `rr` TỪNG CÂU để bên gọi bootstrap bắt cặp — trung bình suông không chứng
    minh được gì.
    """
    import json
    import sys
    import time

    import numpy as np
    sys.path.insert(0, "/root")

    from src.ingestion.vector_index import load_flat_index
    from src.retrieval.score_matrix import rrf_normalize
    from src.retrieval.sources import AsrSource, TextSource, VisualSource

    t0 = time.time()
    dim = payload.get("dim", 512)
    ix = load_flat_index("/idx/_modal_bundle", dim=dim)
    ids = list(ix.ids)
    ocr_map, fms = {}, {}
    with open("/idx/_modal_bundle/ocr_min.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            k = (d["v"], d["n"])
            if d["t"]:
                ocr_map[k] = d["t"]
            fms[k] = d["f"] / d["fps"] * 1000.0
    asr = {k: [(a, b, t) for a, b, t in v]
           for k, v in json.load(open("/idx/_modal_bundle/asr.json")).items()}
    vis = VisualSource(ix)
    tsrc = [TextSource("ocr", ids, ocr_map), AsrSource(ids, fms, asr)]
    TMS = np.array([fms.get((v, n), -1.0) for v, n in ids], dtype=np.float64) / 1000.0
    VID = np.array([v for v, _ in ids])
    RRF_K, TAU = payload["rrf_k"], payload.get("tau", 1.0)
    KS = tuple(payload.get("ks", (1, 5, 10, 20, 50, 100)))
    grid = payload["grid"]      # [{alpha:{...}, beta_goc: float|None}]
    ket = {json.dumps(g, sort_keys=True): {"rr": [], "hit": []} for g in grid}

    for qi, q in enumerate(payload["queries"]):
        qv = np.asarray(q["qv"], dtype=np.float32)[:dim]
        qv /= max(float(np.linalg.norm(qv)), 1e-12)
        v = vis.score(qv)
        # rrf CỦA TỪNG RUN — tính một lần, dùng cho mọi ô lưới
        R = {"visual": [rrf_normalize(v.scores, v.covered, RRF_K)]}
        for s in tsrc:
            mr = q["exp"].get(s.name) or []
            R[s.name] = [rrf_normalize(x.scores, x.covered, RRF_K)
                         for x in [s.score(q["text"])] + [s.score(y) for y in mr]]
        dung = (VID == q["video"]) & (TMS >= 0) & (np.abs(TMS - q["t"]) <= TAU)
        for g in grid:
            al, bg = g["alpha"], g.get("beta_goc")
            sc = np.zeros(ix.n_frames, np.float32)
            for m, runs in R.items():
                if not al.get(m):
                    continue
                if len(runs) == 1 or bg is None:
                    inner = sum(runs) / len(runs)      # ĐỀU
                else:
                    # `bg` cho câu gốc, phần còn lại chia đều cho các bản mở rộng
                    w = [bg] + [(1.0 - bg) / (len(runs) - 1)] * (len(runs) - 1)
                    inner = sum(wi * r for wi, r in zip(w, runs))
                sc = sc + al[m] * inner
            k = json.dumps(g, sort_keys=True)
            if dung.any():
                o = np.argsort(-sc, kind="stable")
                r = int(np.flatnonzero(dung[o])[0]) + 1
            else:
                r = 10 ** 9
            ket[k]["rr"].append(1.0 / r if r < 10 ** 9 else 0.0)
            ket[k]["hit"].append({kk: r <= kk for kk in KS})
        if (qi + 1) % 25 == 0:
            print(f"  {qi + 1}/{len(payload['queries'])} ({time.time() - t0:.0f}s)",
                  flush=True)

    return {"configs": {k: {"MRR": float(np.mean(x["rr"])),
                            **{f"R@{kk}": float(np.mean([h[kk] for h in x["hit"]]))
                               for kk in KS},
                            "rr_moi_cau": [round(float(y), 6) for y in x["rr"]],
                            "n": len(x["rr"])}
                        for k, x in ket.items()},
            "giay": round(time.time() - t0, 1)}
