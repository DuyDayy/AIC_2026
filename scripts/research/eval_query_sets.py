"""
Eval ba bộ đề sinh sẵn (KIS 199 · QA 50 · TRAKE 50) trên chỉ mục 609.476 khung
=============================================================================

Dùng ĐÚNG các module của đường chạy thi — `hierarchical_rrf`, `fused_pool`,
`k_best_alignments` — nên số đo ra nói về hệ thật, không về một bản mô phỏng.

KHÁC đường chạy thi ở ba chỗ, đều do đề bài:
  · KHÔNG rerank (⑤b/⑤c tắt) — không gọi GPU cho VLM
  · DANTE chỉ chạy cho TRAKE; KIS/QA lấy thẳng top-K của điểm ③
  · đáp án Q&A sinh bằng API của đội (MiraiAPI), KHÔNG bằng Qwen-VL

Vì sao chấm theo DUNG SAI THỜI GIAN, không theo `frame_idx` khớp đúng
--------------------------------------------------------------------
Ground truth của ba bộ này dựng trên một bản trích xuất KHÁC bản đang chạy.
[ĐO] chỉ 12/199 khung GT của KIS có mặt y nguyên trong chỉ mục; và `n` thì
KHÔNG tương ứng chút nào — 99/199 câu "khớp" theo `(video, n)` nhưng 0/99
trong số đó trùng `frame_idx`, 0/99 trùng `shot_id`. Khớp theo `n` là trùng
hợp thuần tuý, dùng nó sẽ cho một bảng điểm đẹp và sai.

Nhưng MỌI khung GT đều có khung trong chỉ mục cách ≤28 khung (trung vị 7,
tức ~0,25 giây). Nên phép chấm đúng là cửa sổ thời gian — cùng dạng với
`R(rᵢ) = I(vᵢ = GTᵥ ∧ idᵢ ∈ [s, e])` của thể lệ, chỉ khác là ta phải tự
định nghĩa `[s, e]` vì BTC chưa cho. Dùng `pts_time` nên không phụ thuộc
`fps` — kho này có 4 giá trị fps khác nhau (25 · 29,97 · 30 · 26,438).

TRAKE là ngoại lệ: GT của nó CÓ `gt_start_frame`/`gt_end_frame` sẵn, nên
chấm đúng nguyên văn thể lệ `R = (1/N)·Σⱼ I(id_j ∈ [sⱼ, eⱼ])`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ingestion.jina_encoder import truncate_and_normalize          # noqa: E402
from src.ingestion.vector_index import load_flat_index                 # noqa: E402
from src.retrieval.pool import fused_pool                              # noqa: E402
from src.retrieval.probe import build_probes                           # noqa: E402
from src.retrieval.score_matrix import hierarchical_rrf                # noqa: E402
from src.retrieval.sources import (                                    # noqa: E402
    AsrSource, TextSource, VisualSource, load_asr_segments, load_frame_ms,
    load_ocr_text,
)
from src.submission.kbest import k_best_alignments                     # noqa: E402
from src.submission.writer import (                                    # noqa: E402
    TaskSubmission, pack_submission_zip, write_task_csv,
)

ALPHA = {"visual": 1 / 3, "ocr": 1 / 3, "asr": 1 / 3}
RRF_K = 60.0
POOL_CAP = 300
TOP_K = 100
K_MOCS = (1, 5, 10, 20, 50, 100)
#: Dung sai chính. 1,0 giây ≈ 25–30 khung; trần của chỉ mục đạt 100% ở ~28 khung.
TAU_CHINH = 1.0
TAU_QUET = (0.5, 1.0, 2.0, 5.0)


# ============================================================
# NẠP ĐỀ
# ============================================================


def nap_kis(d: Path) -> list[dict]:
    g = {r["query_id"]: r for r in csv.DictReader(
        open(d / "kis_ground_truth_500.csv", encoding="utf-8"))}
    out = []
    for p in sorted((d / "queries").glob("*.txt")):
        qid = "KIS_" + p.stem.rsplit("-", 1)[-1]
        r = g.get(qid)
        if r is None:
            continue
        out.append({"id": p.stem, "kind": "kis", "text": p.read_text(encoding="utf-8").strip(),
                    "video": r["video_id"], "t": float(r["pts_time"])})
    return out


def nap_qa(d: Path) -> list[dict]:
    g = {r["qa_id"]: r for r in
         (json.loads(l) for l in open(d / "qa_ground_truth_50.jsonl", encoding="utf-8"))}
    out = []
    for p in sorted((d / "queries").glob("*.txt")):
        qid = "QA" + p.stem.rsplit("-", 1)[-1].zfill(3)
        r = g.get(qid)
        if r is None:
            continue
        out.append({"id": p.stem, "kind": "qa", "text": p.read_text(encoding="utf-8").strip(),
                    "video": r["video_id"], "t": float(r["anchor_pts_time"]),
                    "dap_an": r["answer"], "cau_hoi": r.get("question", "")})
    return out


def nap_trake(d: Path) -> list[dict]:
    g = {r["trake_id"]: r for r in
         (json.loads(l) for l in open(d / "trake_ground_truth_50.jsonl", encoding="utf-8"))}
    out = []
    for p in sorted((d / "queries").glob("*.txt")):
        qid = "TRK" + p.stem.rsplit("-", 1)[-1].zfill(3)
        r = g.get(qid)
        if r is None:
            continue
        out.append({"id": p.stem, "kind": "trake",
                    "text": p.read_text(encoding="utf-8").strip(), "video": r["video_id"],
                    "cua_so": [(int(e["gt_start_frame"]), int(e["gt_end_frame"]))
                               for e in r["events"]]})
    return out


# ============================================================
# CHẤM
# ============================================================


def recall_mrr(hits: list[list[bool]]) -> dict:
    """`hits[i]` = danh sách đúng/sai theo thứ hạng của truy vấn i."""
    out = {}
    for k in K_MOCS:
        out[f"R@{k}"] = float(np.mean([any(h[:k]) for h in hits]))
    mrr = []
    for h in hits:
        r = next((i + 1 for i, v in enumerate(h) if v), None)
        mrr.append(1.0 / r if r else 0.0)
    out["MRR"] = float(np.mean(mrr))
    out["n"] = len(hits)
    return out


# ============================================================
# ĐÁP ÁN Q&A — API của đội, KHÔNG phải Qwen-VL
# ============================================================


def sinh_dap_an(jobs: list[dict], model: str, workers: int = 6) -> dict[str, str]:
    """
    Sinh đáp án Q&A bằng MiraiAPI từ NGỮ CẢNH VĂN BẢN quanh khung dự đoán.

    ⚠️ KHÔNG có ảnh. Bộ keyframe không nằm trên đĩa (GT trỏ tới zip
    `Keyframes L21-L25-001` mà kho này không có), nên đường đọc ảnh của ⑥ không
    chạy được. Đây là cận DƯỚI: mọi câu hỏi cần nhìn hình — màu áo, số người
    trong khung — sẽ hỏng, và chúng hỏng vì THIẾU DỮ LIỆU chứ không vì mô hình.
    Câu hỏi trả lời được từ chữ trên màn hoặc lời dẫn thì vẫn đúng.
    """
    from concurrent.futures import ThreadPoolExecutor

    from openai import OpenAI

    client = OpenAI(base_url="https://sv.devquote.shop/v1",
                    api_key=os.environ.get("MIRAI_API_KEY", "") or "dummy",
                    timeout=90.0, max_retries=2)

    HE = ("Bạn đọc ngữ cảnh trích từ một khung hình bản tin tiếng Việt và trả lời câu hỏi. "
          "Chỉ trả về ĐÁP ÁN, ngắn nhất có thể, tối đa 100 ký tự. Không giải thích, "
          "không lặp lại câu hỏi. Nếu ngữ cảnh không đủ, đoán đáp án hợp lý nhất.")

    def mot(j):
        msg = (f"CHỮ TRÊN MÀN (OCR):\n{j['ocr'][:1500]}\n\n"
               f"LỜI DẪN (ASR):\n{j['asr'][:1500]}\n\n"
               f"CÂU HỎI: {j['cau_hoi']}\nĐÁP ÁN:")
        for _ in range(3):
            try:
                r = client.chat.completions.create(
                    model=model, max_tokens=256,
                    messages=[{"role": "system", "content": HE},
                              {"role": "user", "content": msg}])
                t = (r.choices[0].message.content or "").strip()
                if t:
                    return j["id"], " ".join(t.split())[:100]
            except Exception:       # proxy này đã sập giữa buổi trước đây
                continue
        return j["id"], ""

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return dict(ex.map(mot, jobs))


# ============================================================
# CHẠY
# ============================================================


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdir", default="data/queries")
    ap.add_argument("--out", default="data/research/eval_gen299")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--dim", type=int, default=512)
    # HOÃN Q&A: mặc định KHÔNG gọi API. Đường đọc ảnh của ⑥ chưa chạy được vì bộ
    # keyframe chưa có trên đĩa; sinh đáp án từ mỗi OCR+ASR sẽ cho một con số nói về
    # "đoán được bao nhiêu từ chữ", không nói về hệ thật. Phần ĐỊNH VỊ khung của Q&A
    # thì không cần ảnh nên vẫn đo, và đo miễn phí.
    ap.add_argument("--qa-api", action="store_true",
                    help="bật gọi API sinh đáp án Q&A (mặc định TẮT — chờ có keyframe)")
    a = ap.parse_args()

    qd = Path(a.qdir)
    Q = nap_kis(qd / "kis_gen500") + nap_qa(qd / "qa_gen50") + nap_trake(qd / "trake_gen50")
    print(f"nạp {len(Q)} đề: "
          + " · ".join(f"{k} {sum(1 for q in Q if q['kind'] == k)}"
                       for k in ("kis", "qa", "trake")))

    t0 = time.time()
    idx = load_flat_index("data/embed", dim=a.dim)
    FI = np.asarray(idx.frame_idx)
    VID = np.array([v for v, _ in idx.ids])
    fms = load_frame_ms()
    TMS = np.array([fms.get((v, int(n)), -1.0) for v, n in idx.ids], dtype=np.float64)
    vis = VisualSource(idx)
    ocr_map = load_ocr_text("data/OCR/ocr.jsonl")
    tsrc = [TextSource("ocr", idx.ids, ocr_map),
            AsrSource(idx.ids, fms, load_asr_segments("data/ASR"))]
    print(f"nạp chỉ mục {time.time() - t0:.0f}s · {idx.n_frames:,} khung · "
          f"{len(idx.ranges)} video · mốc thời gian {int((TMS >= 0).sum()):,}")

    # ── ① probe ────────────────────────────────────────────────────────────
    for q in Q:
        q["probes"] = [p.text for p in build_probes(q["text"])] or [q["text"]]
        # KIS/QA nộp MỘT khung mỗi dòng ⟹ ma trận điểm phải có đúng một hàng.
        q["gop"] = q["kind"] in ("kis", "qa")
    can = sorted({t for q in Q for t in q["probes"]})
    print(f"① {len(can)} đoạn cần mã hoá "
          f"(TRAKE trung bình {np.mean([len(q['probes']) for q in Q if q['kind']=='trake']):.1f} mốc)")

    # ── mã hoá TẠI MÁY, có ghim revision ───────────────────────────────────
    #
    # Không dùng Modal: 476 đoạn là quá ít để đáng một vòng GPU thuê, và mô hình đã
    # nằm sẵn trong HF cache (1,6 GB). MPS trên máy làm xong trong vài phút, $0.
    #
    # GHIM `revision`/`code_revision` KHÔNG phải cẩn thận thừa. `from_pretrained` không
    # ghim sẽ kéo bản mã remote MỚI NHẤT — nó đã tự tải về ngay trong lần thử đầu. Vector
    # khung mã hoá bằng `e10d47f5…` + mã `39e6a55a…` (artifacts/embed/embed/manifest.json,
    # và `PipelineConfig` của frame_extracting dùng đúng cặp đó). Truy vấn mã hoá bằng
    # bản khác thì hai bên nằm ở hai không gian hơi lệch nhau — cosine vẫn ra số, thứ
    # hạng vẫn sắp được, và không có gì báo.
    import hashlib

    MODEL_SHA = "e10d47f5691d0454a0fb5d13f46f2199b74cb436"
    CODE_SHA = "39e6a55ae971b59bea6e44675d237c99762e7ee2"

    cache_p = Path("data/embed/query_cache.npz")
    cache = dict(np.load(cache_p)) if cache_p.is_file() else {}
    key = lambda t: hashlib.sha1(t.encode()).hexdigest()[:16]      # noqa: E731
    thieu = [t for t in can if key(t) not in cache]
    if thieu:
        import torch
        from transformers import AutoModel
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"  nạp jina-clip-v2 @ {MODEL_SHA[:8]} trên {dev}…", flush=True)
        m = AutoModel.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True,
                                      revision=MODEL_SHA, code_revision=CODE_SHA).eval().to(dev)
        print(f"  mã hoá {len(thieu)} đoạn (đã cache {len(cache)})…", flush=True)
        B = 32
        for i in range(0, len(thieu), B):
            lo = thieu[i:i + B]
            with torch.inference_mode():
                v = np.asarray(m.encode_text(lo, batch_size=B), dtype=np.float32)
            v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
            for t, x in zip(lo, v):
                cache[key(t)] = x
            print(f"    {min(i + B, len(thieu))}/{len(thieu)}  ({time.time() - t0:.0f}s)",
                  flush=True)
        np.savez(cache_p, **cache)
        del m
    QV = {t: truncate_and_normalize(np.asarray([cache[key(t)]], np.float32), a.dim)[0]
          for t in can}
    print(f"① xong {time.time() - t0:.0f}s")

    # ── ②③ chấm ba nguồn, hợp bằng RRF hai tầng ────────────────────────────
    for i, q in enumerate(Q):
        rows = []
        for t in q["probes"]:
            runs = {"visual": [vis.score(QV[t])]}
            for src in tsrc:
                runs[src.name] = [src.score(t)]
            rows.append(hierarchical_rrf(runs, alpha=ALPHA,
                                         beta={m: (1.0,) for m in runs}, k=RRF_K))
        S = np.vstack(rows).astype(np.float32)
        if q["gop"] and S.shape[0] > 1:
            S = S.max(axis=0, keepdims=True)
        q["S"] = S
        if (i + 1) % 25 == 0:
            print(f"  ②③ {i + 1}/{len(Q)}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"②③ xong {time.time() - t0:.0f}s")

    # ── ⑤a rổ · ④ DANTE (chỉ TRAKE) · ⑦ danh sách top-100 ──────────────────
    for q in Q:
        n_mom = q["S"].shape[0]
        if n_mom == 1:
            # KIS/QA: lấy thẳng top-K của điểm ③, khử trùng (video, frame_idx)
            order = np.argsort(-q["S"][0], kind="stable")
            lines, seen = [], set()
            for r in order[: TOP_K * 6].tolist():
                k = (str(VID[r]), int(FI[r]))
                if k in seen:
                    continue
                seen.add(k)
                lines.append((k[0], (k[1],), r))
                if len(lines) >= TOP_K:
                    break
        else:
            pool = fused_pool(q["S"], POOL_CAP)
            byv = {}
            for r in pool.tolist():
                byv.setdefault(str(VID[r]), []).append(r)
            sc = []
            for vid, rr in byv.items():
                rr = sorted(set(rr), key=lambda r: int(FI[r]))
                seen_f, keep = set(), []
                for r in rr:
                    f = int(FI[r])
                    if f not in seen_f:
                        seen_f.add(f)
                        keep.append(r)
                if len(keep) < n_mom:
                    continue
                cand = [int(FI[r]) for r in keep]
                for al in k_best_alignments(cand, q["S"][:, keep].tolist(), k=1):
                    sc.append((float(al.score), vid, tuple(int(f) for f in al.frames)))
            sc.sort(key=lambda x: (-x[0], x[1], x[2]))
            lines = [(v, f, None) for _s, v, f in sc[:TOP_K]]
        q["lines"] = lines
    print(f"⑤④⑦ xong {time.time() - t0:.0f}s")

    # ── ⑥ đáp án Q&A bằng API của đội ──────────────────────────────────────
    qa = [q for q in Q if q["kind"] == "qa"]
    if qa and a.qa_api:
        jobs = []
        for q in qa:
            r = q["lines"][0][2]
            vid, n = idx.ids[r]
            ms = TMS[r]
            asr_txt = " ".join(
                t for x, y, t in tsrc[1].segments.get(str(vid), [])
                if ms >= 0 and x - 5000 <= ms <= y + 5000)
            jobs.append({"id": q["id"], "cau_hoi": q["cau_hoi"] or q["text"],
                         "ocr": ocr_map.get((str(vid), int(n)), ""), "asr": asr_txt})
        print(f"⑥ gọi API sinh {len(jobs)} đáp án ({a.model})…", flush=True)
        ans = sinh_dap_an(jobs, a.model)
        for q in qa:
            q["answer"] = ans.get(q["id"], "")
        n_rong = sum(1 for q in qa if not q["answer"])
        print(f"⑥ xong · {len(qa) - n_rong}/{len(qa)} có đáp án"
              + (f" · ⚠ {n_rong} rỗng" if n_rong else ""))
    else:
        # Chỗ giữ chỗ, KHÔNG phải đáp án. Nó chỉ tồn tại để file CSV Q&A đủ ba cột
        # đúng thể lệ; phép chấm đáp án bị bỏ hẳn khỏi bảng, không chấm bằng nó.
        for q in qa:
            q["answer"] = "CHUA_SINH"
        print("⑥ BỎ QUA — chờ bộ keyframe. Chỉ đo ĐỊNH VỊ khung của Q&A.")

    # ── ⑦ ghi bài nộp theo THỂ LỆ ──────────────────────────────────────────
    out = Path(a.out); sub = out / "submission"
    for q in Q:
        s = TaskSubmission(
            task_id=q["id"], task_type=q["kind"],
            answers=tuple((v, f, q.get("answer")) for v, f, _r in q["lines"]),
            n_moments=q["S"].shape[0])
        write_task_csv(s, sub, budget=TOP_K)
    z = pack_submission_zip(sub, out / "submission.zip")
    print(f"⑦ {len(Q)} file CSV → {z}")

    # ── CHẤM ───────────────────────────────────────────────────────────────
    bang = {}
    for kind in ("kis", "qa"):
        nhom = [q for q in Q if q["kind"] == kind]
        for tau in TAU_QUET:
            hits = []
            for q in nhom:
                h = []
                for v, f, r in q["lines"]:
                    t = TMS[r] / 1000.0 if r is not None else -1
                    h.append(v == q["video"] and t >= 0 and abs(t - q["t"]) <= tau)
                hits.append(h)
            bang[f"{kind} τ={tau}s"] = recall_mrr(hits)
        if kind == "qa" and a.qa_api:
            from src.scoring.rscore import normalize_answer
            hits = []
            for q in nhom:
                dung = normalize_answer(q.get("answer", "")) == normalize_answer(q["dap_an"])
                h = []
                for v, f, r in q["lines"]:
                    t = TMS[r] / 1000.0 if r is not None else -1
                    h.append(dung and v == q["video"] and t >= 0
                             and abs(t - q["t"]) <= TAU_CHINH)
                hits.append(h)
            bang[f"qa τ={TAU_CHINH}s + ĐÁP ÁN"] = recall_mrr(hits)

    nhom = [q for q in Q if q["kind"] == "trake"]
    for nguong in (1.0, 0.5):          # 1,0 = trúng HẾT mốc; 0,5 = trúng nửa
        hits = []
        for q in nhom:
            h = []
            for v, fr, _r in q["lines"]:
                if v != q["video"] or len(fr) != len(q["cua_so"]):
                    h.append(False); continue
                r = sum(s <= f <= e for f, (s, e) in zip(fr, q["cua_so"])) / len(q["cua_so"])
                h.append(r >= nguong)
            hits.append(h)
        bang[f"trake R≥{nguong}"] = recall_mrr(hits)

    (out / "metrics.json").write_text(
        json.dumps(bang, ensure_ascii=False, indent=1), encoding="utf-8")

    print()
    print(f"{'bộ':<26} {'n':>4} " + " ".join(f"{'R@' + str(k):>7}" for k in K_MOCS)
          + f" {'MRR':>8}")
    print("─" * 82)
    for k, v in bang.items():
        print(f"{k:<26} {v['n']:>4} "
              + " ".join(f"{v['R@'+str(kk)]:>7.3f}" for kk in K_MOCS)
              + f" {v['MRR']:>7.4f}")
    print()
    print(f"✓ {out}/metrics.json · {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
