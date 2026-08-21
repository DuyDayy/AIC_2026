"""
Chạy kiến trúc thi trên Modal cho bộ đề thử nghiệm BTC → bài nộp chuẩn thể lệ
=============================================================================

    [MODAL]  ① mã hoá → ②③ 7 run → ⑤a rổ
       │  gói nhỏ: rổ ≤300 khung + S[:, rổ] + văn bản OCR/ASR của 60 khung đầu
       ▼
    [MÁY]    ⑤c rerank (API) → ④ DANTE → ⑦ CSV + zip

Tách ở đây vì ①②③⑤a là phần tốn máy (BM25 trên 609.476 khung, `emb.npy` 1,2 GB
trong RAM) còn ⑤c⑦ thì không tốn máy nhưng cần khoá API và ổ đĩa của người dùng.
Trọng số ĐỀU cả hai tầng; λ mặc định 0,0 (giá trị đã quét).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval.dante import DEFAULT_LAMBDA                     # noqa: E402
from src.retrieval.probe import build_probes, declarativize        # noqa: E402
from src.retrieval.score_matrix import SourceScores, hierarchical_rrf   # noqa: E402
from src.submission.kbest import k_best_alignments                 # noqa: E402
from src.submission.writer import (                                # noqa: E402
    TaskSubmission, pack_submission_zip, task_type_from_filename, write_task_csv,
)

ALPHA = {"visual": 1 / 3, "ocr": 1 / 3, "asr": 1 / 3}
RRF_K, TOP_K = 60.0, 100
#: Mỗi video góp bao nhiêu đường TRAKE.
#:
#: `1` bỏ trống phần lớn ngân sách: [ĐO] ở rổ 300 chỉ điền 23/100 dòng, và thêm chặn
#: `min_gap=30` thì còn 11/100 — mà mọi ô tới 100 đều có trọng số dương (Định lý 2),
#: nên ô trống là điểm vứt đi.
#:
#: Nới `k` KHÔNG đụng tới R@1: đường thứ hai của một video luôn có điểm ≤ đường tốt
#: nhất của chính nó, nên đỉnh bảng không đổi. Nó chỉ chen vào các hạng dưới — có thể
#: đẩy tụt đường tốt nhất của video khác, đổi lại lấp được ô trống. Ở mức đang lấp
#: 11/100 thì phần lấp áp đảo phần đẩy tụt.
TRAKE_K_PER_VIDEO = 3
# Rổ = ĐÚNG top-100, và rerank chấm lại CẢ rổ.
#
# Vì sao bỏ rổ 300: ta chỉ nộp 100 dòng, nên khung xếp hạng 101–300 chỉ có giá trị
# nếu rerank kéo nó lên trong 100. Rerank thì chỉ chấm 60 khung đầu — tức 240 khung
# còn lại của rổ 300 KHÔNG bao giờ đổi được hạng, chúng chỉ tốn công dựng rổ.
# Cho rổ = 100 và rerank = cả 100 thì mọi ứng viên đều thật sự tranh được suất.
#
# ⚠️ TRAKE là NGOẠI LỆ, và đó là lý do cấu trúc chứ không phải giữ số cũ cho quen:
# ④ cần ≥N khung CÙNG một video trong rổ mới dựng nổi một đường. Rổ 100 trải trên
# hàng chục video thì rất ít video đủ N khung. [ĐO] ngay ở rổ 300 đã chỉ điền 23–29
# trên 100 dòng, mà mọi ô tới 100 đều có trọng số dương (Định lý 2) nên bỏ trống là
# vứt điểm. KIS/QA không dính vì mỗi dòng chỉ cần 1 khung.
POOL_CAP = {"kis": 100, "qa": 100, "trake": 300}
RERANK_W, RERANK_TOP, RERANK_BATCH = 0.25, 100, 10

#: Chặn cứng span giữa hai mốc TRAKE, đơn vị KHUNG.
#:
#: `min_gap = 1` chỉ đòi "tăng nghiêm ngặt", nên N mốc được phép nằm trong N keyframe
#: kề nhau — chưa tới một giây. [ĐO trên 50 câu TRAKE có ground truth]: bài nộp có
#: 62/178 = 35% cặp mốc liền nhau cách dưới 30 khung, còn GT có 0/178 (trung vị 181,
#: p10 74). Nên 30 là ngưỡng mà GT CHƯA TỪNG vi phạm: nó không loại bỏ chuỗi hợp lệ
#: nào, chỉ chặn kiểu dồn.
#:
#: Chặn cứng, KHÔNG phải phạt λ: `−λ(t−τ)` phạt theo khoảng cách tuyệt đối nên nó
#: cộng cùng một handicap cho mọi đường span lớn ở MỌI video, tức can thiệp vào xếp
#: hạng video chứ không chỉ định hình đường trong một video. Chặn cứng không bóp méo
#: điểm nên không có tác dụng phụ đó — chính docstring của λ ghi đây là hướng chưa thử.
MIN_GAP = 30

sys.path.insert(0, str(ROOT / "scripts"))
from run_thunghiem import nap_de, nap_mo_rong, rerank_api          # noqa: E402


def cat_khung(video_id: str, frame_idx: int, root: str = "data/video") -> bytes | None:
    """
    Cắt ĐÚNG khung `frame_idx` khỏi video gốc → JPEG. `None` nếu không có file.

    Vì sao không lấy từ kho keyframe: `aic-keyframes` đánh số theo `n` của bộ trích
    xuất CŨ — [ĐO] 601 ảnh cho `L21_V001` trong khi chỉ mục hiện tại có 1.778 khung và
    `n` chạy tới 3.711. Tra theo `n` ở đó cho ra ẢNH KHÁC, và không có gì báo. Video
    gốc thì chỉ có một cách hiểu: khung thứ `frame_idx`.

    `select='eq(n\\,K)'` đếm khung ĐÃ GIẢI MÃ từ 0 — cùng quy ước với `frame_idx`
    của ta (đã kiểm: δ = 0 trên 609.476/609.476 dòng, và ffprobe cho khung đầu
    pts_time=0.000000).
    """
    import subprocess
    p = Path(root) / f"{video_id}.mp4"
    if not p.is_file():
        return None
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(p),
         "-vf", f"select=eq(n\\,{int(frame_idx)}),scale='min(896,iw)':-2",
         "-vsync", "vfr", "-frames:v", "1", "-q:v", "3", "-f", "image2pipe",
         "-vcodec", "mjpeg", "-"],
        capture_output=True)
    return r.stdout or None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="THUNGHIEM-bo-de-thi")
    ap.add_argument("--expansions", default="expansions")
    ap.add_argument("--out", default="submission_thunghiem")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--lam", type=float, default=DEFAULT_LAMBDA)
    ap.add_argument("--min-gap", type=int, default=MIN_GAP)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--no-qa", action="store_true", help="bỏ ⑥ sinh đáp án Q&A")
    ap.add_argument("--video-root", default="data/video")
    a = ap.parse_args()

    import modal

    t0 = time.time()
    Q = nap_de(Path(a.dir))
    EXP = nap_mo_rong(Path(a.expansions))
    print(f"① {len(Q)} đề: "
          + " · ".join(f"{k} {sum(1 for q in Q if q['kind'] == k)}"
                       for k in ("kis", "qa", "trake"))
          + f" · mở rộng phủ {len(EXP)}/{len(Q)}")

    for q in Q:
        src = declarativize(q["text"]) if q["kind"] == "qa" else q["text"]
        q["probes"] = [p.text for p in build_probes(src)] or [src]
        q["gop"] = q["kind"] in ("kis", "qa")
    can = sorted({t for q in Q for t in q["probes"]}
                 | {x for q in Q for v in EXP.get(q["id"], {}).values() for x in v})

    fn_enc = modal.Function.from_name("aic-query", "encode_text")
    fn_sc = modal.Function.from_name("aic-query", "score")
    print(f"① mã hoá {len(can)} đoạn trên A10G…", flush=True)
    vecs = dict(zip(can, fn_enc.remote(can)))
    print(f"① xong {time.time() - t0:.0f}s", flush=True)

    payload = {
        "alpha": ALPHA, "rrf_k": RRF_K, "pool_cap": POOL_CAP["kis"],
        "rerank_top": RERANK_TOP, "dim": a.dim,
        "queries": [{"id": q["id"], "probes": q["probes"], "gop": q["gop"],
                     "cap": POOL_CAP[q["kind"]],
                     "exp": EXP.get(q["id"], {}),
                     "qv": {t: vecs[t] for t in q["probes"]}} for q in Q],
    }
    print("②③⑤a chấm trên Modal (8 CPU · 32 GB) · rổ "
          + " · ".join(f"{k} {v}" for k, v in POOL_CAP.items()), flush=True)
    res = fn_sc.remote(payload)
    R = {x["id"]: x for x in res["queries"]}
    print(f"②③⑤a xong · Modal {res['giay']}s · {res['n_frames']:,} khung "
          f"· tổng {time.time() - t0:.0f}s", flush=True)

    for q in Q:
        r = R[q["id"]]
        q["pool"] = np.asarray(r["pool"], dtype=np.int64)
        q["Sp"] = np.asarray(r["S_pool"], dtype=np.float32)   # (n_mốc, |rổ|)
        q["vid"] = r["vid"]
        q["fi"] = r["fi"]
        q["rr_rows"] = r["rerank_rows"]
        q["ctx"] = r["ctx"]

    # ── ⑤c rerank bằng API, hợp vào nền bằng RRF TẦNG BA ───────────────────
    if not a.no_rerank:
        jobs = [{"key": f"{q['id']}#{r}", "query": q["text"],
                 "ocr": q["ctx"][str(r)]["ocr"], "asr": q["ctx"][str(r)]["asr"]}
                for q in Q for r in q["rr_rows"]]
        print(f"⑤c rerank API: {len(jobs)} khung, "
              f"~{len(jobs) // RERANK_BATCH + 1} lời gọi…", flush=True)
        diem = rerank_api(jobs, a.model)
        print(f"⑤c API chấm {len(diem)}/{len(jobs)} khung "
              f"({time.time() - t0:.0f}s)", flush=True)
        for q in Q:
            vi = {int(r): i for i, r in enumerate(q["pool"].tolist())}
            m = len(q["pool"])
            sc = np.zeros(m, dtype=np.float32)
            cov = np.zeros(m, dtype=bool)
            for r in q["rr_rows"]:
                d = diem.get(f"{q['id']}#{r}")
                if d is not None and int(r) in vi:
                    sc[vi[int(r)]] = d
                    cov[vi[int(r)]] = True
            if not cov.any():
                continue
            rr = SourceScores("rerank", sc, cov)
            aw = {"fused": 1.0 / (1.0 + RERANK_W), "rerank": RERANK_W / (1.0 + RERANK_W)}
            moi = [hierarchical_rrf(
                {"fused": [SourceScores("fused", hang, np.ones(m, dtype=bool))],
                 "rerank": [rr]}, alpha=aw,
                beta={"fused": (1.0,), "rerank": (1.0,)}, k=RRF_K) for hang in q["Sp"]]
            q["Sp"] = np.vstack(moi).astype(np.float32)
    else:
        print("⑤c BỎ QUA (--no-rerank)")

    # ── ④ DANTE ────────────────────────────────────────────────────────────
    for q in Q:
        n_mom = q["Sp"].shape[0]
        if n_mom == 1:
            order = np.argsort(-q["Sp"][0], kind="stable")
            lines, seen = [], set()
            for j in order.tolist():
                k = (q["vid"][j], int(q["fi"][j]))
                if k in seen:
                    continue
                seen.add(k)
                lines.append((k[0], (k[1],)))
                if len(lines) >= TOP_K:
                    break
        else:
            byv = {}
            for j in range(len(q["pool"])):
                byv.setdefault(q["vid"][j], []).append(j)
            sc = []
            for vid, cols in byv.items():
                cols = sorted(set(cols), key=lambda j: int(q["fi"][j]))
                seen_f, keep = set(), []
                for j in cols:
                    f = int(q["fi"][j])
                    if f not in seen_f:
                        seen_f.add(f)
                        keep.append(j)
                if len(keep) < n_mom:
                    continue
                cand = [int(q["fi"][j]) for j in keep]
                for al in k_best_alignments(cand, q["Sp"][:, keep].tolist(),
                                            k=min(TRAKE_K_PER_VIDEO, len(keep)),
                                            min_gap=a.min_gap, pacing_penalty=a.lam):
                    sc.append((float(al.score), vid, tuple(int(f) for f in al.frames)))
            sc.sort(key=lambda x: (-x[0], x[1], x[2]))
            lines = [(v, f) for _s, v, f in sc[:TOP_K]]

        q["lines"] = lines

    # ── ⑥ ĐÁP ÁN Q&A — Qwen2.5-VL trên Modal, ảnh cắt từ VIDEO GỐC ──────────
    qa = [q for q in Q if q["kind"] == "qa"]
    for q in qa:
        q["answer"] = ""
    if qa and not a.no_qa:
        import base64
        jobs, ai = [], []
        for q in qa:
            if not q["lines"]:
                continue
            vid, fr = q["lines"][0]
            b = cat_khung(vid, fr[0], a.video_root)
            if b is None:
                print(f"  ⚠ {q['id']}: không có {vid}.mp4 hoặc cắt khung hỏng",
                      file=sys.stderr)
                continue
            # ngữ cảnh chữ của ĐÚNG khung đó, lấy từ gói Modal đã trả về
            ctx = R[q["id"]]["ctx"].get(str(R[q["id"]]["rerank_rows"][0]), {}) \
                if R[q["id"]]["rerank_rows"] else {}
            jobs.append({"image_b64": base64.b64encode(b).decode(),
                         "question": q["text"],
                         "ocr": ctx.get("ocr", ""), "asr": ctx.get("asr", "")})
            ai.append(q)
        if jobs:
            print(f"⑥ Qwen2.5-VL trên Modal: {len(jobs)} đề Q&A…", flush=True)
            fn_qa = modal.Function.from_name("aic-query", "qa_answer")
            for q, ans in zip(ai, fn_qa.remote(jobs)):
                q["answer"] = ans
                print(f"   {q['id']}: {ans!r}")
    elif qa:
        print("⑥ BỎ QUA (--no-qa) — cột đáp án sẽ là CHUA_SINH")

    # ── ⑦ ghi bài nộp theo THỂ LỆ ──────────────────────────────────────────
    odir = Path(a.out) / "submission"
    bao = []
    for q in Q:
        lines, n_mom = q["lines"], q["Sp"].shape[0]
        ans = (q.get("answer") or "CHUA_SINH") if q["kind"] == "qa" else None
        sub = TaskSubmission(task_id=q["id"], task_type=task_type_from_filename(q["id"]),
                             answers=tuple((v, f, ans) for v, f in lines),
                             n_moments=n_mom)
        write_task_csv(sub, odir, budget=TOP_K)
        bao.append({"id": q["id"], "kind": q["kind"], "n_moc": n_mom,
                    "n_dong": len(lines), "answer": ans,
                    "top1": [lines[0][0], list(lines[0][1])] if lines else None})
        print(f"  ⑦ {q['id']:<22} {q['kind']:<6} {len(lines):>3} dòng"
              + (f" · top1 {lines[0][0]}@{lines[0][1][0]}" if lines else " · RỖNG"))

    z = pack_submission_zip(odir, Path(a.out) / "submission.zip",
                            expected=[q["id"] for q in Q])
    (Path(a.out) / "_report.json").write_text(json.dumps({
        "kien_truc": {"alpha": ALPHA, "beta": "ĐỀU 1/n trong từng modality",
                      "rrf_k": RRF_K, "pool_cap": POOL_CAP,
                      "rerank": "TẮT" if a.no_rerank else f"API {a.model}, w={RERANK_W}",
                      "lambda": a.lam, "min_gap": a.min_gap,
                      "trake_k_per_video": TRAKE_K_PER_VIDEO,
                      "qa": "TẮT" if a.no_qa else "Qwen2.5-VL @ Modal, ảnh từ video gốc",
                      "chay_o": "Modal (①②③⑤a) + máy (⑤c④⑦)"},
        "truy_van": bao, "giay": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n✓ {len(Q)} CSV → {z} ({z.stat().st_size / 1e3:.0f} KB) "
          f"· {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
