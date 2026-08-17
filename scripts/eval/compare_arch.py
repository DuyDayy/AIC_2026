#!/usr/bin/env python3
"""
So KIẾN TRÚC MỚI (rổ ứng viên + rerank 2 bậc) với kiến trúc cũ (hợp điểm)
=========================================================================

    python scripts/eval/compare_arch.py --new submission_new --old submission_bench

Chấm bằng `src/scoring/rscore.py` — nguyên văn thể lệ. Báo theo **mô hình mốc lệch
ngẫu nhiên**, vì hai mô hình kia ôm keyframe ground truth nên tự thưởng cho việc nộp lại
chính cái nhãn của mình (xem README, *Bài học đo lường*).

Có `_rerank_scores.npz` thì quét lại **trọng số rerank mà không chạy lại GPU**: bản lưu
giữ rổ + điểm nền + điểm mảnh cắt + điểm VLM cho từng truy vấn.

⚠️ GIỚI HẠN của phần suy: `fuse` chuẩn hoá z trên **toàn kho** 173.426 khung, còn npz chỉ
lưu phần TRONG RỔ. Nên `z_normalize` ở đây tính trên rổ, khác thang với bài chạy thật.
Tỉ lệ giữa các trọng số vẫn so được với nhau, nhưng con số Final tuyệt đối KHÔNG so ngang
với bài nộp thật — muốn số thật thì phải chạy `scripts/run.py`.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ingestion.vector_index import load_flat_index
from src.retrieval.score_matrix import z_normalize
from src.retrieval.sources import load_shot_bounds, load_video_last_frame
from src.scoring.rscore import Interval, KISAnswer, KISGroundTruth, final_score, stable_seed
from src.submission.writer import SubmissionError, TaskSubmission, validate_all

SEEDS = 32
HALF = 4          # L = 9, dải thể lệ nêu ("thường dưới 10 frame")


def read_sub(d: Path) -> dict[str, list[tuple[str, int]]]:
    out = {}
    for p in sorted(d.glob("*.csv")):
        out[p.stem.split("-")[-1]] = [(r[0], int(r[1])) for r in
                                      csv.reader(open(p, encoding="utf-8")) if r and r[0]]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", type=Path, default=Path("submission_new"))
    ap.add_argument("--old", type=Path, default=Path("submission_bench"))
    ap.add_argument("--gt", type=Path,
                    default=Path("export_for_fusion/benchmark_queries.json"))
    ap.add_argument("--index", type=Path, default=Path("data/embed"))
    args = ap.parse_args()

    gt = json.loads(args.gt.read_text(encoding="utf-8"))
    idx = load_flat_index(args.index, dim=512)
    FI = np.asarray(idx.frame_idx)
    VID = np.array([v for v, _ in idx.ids])
    pos = {k: i for i, k in enumerate(idx.ids)}
    shot_b, last_f = load_shot_bounds(), load_video_last_frame()

    win: dict[int, tuple[int, int]] = {}
    for v, (lo, hi) in idx.ranges.items():
        o = np.argsort(FI[lo:hi])
        f = FI[lo:hi][o]
        vmax = last_f.get(v, int(f[-1])) - 1
        for j, rr in enumerate(o):
            r = lo + int(rr)
            c = int(f[j])
            prev = int(f[j - 1]) if j else c - 71
            nxt = int(f[j + 1]) if j + 1 < len(f) else c + 71
            a, b = c - (c - prev) // 2, c + (nxt - c) // 2
            ss, se = shot_b.get(idx.ids[r], (a, b))
            win[r] = (max(0, a, ss), min(b, se, vmax))

    # Mốc ngữ nghĩa THẬT: rơi đều trong khe quanh keyframe, cùng hạt giống mọi phương án
    jit: dict[str, list[int]] = {}
    for r in gt:
        g = pos[(r["video_id"], int(r["frame_id"]))]
        wl, wh = win[g]
        rng = np.random.default_rng(stable_seed(r["query_id"]))
        jit[r["query_id"]] = [int(rng.integers(wl, wh + 1)) for _ in range(SEEDS)]

    def score_rows(qid: str, vid_gt: str, rows) -> float:
        ans = [KISAnswer(v, f) for v, f in rows]
        return float(np.mean([final_score(ans, KISGroundTruth(
            vid_gt, Interval(t - HALF, t + HALF))).final for t in jit[qid]]))

    # ── 1. KIỂM ĐỊNH DẠNG cả hai bài nộp ────────────────────────────────────
    bounds = {v: int(FI[lo:hi].max()) + 1 for v, (lo, hi) in idx.ranges.items()}
    for lab, d in (("mới", args.new), ("cũ", args.old)):
        subs = read_sub(d)
        if not subs:
            print(f"✗ {d}/ rỗng")
            return 1
        tasks = [TaskSubmission(task_id=k, task_type="kis",
                                answers=tuple((v, (f,), None) for v, f in rows),
                                n_moments=1) for k, rows in subs.items()]
        try:
            validate_all(tasks, budget=100, frame_bounds=bounds)
            print(f"✓ {lab}: định dạng hợp lệ · {len(tasks)} truy vấn")
        except SubmissionError as e:
            print(f"✗ {lab}: ĐỊNH DẠNG SAI — {e}")
            return 1

    # ── 2. CHẤM hai bài nộp như đã ghi ra đĩa ───────────────────────────────
    new, old = read_sub(args.new), read_sub(args.old)
    a, b = [], []
    for r in gt:
        q = r["query_id"]
        a.append(score_rows(q, r["video_id"], old[q]))
        b.append(score_rows(q, r["video_id"], new[q]))
    a, b = np.array(a), np.array(b)
    d = b - a
    rng = np.random.default_rng(0)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)])
    lo95, hi95 = np.percentile(bs, [2.5, 97.5])
    print(f"\nFinal · mô hình mốc lệch · L=9 · {SEEDS} lần gieo\n")
    print(f"  cũ  (hợp điểm)          {a.mean():.4f}")
    print(f"  mới (rổ + rerank 2 bậc) {b.mean():.4f}   Δ={d.mean():+.4f}")
    print(f"  KTC95 [{lo95:+.4f}, {hi95:+.4f}]  thắng {(d > 0).sum()} / thua {(d < 0).sum()}"
          f"  {'✓ chắc' if lo95 > 0 else '✗ không chắc'}")

    # ── 3. Quét TRỌNG SỐ RERANK từ bản lưu — không chạy lại GPU ─────────────
    npz = args.new / "_rerank_scores.npz"
    if not npz.exists():
        print(f"\n(không có {npz.name} — bỏ qua phần quét trọng số)")
        return 0
    z = np.load(npz)
    keys = {k.split("/")[0] for k in z.files}
    by_qid = {k.split("-")[-1]: k for k in keys}
    cfg = json.loads((args.new / "_report.json").read_text(encoding="utf-8"))
    RW = cfg.get("rerank_weights")
    if not RW or "fused4" not in RW:
        print(f"✗ {args.new}/_report.json thiếu `rerank_weights` khoá `fused4` — bài nộp "
              f"này chạy bằng lược đồ trọng số CŨ, không quét được")
        return 1

    def fused(tid, w):
        """Điểm rerank trên rổ. `fused4` là điểm ĐÃ HỢP của ④ nguồn, không phải riêng
        thị giác — dùng khoá `visual` ở đây là lỗi sót của lược đồ cũ."""
        rows = z[f"{tid}/rows"]
        s = w["fused4"] * z_normalize(z[f"{tid}/base"][0], np.ones(len(rows), bool))
        if f"{tid}/crop" in z and w.get("crop"):
            s = s + w["crop"] * z_normalize(z[f"{tid}/crop"][0], z[f"{tid}/crop_cov"][0])
        if f"{tid}/vlm" in z and w.get("vlm"):
            s = s + w["vlm"] * z_normalize(z[f"{tid}/vlm"], z[f"{tid}/vlm_cov"])
        return rows, s

    def emit(rows, s):
        """⑦ đúng như đường chạy hiện tại: mỗi keyframe ĐÚNG MỘT dòng, không rải."""
        lines, seen = [], set()
        for row in rows[np.argsort(-s, kind="stable")]:
            row = int(row)
            k = (VID[row], int(FI[row]))
            if k in seen:
                continue
            seen.add(k)
            lines.append(k)
            if len(lines) >= 100:
                break
        return lines

    def run_cfg(w):
        return float(np.mean([score_rows(r["query_id"], r["video_id"],
                                         emit(*fused(by_qid[r["query_id"]], w)))
                              for r in gt]))

    print(f"\nSuy từ {npz.name} ({npz.stat().st_size / 1e6:.1f} MB) — KHÔNG tốn GPU")
    ref = run_cfg(RW)
    print("\nTRỌNG SỐ RERANK — `fused4` cố định 1,0 vì chỉ TỈ LỆ giữa các trọng số "
          "có nghĩa\n")
    print(f"{'fused4':>8}{'mảnh cắt':>10}{'VLM':>7}{'Final':>9}{'Δ':>9}")
    print("-" * 44)
    # Bài nộp chạy với `crop = 0` thì npz KHÔNG có khoá `crop`, nên quét nó ra 4 dòng
    # giống hệt nhau — trông như đã đo mà thật ra tham số vô hiệu. Chỉ quét khi có dữ liệu.
    has_crop = any(f"{t}/crop" in z for t in by_qid.values())
    crops = (0.0, 0.25, 0.5, 1.0) if has_crop else (RW.get("crop", 0.0),)
    if not has_crop:
        print(f"  (npz không có điểm mảnh cắt — bài nộp chạy với crop={crops[0]}; "
              f"muốn quét crop thì chạy lại với `--crop-w 1.0`)")
    out = []
    for c_ in crops:
        for m in (0.0, 0.125, 0.25, 0.5, 1.0):
            w = {"fused4": 1.0, "crop": c_, "vlm": m}
            f = run_cfg(w)
            out.append((f, w))
            print(f"{1.0:>8}{c_:>10}{m:>7}{f:>9.4f}{f - ref:>+9.4f}", flush=True)
    print("-" * 44)
    bf, bw = max(out, key=lambda x: x[0])
    print(f"\ncao nhất {bw} · {bf:.4f}   |   đang dùng {RW} · {ref:.4f}")
    print("\u26a0 argmax trên chính bộ này là KHỚP QUÁ — tìm vùng phẳng, đừng lấy đỉnh.")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
