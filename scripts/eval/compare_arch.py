#!/usr/bin/env python3
"""
So KIẾN TRÚC MỚI (rổ ứng viên + rerank 2 bậc) với kiến trúc cũ (hợp điểm)
=========================================================================

    python scripts/eval/compare_arch.py --new submission_new --old submission_bench

Chấm bằng `src/scoring/rscore.py` — nguyên văn thể lệ. Báo theo **mô hình mốc lệch
ngẫu nhiên**, vì hai mô hình kia ôm keyframe ground truth nên tự thưởng cho việc nộp lại
chính cái nhãn của mình (xem README, *Bài học đo lường*).

Có `_rerank_scores.npz` thì suy thêm biến thể `spread` khác **mà không chạy lại GPU**:
bản lưu giữ rổ + điểm nền + điểm mảnh cắt + điểm VLM cho từng truy vấn.
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
from src.scoring.rscore import Interval, KISAnswer, KISGroundTruth, final_score
from src.submission.coverage import spread_in_window
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
    ap.add_argument("--spreads", type=int, nargs="*", default=[1, 3, 5, 7, 9])
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
        rng = np.random.default_rng(abs(hash(r["query_id"])) % 2**31)
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

    # ── 3. Suy các `spread` khác TỪ BẢN LƯU — không chạy lại GPU ────────────
    npz = args.new / "_rerank_scores.npz"
    if not npz.exists():
        print(f"\n(không có {npz.name} — bỏ qua phần suy `spread`)")
        return 0
    z = np.load(npz)
    keys = {k.split("/")[0] for k in z.files}
    by_qid = {k.split("-")[-1]: k for k in keys}
    cfg = json.loads((args.new / "_report.json").read_text(encoding="utf-8"))
    RW = cfg.get("rerank_weights", {"visual": 0.30, "crop": 1.0, "vlm": 1.0})

    def fused(tid, w):
        rows = z[f"{tid}/rows"]
        s = w["visual"] * z_normalize(z[f"{tid}/base"][0], np.ones(len(rows), bool))
        if f"{tid}/crop" in z and w.get("crop"):
            s = s + w["crop"] * z_normalize(z[f"{tid}/crop"][0], z[f"{tid}/crop_cov"][0])
        if f"{tid}/vlm" in z and w.get("vlm"):
            s = s + w["vlm"] * z_normalize(z[f"{tid}/vlm"], z[f"{tid}/vlm_cov"])
        return rows, s

    def emit(rows, s, sp):
        order = rows[np.argsort(-s, kind="stable")]
        lines, seen = [], set()
        for row in order:
            row = int(row)
            c = int(FI[row])
            wl, wh = win[row]
            for f in spread_in_window(c, min(wl, c), max(wh, c), sp):
                k = (VID[row], int(f))
                if k in seen:
                    continue
                seen.add(k)
                lines.append(k)
                if len(lines) >= 100:
                    return lines
        return lines

    def run_cfg(w, sp):
        vals = [score_rows(r["query_id"], r["video_id"],
                           emit(*fused(by_qid[r["query_id"]], w), sp)) for r in gt]
        return float(np.mean(vals))

    print(f"\nSuy từ {npz.name} ({npz.stat().st_size / 1e6:.1f} MB) — KHÔNG tốn GPU")
    print(f"\n\u2460 `spread`  (trọng số rerank giữ nguyên {RW})\n")
    print(f"{'spread':>8}{'mốc':>7}{'Final':>9}")
    print("-" * 26)
    for sp in args.spreads:
        print(f"{sp:>8}{100 // sp:>7}{run_cfg(RW, sp):>9.4f}", flush=True)
    print("-" * 26)

    sp0 = int(cfg.get("spread", 7))
    ref = run_cfg(RW, sp0)
    print(f"\n\u2461 TRỌNG SỐ RERANK — chỗ `VLM_WEIGHT` được quét (spread={sp0})\n")
    print(f"{'nền':>6}{'mảnh cắt':>10}{'VLM':>7}{'Final':>9}{'Δ':>9}")
    print("-" * 42)
    out = []
    for v in (0.0, 0.15, 0.30, 1.0):
        for c_ in (0.0, 0.5, 1.0, 2.0):
            for m in (0.0, 0.5, 1.0, 2.0):
                if v == 0 and c_ == 0 and m == 0:
                    continue
                w = {"visual": v, "crop": c_, "vlm": m}
                f = run_cfg(w, sp0)
                out.append((f, w))
                print(f"{v:>6}{c_:>10}{m:>7}{f:>9.4f}{f - ref:>+9.4f}", flush=True)
    print("-" * 42)
    bf, bw = max(out, key=lambda x: x[0])
    print(f"\ncao nhất {bw} · {bf:.4f}   |   đang dùng {RW} · {ref:.4f}")
    print("\u26a0 argmax trên chính bộ này là KHỚP QUÁ — tìm vùng phẳng, đừng lấy đỉnh.")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
