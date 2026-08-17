#!/usr/bin/env python3
"""
Chấm bài nộp theo GROUND TRUTH CẤP SHOT (`benchmark_ground_truth_final_v2`)
===========================================================================

    python scripts/eval/score_shots.py --sub submission_gtv2 \\
        --gt benchmark_ground_truth_final_v2/ground_truth_final.jsonl

=============================================================================
VÌ SAO ĐÂY LÀ THƯỚC KHÁC VỚI `score_submission.py`
=============================================================================

`score_submission.py` chấm đúng luật BTC: `R-Score = I(video khớp ∧ idᵢ ∈ [s,e])` với
`[s,e]` là cửa sổ KHUNG hẹp (~10 khung) mà ta **không biết**, nên nó phải quét `L`.

Bộ GT này chấm theo **shot**, và README của nó ghi rõ *"Không chấm theo exact frame"*.
Nên hai bộ số **không so ngang được**:

  · chấm theo shot **rộng tay hơn** — trúng bất kỳ khung nào trong shot là được, mà shot
    dài 1,5–7 giây tức 40–190 khung, so với cửa sổ ~10 khung của thể lệ;
  · nhưng nó đo đúng **chất lượng truy xuất**, không bị nhiễu bởi tham số `L` chưa biết.

Dùng bộ này để so các cấu hình với nhau; dùng `score_submission.py` để ước lượng điểm thi.

=============================================================================
ÁNH XẠ — CHỖ DUY NHẤT DỄ SAI
=============================================================================

Bài nộp ghi `(video_id, frame_idx)`; GT định danh `shot_id` dạng `L21_V008_shot_077`.
Nối bằng metadata: `shot_id` của GT **khớp chính xác** cột `shot_id` trong
`data/Framme/*/metadata/*.csv`, và cột `shot_start_frame`/`shot_end_frame` cho biên thật.

⚠️ ĐỪNG nối bằng `shot_start`/`shot_end` (giây) của GT — hai trường đó là thời điểm
**keyframe đầu và cuối trong shot**, KHÔNG phải biên shot. So chúng với biên metadata cho
0/105 khớp và làm tưởng dữ liệu lệch; thật ra chỉ là so hai đại lượng khác nhau.

=============================================================================
THANG ĐIỂM
=============================================================================

GT cho `relevance` 2 (positive) và 1 (partial). Thể lệ BTC cho phép `R-Score` nhận giá trị
trung gian, nên bảng dưới báo **hai thang**:

    NGHIÊM   chỉ `positive_shots` được 1,0; còn lại 0
    CÓ PHẦN  `positive` 1,0 · `partial` 0,5 · còn lại 0

`R@k = max_{i≤k} R-Score(rᵢ)` và `Final = (1/5)·Σ R@k` giữ nguyên công thức thể lệ.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.scoring.rscore import K_THRESHOLDS, MAX_ANSWERS

SHOT_RE = re.compile(r"^(.+)_shot_(\d+)$")


def load_shot_frames(meta_glob: str) -> dict[tuple[str, int], tuple[int, int]]:
    """`{(video, shot_id) → (khung đầu, khung cuối)}` từ metadata."""
    out: dict[tuple[str, int], tuple[int, int]] = {}
    for f in glob.glob(meta_glob):
        vid = Path(f).stem
        with open(f, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("shot_id") and r.get("shot_start_frame"):
                    out.setdefault((vid, int(r["shot_id"])),
                                   (int(r["shot_start_frame"]), int(r["shot_end_frame"])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sub", type=Path, required=True)
    ap.add_argument("--gt", type=Path,
                    default=Path("benchmark_ground_truth_final_v2/ground_truth_final.jsonl"))
    ap.add_argument("--meta", default="data/Framme/*/metadata/*.csv")
    ap.add_argument("--by-type", action="store_true",
                    help="tách theo query_type. CHỈ để phân tích — đường ống không thấy.")
    args = ap.parse_args()

    Q = [json.loads(l) for l in args.gt.read_text(encoding="utf-8").splitlines() if l.strip()]
    shot_fr = load_shot_frames(args.meta)
    if not shot_fr:
        print(f"✗ không đọc được metadata từ {args.meta}")
        return 1

    # (video, shot_id) → relevance, theo từng truy vấn
    rel: dict[str, dict[tuple[str, int], float]] = {}
    unresolved = 0
    for q in Q:
        m: dict[tuple[str, int], float] = {}
        for key, r in (("positive_shots", 2.0), ("partial_shots", 1.0)):
            for s in q.get(key) or []:
                g = SHOT_RE.match(s["shot_id"])
                if not g:
                    unresolved += 1
                    continue
                k = (g.group(1), int(g.group(2)))
                if k not in shot_fr:
                    unresolved += 1
                    continue
                m[k] = max(m.get(k, 0.0), r)
        rel[q["query_id"]] = m
    if unresolved:
        print(f"⚠ {unresolved} shot của GT không tra được trong metadata")

    # biên shot theo video, để tra khung → shot bằng tìm nhị phân
    by_vid: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for (v, sid), (a, b) in shot_fr.items():
        by_vid[v].append((a, b, sid))
    for v in by_vid:
        by_vid[v].sort()
    starts = {v: np.array([a for a, _, _ in L]) for v, L in by_vid.items()}

    # Video có trong bài nộp mà metadata KHÔNG có → mọi khung của nó tính trượt. Phải
    # nói rõ, vì con số `nomap` trần trụi trông giống lỗi nộp sai khung. [ĐO] hiện có
    # đúng 1 video như vậy: `L25_V081`, 1/873.
    no_meta: set[str] = set()

    def shot_of(vid: str, frame: int) -> int | None:
        L = by_vid.get(vid)
        if not L:
            no_meta.add(vid)
            return None
        i = int(np.searchsorted(starts[vid], frame, side="right")) - 1
        if i < 0:
            return None
        a, b, sid = L[i]
        return sid if a <= frame <= b else None

    subs = {}
    for p in sorted(args.sub.glob("*.csv")):
        rows = [(r[0], int(r[1])) for r in csv.reader(open(p, encoding="utf-8"))
                if r and r[0]]
        subs[p.stem.split("-")[-1]] = rows
    miss = [q["query_id"] for q in Q if q["query_id"] not in subs]
    if miss:
        print(f"⚠ {len(miss)} truy vấn không có file nộp")

    out = {"NGHIÊM": defaultdict(list), "CÓ PHẦN": defaultdict(list)}
    per_type = {"NGHIÊM": defaultdict(lambda: defaultdict(list))}
    nomap = 0
    for q in Q:
        qid = q["query_id"]
        if qid not in subs:
            continue
        rows = subs[qid][:MAX_ANSWERS]
        sc_strict, sc_grade = [], []
        for v, f in rows:
            sid = shot_of(v, f)
            if sid is None:
                nomap += 1
                sc_strict.append(0.0); sc_grade.append(0.0)
                continue
            r = rel[qid].get((v, sid), 0.0)
            sc_strict.append(1.0 if r >= 2.0 else 0.0)
            sc_grade.append(1.0 if r >= 2.0 else (0.5 if r >= 1.0 else 0.0))
        for lab, sc in (("NGHIÊM", sc_strict), ("CÓ PHẦN", sc_grade)):
            for k in K_THRESHOLDS:
                out[lab][k].append(max(sc[:k], default=0.0))
        for k in K_THRESHOLDS:
            per_type["NGHIÊM"][q.get("query_type", "?")][k].append(
                max(sc_strict[:k], default=0.0))

    n = len(out["NGHIÊM"][1])
    print(f"\nChấm CẤP SHOT · {n} truy vấn · ngân sách {MAX_ANSWERS}")
    if no_meta:
        print(f"⚠ {len(no_meta)} video trong bài nộp KHÔNG có file metadata nên mọi khung "
              f"của chúng tính trượt: {sorted(no_meta)}")
    print(f"({nomap:,} dòng nộp không tra được shot — trong đó phần thuộc video thiếu "
          f"metadata ở trên, phần còn lại là khung nằm trong khe giữa hai shot)\n")
    print(f"{'thang':<10}" + "".join(f"{'R@'+str(k):>9}" for k in K_THRESHOLDS)
          + f"{'Final':>10}")
    print("-" * (10 + 9 * len(K_THRESHOLDS) + 10))
    for lab in ("NGHIÊM", "CÓ PHẦN"):
        row = [float(np.mean(out[lab][k])) for k in K_THRESHOLDS]
        print(f"{lab:<10}" + "".join(f"{v:>9.4f}" for v in row)
              + f"{np.mean(row):>10.4f}")
    print("-" * (10 + 9 * len(K_THRESHOLDS) + 10))
    print("  NGHIÊM  = chỉ `positive_shots` tính điểm")
    print("  CÓ PHẦN = `positive` 1,0 · `partial` 0,5")

    if args.by_type:
        print("\n(theo loại đề — CHỈ để phân tích, đường ống không thấy nhãn này)\n")
        t = per_type["NGHIÊM"]
        print(f"{'loại':<18}{'n':>4}" + "".join(f"{'R@'+str(k):>9}" for k in K_THRESHOLDS)
              + f"{'Final':>10}")
        for ty in sorted(t):
            row = [float(np.mean(t[ty][k])) for k in K_THRESHOLDS]
            print(f"{ty:<18}{len(t[ty][1]):>4}" + "".join(f"{v:>9.4f}" for v in row)
                  + f"{np.mean(row):>10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
