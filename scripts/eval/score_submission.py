#!/usr/bin/env python3
"""
Chấm bài nộp ĐÚNG LUẬT BTC — R-Score và Final nguyên văn thể lệ
=================================================================

    python scripts/eval/score_submission.py --sub submission_bench \\
        --gt export_for_fusion/benchmark_queries.json

Hai việc, tách hẳn nhau:

  1. **KIỂM ĐỊNH DẠNG** bằng `src/submission/writer.py::validate_all` — đúng bộ kiểm
     sẽ chạy khi ghi file nộp thật. Sai định dạng thì điểm vô nghĩa, nên kiểm trước.
  2. **CHẤM ĐIỂM** bằng `src/scoring/rscore.py::final_score` — cài đặt nguyên văn:

         R(rᵢ)  = I(vᵢ = GTᵥ ∧ idᵢ ∈ [s,e])                    KIS
         R(rᵢ)  = I(… ∧ aᵢ = GTₐ)                              Q&A
         R(rᵢ)  = (1/N)·Σⱼ I(id_{i,j} ∈ [sⱼ,eⱼ])  nếu đúng video, 0 nếu sai   TRAKE
         R@k    = max_{i≤k} R(rᵢ),   k ∈ {1,5,20,50,100}
         Final  = (1/5)·Σ_k R@k

=============================================================================
MỘT THAM SỐ KHÔNG BIẾT: BỀ RỘNG `L` CỦA ĐOẠN [s, e]
=============================================================================

Mọi thứ khác trong công thức là xác định. Chỉ `[s,e]` là do BTC quy định và ta không
có. Thể lệ cho hai manh mối:

    "đoạn ứng với khoảnh khắc ngữ nghĩa này thường rất ngắn, thông thường là DƯỚI 10
     FRAME" — và "cùng nguyên tắc … như ở Textual KIS và Q&A"
    ví dụ KIS trong thể lệ: đáp án [500, 510] = 11 khung

Nên script này **không báo một con số**. Nó quét `L` và in **đường cong Final(L)**.
Chọn một `L` rồi gọi đó là "điểm của ta" là giấu đi tham số duy nhất mình không biết.

Giả định vị trí: đoạn đặt **đối xứng quanh mốc ngữ nghĩa**, và mốc ngữ nghĩa lấy bằng
`frame_idx` của keyframe ground truth. Đó là giả định **lạc quan** — thể lệ nói rõ khung
ngữ nghĩa KHÁC keyframe kỹ thuật đã cấp, nên mốc thật lệch đi một khoảng chưa biết.
Cột `lệch ngẫu nhiên` mô phỏng chiều bi quan: mốc rơi đều trong **nửa khe quanh
keyframe ∩ biên shot ∩ độ dài video**. Chặn biên shot là bắt buộc — mốc ngữ nghĩa
thuộc MỘT shot, không thể rơi sang shot bên cạnh.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ingestion.vector_index import load_flat_index
from src.retrieval.sources import load_shot_bounds, load_video_last_frame
from src.scoring.rscore import (
    Interval,
    KISAnswer,
    KISGroundTruth,
    K_THRESHOLDS,
    MAX_ANSWERS,
    final_score,
    stable_seed,
)
from src.submission.writer import SubmissionError, TaskSubmission, validate_all

LS = (1, 5, 9, 11, 21, 51, 101)
SEEDS = 24


def read_submissions(d: Path) -> dict[str, list[tuple[str, tuple[int, ...], str | None]]]:
    """`{task_id: [(video_id, (frame_idx…), answer)]}` từ các file CSV."""
    out = {}
    for p in sorted(d.glob("*.csv")):
        rows = []
        for r in csv.reader(open(p, encoding="utf-8")):
            if not r or not r[0]:
                continue
            frames = tuple(int(x) for x in r[1:] if x.strip().lstrip("-").isdigit())
            ans = r[-1] if len(r) > 1 and not r[-1].strip().lstrip("-").isdigit() else None
            rows.append((r[0], frames, ans))
        out[p.stem] = rows
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sub", type=Path, default=Path("submission_bench"))
    ap.add_argument("--gt", type=Path,
                    default=Path("export_for_fusion/benchmark_queries.json"))
    ap.add_argument("--index", type=Path, default=Path("data/embed"))
    args = ap.parse_args()

    gt_rows = json.loads(args.gt.read_text(encoding="utf-8"))
    subs = read_submissions(args.sub)
    idx = load_flat_index(args.index, dim=512)
    pos = {k: i for i, k in enumerate(idx.ids)}
    FI = np.asarray(idx.frame_idx)

    # Khớp task_id: tên file có dạng `NNN-kis-<query_id>`; ghép theo query_id.
    by_qid = {}
    for tid, rows in subs.items():
        by_qid[tid.split("-")[-1]] = (tid, rows)

    # ── 1. KIỂM ĐỊNH DẠNG, đúng bộ kiểm dùng lúc ghi file nộp thật ─────────
    bounds = {}
    for v, (lo, hi) in idx.ranges.items():
        bounds[v] = int(FI[lo:hi].max()) + 1
    tasks = [TaskSubmission(task_id=tid, task_type="kis",
                            answers=tuple(rows), n_moments=1)
             for tid, rows in subs.items()]
    try:
        validate_all(tasks, budget=MAX_ANSWERS, frame_bounds=bounds)
        print(f"✓ ĐỊNH DẠNG hợp lệ · {len(tasks)} truy vấn · "
              f"kiểm bằng writer.validate_all")
    except SubmissionError as e:
        print(f"✗ ĐỊNH DẠNG SAI: {e}")
        return 1

    # ── 2. CHẤM ĐIỂM, quét L vì đó là tham số duy nhất không biết ──────────
    # Cửa sổ mốc lệch = nửa khe keyframe ∩ BIÊN SHOT ∩ độ dài video.
    #
    # 🔴 Chặn biên shot là BẮT BUỘC, không phải tinh chỉnh. Mốc ngữ nghĩa nằm trong
    # một shot — nó không thể rơi sang shot bên cạnh. Bản trước chỉ lấy nửa khe, nên
    # cửa sổ RỘNG hơn thực tế và điểm THẤP hơn thực tế: bộ 110 giữ kín ra 0,2605 thay
    # vì 0,2801, tức lệch 0,0197 — gấp 3 lần biên độ nhiễu hạt giống (0,0068 ở 32 lần
    # gieo). `compare_arch.py` vốn đã chặn; hai bộ chấm trong cùng kho từng cho hai
    # con số khác nhau cho cùng một bài nộp.
    shot_b, last_f = load_shot_bounds(), load_video_last_frame()
    nbr = {}
    for v, (lo, hi) in idx.ranges.items():
        o = np.argsort(FI[lo:hi])
        f = FI[lo:hi][o]
        vmax = last_f.get(v, int(f[-1])) - 1
        for j, r in enumerate(o):
            row = lo + int(r)
            c = int(f[j])
            prev = int(f[j - 1]) if j else c - 71
            nxt = int(f[j + 1]) if j + 1 < len(f) else c + 71
            a, b = c - (c - prev) // 2, c + (nxt - c) // 2
            ss, se = shot_b.get(idx.ids[row], (a, b))
            nbr[row] = (max(0, a, ss), min(b, se, vmax))

    fixed = {L: [] for L in LS}
    jitter = {L: [] for L in LS}
    per_type = defaultdict(lambda: {L: [] for L in LS})
    # R@k tách riêng — `Final` gộp 5 mốc thành một số, che mất hình dạng đường cong.
    # Đường cong mới nói hệ hỏng ở đâu: R@1 thấp mà R@100 cao ⟹ xếp hạng thô nhưng
    # recall tốt; R@1 cao mà phẳng về sau ⟹ ngược lại.
    perk_fixed = {L: {k: [] for k in K_THRESHOLDS} for L in LS}
    perk_jit = {L: {k: [] for k in K_THRESHOLDS} for L in LS}
    missing = 0
    for r in gt_rows:
        qid = r["query_id"]
        if qid not in by_qid:
            missing += 1
            continue
        _, rows = by_qid[qid]
        answers = [KISAnswer(v, f[0]) for v, f, _ in rows if f]
        g = pos[(r["video_id"], int(r["frame_id"]))]
        gf = int(FI[g])
        lo, hi = nbr[g]          # đã chặn biên shot ở khối dựng `nbr`
        rng = np.random.default_rng(stable_seed(qid))
        for L in LS:
            half = (L - 1) // 2
            s = final_score(answers,
                            KISGroundTruth(r["video_id"], Interval(gf - half, gf + half)))
            fixed[L].append(s.final)
            per_type[r["query_type"]][L].append(s.final)
            for k in K_THRESHOLDS:
                perk_fixed[L][k].append(float(s.per_k[k]))
            js, jk = [], {k: [] for k in K_THRESHOLDS}
            for _ in range(SEEDS):
                t = int(rng.integers(lo, hi + 1))
                sj = final_score(answers, KISGroundTruth(
                    r["video_id"], Interval(t - half, t + half)))
                js.append(sj.final)
                for k in K_THRESHOLDS:
                    jk[k].append(float(sj.per_k[k]))
            jitter[L].append(float(np.mean(js)))
            for k in K_THRESHOLDS:
                perk_jit[L][k].append(float(np.mean(jk[k])))
    if missing:
        print(f"⚠ {missing} truy vấn không có file nộp")

    n = len(fixed[LS[0]])
    print(f"\nR@k tại k ∈ {list(K_THRESHOLDS)} · Final = (1/5)·Σ R@k · "
          f"ngân sách {MAX_ANSWERS} · {n} truy vấn\n")
    print(f"{'L (bề rộng [s,e])':<20}{'mốc = keyframe':>18}{'mốc lệch ngẫu nhiên':>23}")
    print("-" * 62)
    for L in LS:
        print(f"{L:<20}{np.mean(fixed[L]):>18.4f}{np.mean(jitter[L]):>23.4f}")
    print("-" * 62)
    print("  cột trái  = giả định LẠC QUAN: mốc ngữ nghĩa trùng keyframe của ta")
    print("  cột phải  = giả định BI QUAN: mốc rơi đều trong nửa khe quanh keyframe ∩ biên shot")
    print("  thể lệ nói đoạn đáp án 'thường dưới 10 frame'; ví dụ KIS là 11 khung")

    # ── BẢNG R@k — `Final` gộp 5 mốc, bảng này mở chúng ra ──────────────────
    for lab, tab in (("mốc = keyframe", perk_fixed), ("mốc lệch ngẫu nhiên", perk_jit)):
        print(f"\nR@k · {lab}\n")
        print(f"{'L':>4}" + "".join(f"{'R@'+str(k):>9}" for k in K_THRESHOLDS)
              + f"{'Final':>10}")
        print("-" * (4 + 9 * len(K_THRESHOLDS) + 10))
        for L in LS:
            row = [np.mean(tab[L][k]) for k in K_THRESHOLDS]
            print(f"{L:>4}" + "".join(f"{v:>9.4f}" for v in row)
                  + f"{np.mean(row):>10.4f}")
        print("-" * (4 + 9 * len(K_THRESHOLDS) + 10))
    print("  `Final` ở cột cuối = trung bình 5 cột R@k, đúng công thức thể lệ")

    if len(per_type) > 1:
        order = ["vision", "vision+ocr", "vision+asr", "vision+ocr+asr"]
        show = [L for L in (9, 11, 21) if L in LS]
        print(f"\n{'loại':<18}{'n':>4}" + "".join(f"{'L='+str(L):>10}" for L in show))
        print("-" * (22 + 10 * len(show)))
        for t in sorted(per_type, key=lambda x: order.index(x) if x in order else 9):
            v = per_type[t]
            print(f"{t:<18}{len(v[show[0]]):>4}"
                  + "".join(f"{np.mean(v[L]):>10.4f}" for L in show))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
