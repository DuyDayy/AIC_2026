#!/usr/bin/env python3
"""
Eval chỉ mục vector — ba phép đo THẬT, không cần nhãn tay
=========================================================

Ta **không có** bộ nhãn truy vấn→khung. Nên eval này không giả vờ đo "độ chính xác";
nó đo ba thứ đo được, và mỗi thứ trả lời một câu hỏi khác nhau.

=============================================================================
A. KẾT DÍNH THỜI GIAN — phép kiểm SỐNG/CHẾT, không cần nhãn gì
=============================================================================

Khung trong CÙNG một cảnh phải giống nhau hơn khung của video khác. Nếu không thì
encoder hỏng, và mọi số khác vô nghĩa. Đây là phép kiểm rẻ nhất và phải chạy trước.

Báo `cos` trung bình trong-cảnh so với xuyên-video, và **khoảng cách** giữa hai mức.

=============================================================================
B. ĐỒNG THUẬN LIÊN CHỈ MỤC — tín hiệu thật, dựa trên hai model ĐỘC LẬP
=============================================================================

Ta có `data/objects-full`: mỗi khung kèm danh sách lớp mà **detector OpenImages V4**
nhìn thấy. Nên với lớp `Car` (tiếng Việt *"xe hơi"*), hỏi bằng tiếng Việt rồi xem
**bao nhiêu phần trăm khung top-k thật sự chứa `Car` theo detector**.

Hai model độc lập (Jina CLIP thị giác ↔ Faster R-CNN detector) đồng thuận là bằng
chứng có ý nghĩa; lệch nhau là thông tin.

⚠️ **P@k một mình vô nghĩa.** P@10 = 0,30 là xuất sắc nếu lớp đó chỉ có ở 1% khung, và
tệ nếu nó có ở 40%. Nên luôn báo kèm **tần suất nền** và **hệ số lợi (lift)** = P@k
chia tần suất nền. Lift ≈ 1 nghĩa là truy vấn không mua được gì so bốc ngẫu nhiên.

Giới hạn nói thẳng: nó đo **đồng thuận với detector**, không đo sự thật. Detector cũng
sai. Nhưng nó là bằng chứng ĐỘC LẬP duy nhất ta có mà không phải gán nhãn tay.

=============================================================================
C. QUÉT CHIỀU MATRYOSHKA — chốt một quyết định đang mở
=============================================================================

Chỉ mục lưu 1024 chiều nên cắt xuống là miễn phí. Câu hỏi đang mở: **512 giữ được bao
nhiêu?** Đo bằng hai cách trên cùng bộ truy vấn:

    độ trùng top-10 giữa chiều d và 1024   ← thứ hạng có đổi không
    P@k tại từng chiều                      ← chất lượng có tụt không

Đây là số duy nhất cho phép chốt `dim` bằng bằng chứng thay vì bằng cảm giác.

Chạy:
    modal run scripts/embed/03_eval_index.py::encode_queries   # 1 lần, ~$0.05, cache lại
    python scripts/embed/03_eval_index.py --index data/embed   # $0, chạy lại bao nhiêu cũng được
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

QUERY_VEC = Path("data/embed/eval_queries.npz")
OBJECTS_FULL = Path("data/objects-full")

# Số khung tối thiểu để một lớp được đưa vào eval. Dưới mức này P@10 nhiễu quá để đọc.
MIN_FRAMES_PER_CLASS = 150
# Bỏ lớp phủ quá rộng: lift của chúng bị chặn trên bởi chính tần suất nền, nên chúng
# không phân biệt được encoder tốt với encoder tầm thường.
MAX_PREVALENCE = 0.35
DIMS = (1024, 512, 256, 64)


def load_labels(root: Path, video_prefixes: tuple[str, ...] | None = None):
    """
    `({(video,n): {lớp}}, {lớp: cụm tiếng Việt})` từ chỉ mục detection.

    Map lớp→tiếng Việt lấy từ `zip(detection_class_entities, entities_vi)` — hai mảng
    này song song ĐÚNG THEO VỊ TRÍ trong bản ghi (khác `classes_vi`, đã gộp theo từ
    chính nên độ dài không khớp `classes`).
    """
    labels: dict[tuple[str, int], set[str]] = {}
    vi: dict[str, str] = {}
    for vd in sorted(p for p in root.iterdir() if p.is_dir()):
        if video_prefixes and not vd.name.startswith(video_prefixes):
            continue
        for f in vd.glob("*.json"):
            d = json.loads(f.read_text(encoding="utf-8"))
            labels[(vd.name, int(f.stem))] = set(d["classes"])
            for e, v in zip(d["detection_class_entities"], d["entities_vi"]):
                vi.setdefault(e, v)
    return labels, vi


def pick_classes(labels, vi, min_frames=MIN_FRAMES_PER_CLASS, max_prev=MAX_PREVALENCE):
    """Lớp đủ nhiều để đo, và không phủ quá rộng để lift còn nói được điều gì."""
    n = len(labels) or 1
    cnt = Counter(c for s in labels.values() for c in s)
    out = []
    for c, k in cnt.most_common():
        if k >= min_frames and k / n <= max_prev and c in vi:
            out.append((c, k / n, vi[c]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=Path, default=Path("data/embed"))
    ap.add_argument("--objects", type=Path, default=OBJECTS_FULL)
    ap.add_argument("--queries", type=Path, default=QUERY_VEC)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("data/embed/eval_report.json"))
    args = ap.parse_args()

    from src.ingestion.jina_encoder import truncate_and_normalize
    from src.ingestion.vector_index import load_flat_index

    idx = load_flat_index(args.index)
    pref = tuple(sorted({v.split("_")[0] + "_" for v, _ in idx.ids}))
    print(f"chỉ mục {idx.n_frames:,} khung × {idx.dim} chiều · {len(idx.ranges)} video "
          f"· nhóm {[p[:-1] for p in pref]}")

    row_of = {k: i for i, k in enumerate(idx.ids)}
    labels, vi = load_labels(args.objects, pref)
    labels = {k: v for k, v in labels.items() if k in row_of}
    print(f"nhãn detector cho {len(labels):,} khung khớp chỉ mục")

    report: dict = {"n_frames": idx.n_frames, "dim": idx.dim, "groups": [p[:-1] for p in pref]}

    # ---------- A. kết dính thời gian ----------
    rng = random.Random(0)
    within, across = [], []
    vids = list(idx.ranges)
    for vid in rng.sample(vids, min(60, len(vids))):
        lo, hi = idx.ranges[vid]
        if hi - lo < 3:
            continue
        for _ in range(20):
            i = rng.randrange(lo, hi - 1)
            within.append(float(idx.emb[i] @ idx.emb[i + 1]))   # hai khung LIỀN KỀ
        other = rng.choice([v for v in vids if v != vid])
        olo, ohi = idx.ranges[other]
        for _ in range(20):
            across.append(float(idx.emb[rng.randrange(lo, hi)] @ idx.emb[rng.randrange(olo, ohi)]))
    a = {"cos_lien_ke_cung_video": round(float(np.mean(within)), 4),
         "cos_xuyen_video": round(float(np.mean(across)), 4)}
    a["khoang_cach"] = round(a["cos_lien_ke_cung_video"] - a["cos_xuyen_video"], 4)
    a["PASS"] = a["khoang_cach"] > 0.05
    report["A_ket_dinh_thoi_gian"] = a
    print(f"\nA. KẾT DÍNH THỜI GIAN  liền kề {a['cos_lien_ke_cung_video']} · "
          f"xuyên video {a['cos_xuyen_video']} · cách {a['khoang_cach']} "
          f"⟹ {'ĐẠT' if a['PASS'] else 'HỎNG — encoder không mã hoá nội dung'}")

    if not args.queries.exists():
        print(f"\nthiếu {args.queries} — chạy `modal run {__file__}::encode_queries` trước "
              f"(B và C cần vector truy vấn)")
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        return 0

    z = np.load(args.queries, allow_pickle=True)
    qvecs, qclasses, qtexts = z["vecs"].astype(np.float32), list(z["classes"]), list(z["texts"])
    prev = {c: p for c, p, _ in pick_classes(labels, vi)}

    # ---------- B + C ----------
    per_dim: dict[int, dict] = {}
    top_at_full: dict[str, list[int]] = {}
    for dim in DIMS:
        if dim > idx.dim:
            continue
        emb = truncate_and_normalize(idx.emb, dim)
        rows = []
        for c, t, qv in zip(qclasses, qtexts, qvecs):
            if c not in prev:
                continue
            q = truncate_and_normalize(qv.reshape(1, -1), dim)[0]
            sc = emb @ q
            k = min(args.top_k, len(sc))
            part = np.argpartition(-sc, k - 1)[:k]
            top = list(part[np.argsort(-sc[part])])
            hit = sum(1 for i in top if c in labels[idx.ids[i]])
            p_at_1 = 1.0 if c in labels[idx.ids[top[0]]] else 0.0
            if dim == idx.dim:
                top_at_full[c] = top
            rows.append({"class": c, "query": t, "prevalence": round(prev[c], 4),
                         "p_at_1": p_at_1, "p_at_k": hit / k,
                         "lift": round((hit / k) / max(prev[c], 1e-9), 2),
                         "overlap_vs_full": (len(set(top) & set(top_at_full.get(c, top))) / k)})
        per_dim[dim] = {
            "n_queries": len(rows),
            "P@1": round(float(np.mean([r["p_at_1"] for r in rows])), 4),
            f"P@{args.top_k}": round(float(np.mean([r["p_at_k"] for r in rows])), 4),
            "lift_trung_binh": round(float(np.mean([r["lift"] for r in rows])), 2),
            "trung_lap_top_k_voi_1024": round(float(np.mean([r["overlap_vs_full"] for r in rows])), 4),
            "rows": rows,
        }

    report["B_dong_thuan_lien_chi_muc"] = {d: {k: v for k, v in s.items() if k != "rows"}
                                          for d, s in per_dim.items()}
    report["C_quet_chieu"] = {str(d): per_dim[d]["trung_lap_top_k_voi_1024"] for d in per_dim}
    report["chi_tiet_theo_lop"] = per_dim[max(per_dim)]["rows"]

    print(f"\nB+C. ĐỒNG THUẬN VỚI DETECTOR trên {per_dim[max(per_dim)]['n_queries']} lớp")
    print(f"{'chiều':>6}{'P@1':>8}{'P@'+str(args.top_k):>8}{'lift':>8}{'trùng top-k vs 1024':>22}")
    for d, s in sorted(per_dim.items(), reverse=True):
        print(f"{d:>6}{s['P@1']:>8.3f}{s['P@'+str(args.top_k)]:>8.3f}"
              f"{s['lift_trung_binh']:>8.2f}{s['trung_lap_top_k_voi_1024']:>22.3f}")

    best = per_dim[max(per_dim)]["rows"]
    print(f"\n  5 lớp KHỚP NHẤT (lift cao):")
    for r in sorted(best, key=lambda r: -r["lift"])[:5]:
        print(f"    {r['query'][:30]:<32} P@{args.top_k}={r['p_at_k']:.2f} "
              f"nền={r['prevalence']:.3f} lift={r['lift']:.1f}×")
    print(f"  5 lớp KHỚP KÉM NHẤT:")
    for r in sorted(best, key=lambda r: r["lift"])[:5]:
        print(f"    {r['query'][:30]:<32} P@{args.top_k}={r['p_at_k']:.2f} "
              f"nền={r['prevalence']:.3f} lift={r['lift']:.1f}×")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
