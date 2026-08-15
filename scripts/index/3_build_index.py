#!/usr/bin/env python3
"""
Gộp các file .npy theo-từng-video thành MỘT chỉ mục phẳng — $0, không GPU
=========================================================================

`2_encode_frames.py` ghi mỗi video một file để checkpoint được (container Modal là
spot). Nhưng tầng truy vấn cần **một ma trận**: quét phẳng là một lệnh `emb @ q`, và
DANTE cần lát `emb[lo:hi]` liên tục cho mỗi video.

Bước này đọc các file rời rồi gọi `build_flat_index`, hàm tự sắp theo `(video_id, n)`
và tự kiểm chuẩn hoá — nên thứ tự các container chạy xong không ảnh hưởng gì.

=============================================================================
KIỂM ĐỘ PHỦ TRƯỚC KHI GHI, KHÔNG PHẢI SAU
=============================================================================

Chỉ mục thiếu khung là lỗi im lặng: mọi truy vấn vẫn chạy, vẫn trả kết quả, chỉ là
khung đúng không bao giờ có cơ hội xuất hiện. Nên bước này đối chiếu với metadata của
team TRƯỚC khi ghi, và **từ chối ghi** nếu thiếu — trừ khi `--allow-partial`.

Chạy:
    modal volume get aic-data-vol embed-jina-v2 /tmp/emb/
    python scripts/index/3_build_index.py --src /tmp/emb/embed-jina-v2
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.retrieval.sources import load_frame_idx
from src.ingestion.vector_index import (
    build_flat_index,
    check_alignment,
    load_flat_index,
    save_flat_index,
)

OUT_DIR = Path("data/embed")
TEAM_META_GLOB = "data/Framme/*/metadata/*.csv"


def team_keyframes() -> list[tuple[str, int]]:
    """`[(video_id, n)]` — tập khung mà chỉ mục PHẢI phủ."""
    out: list[tuple[str, int]] = []
    for mp in sorted(Path().glob(TEAM_META_GLOB)):
        lines = open(mp, encoding="utf-8-sig").readlines()
        i = next(j for j, l in enumerate(lines) if l.strip())
        for r in csv.DictReader(lines[i:]):
            if r.get("frame_idx"):
                out.append((mp.stem, int(r["n"])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="thư mục .npy tải từ volume")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--allow-partial", action="store_true",
                    help="ghi dù thiếu khung — chỉ dùng khi đang chạy dở, có chủ ý")
    args = ap.parse_args()

    expected = team_keyframes()
    print(f"cần phủ {len(expected):,} khung / {len({v for v, _ in expected})} video")

    rows: list[tuple[str, int, np.ndarray]] = []
    missing_files: list[str] = []
    for vid in sorted({v for v, _ in expected}):
        vp, jp = args.src / f"{vid}.npy", args.src / f"{vid}.json"
        if not (vp.exists() and jp.exists()):
            missing_files.append(vid)
            continue
        mat = np.load(vp).astype(np.float32)
        ns = json.loads(jp.read_text(encoding="utf-8"))
        if mat.shape[0] != len(ns):
            raise SystemExit(
                f"{vid}: {mat.shape[0]} vector ≠ {len(ns)} khung — hai mảng song song "
                f"lệch, KHÔNG ghép theo vị trí khi đã lệch"
            )
        rows.extend((vid, int(n), mat[i]) for i, n in enumerate(ns))

    print(f"đọc {len(rows):,} vector từ {len({v for v, _, _ in rows})} video"
          f"{f' · THIẾU FILE: {len(missing_files)} video' if missing_files else ''}")

    # Chỉ mục PHẢI mang số khung thật: `n` là số thứ tự keyframe, còn luật thi
    # chấm `frame_idx`, và đo được 0/173.426 khung có hai giá trị bằng nhau.
    index = build_flat_index(rows, load_frame_idx())
    missing = check_alignment(index, expected)
    if missing and not args.allow_partial:
        raise SystemExit(
            f"TỪ CHỐI ghi: thiếu {len(missing):,} khung, vd {missing[:3]}. Chỉ mục thiếu "
            f"khung là lỗi IM LẶNG — truy vấn vẫn chạy, khung đúng không bao giờ hiện ra. "
            f"Chạy nốt phần thiếu, hoặc --allow-partial nếu cố ý."
        )

    save_flat_index(args.out, index)
    back = load_flat_index(args.out)          # đọc lại để bất biến được KIỂM, không chỉ ghi
    mb = sum(f.stat().st_size for f in args.out.iterdir()) / 1e6
    print(f"\n✓ {args.out} · {back.n_frames:,} × {back.dim} · {len(back.ranges)} lát video"
          f" · {mb:.1f} MB")
    print(f"  thiếu khung: {len(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
