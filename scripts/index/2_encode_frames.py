#!/usr/bin/env python3
"""
Mã hoá 173.426 keyframe bằng jina-clip-v2 trên Modal
=====================================================

Tháp ẢNH chạy MỘT LẦN offline. Kết quả là chỉ mục phẳng mà cả tầng truy vấn đứng lên:
`emb.npy` + `ids.npy` + `ranges.json` — xem `src/ingestion/vector_index.py`.

=============================================================================
`--benchmark` TRƯỚC, LUÔN LUÔN
=============================================================================

Ở tầng detection, ước lượng theo FLOP nói T4 chạy 14,7 khung/s. **Đo thật: 1,76.**
Sai **8,4×**, và nếu tin ước lượng thì đã chốt sai cấu hình lẫn sai ngân sách.

Nên script này KHÔNG có đường chạy đủ nào mà không đi qua một phép đo trước. `--benchmark`
mã hoá vài trăm khung thật, đo thông lượng, rồi chiếu chi phí cho 173.426 khung. Cổng
`--max-usd` từ chối chạy nếu phép chiếu vượt ngân sách.

Một chi tiết của phép đo: **bỏ qua thời gian nạp model**. Lần `--verify 20` ở tầng
detection cho 0,24 khung/s vì 20 khung quá ít để khấu hao lần nạp — con số đó bi quan
**2,2×** so lượt thật. Ở đây `t0` đặt SAU khi model đã nạp và đã chạy một lô khởi động.

=============================================================================
GHI THEO TỪNG VIDEO, GỘP Ở CUỐI
=============================================================================

Container Modal là spot, bị thu hồi bất cứ lúc nào. Đơn vị làm lại là **một video**:
mỗi video xong ghi `{video_id}.npy` + `.json` rồi `commit()`. Lượt sau bỏ qua video đã
có đủ file.

Chỉ mục phẳng dựng ở bước cuối, trên máy, bằng `build_flat_index` — nó tự sắp theo
`(video_id, n)` và tự kiểm chuẩn hoá, nên thứ tự các container chạy xong không ảnh
hưởng gì.

Chạy:
    modal run scripts/index/2_encode_frames.py --benchmark 400   # ĐO trước
    modal run scripts/index/2_encode_frames.py                   # lượt đủ
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import modal

MAX_CONTAINERS = 10
DEFAULT_GPU = "T4"

MODEL_NAME = "jinaai/jina-clip-v2"

# LƯU ĐỦ 1024 chiều, KHÔNG cắt lúc mã hoá.
#
# Giá trị của Matryoshka là **quyền chọn số chiều SAU**, không phải bản thân phép cắt.
# Cắt lúc mã hoá là vứt đúng cái quyền đó đi để tiết kiệm 178 MB đĩa — mà muốn lấy lại
# thì phải mã hoá lại toàn bộ 173.426 khung, tức $4,79 và 26 phút.
#
#     lưu 1024 → 355 MB đĩa · cắt xuống 512/256/64 lúc đọc, MIỄN PHÍ
#     lưu  512 → 178 MB đĩa · không bao giờ quay lại 1024 được
#
# Chênh 178 MB. Không đáng để mất quyền chọn.
STORE_DIM = 1024
KEYFRAME_ZIP_GLOB = "Framme/*/*.zip"
OUT_SUBDIR = "embed-jina-v2"

app = modal.App("aic-embed-jina")
data = modal.Volume.from_name("aic-data-vol", create_if_missing=False)
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        # Ghim đúng phiên bản đã kiểm ở máy. Bài học từ tầng detection: để `transformers`
        # tự do làm `dtype=` đổi chữ ký giữa hai phiên bản và nổ trong container.
        "torch==2.5.1",
        "torchvision==0.20.1",
        "transformers==4.48.0",
        "pillow==11.1.0",
        "numpy==1.26.4",
        # jina-clip-v2 dùng `trust_remote_code`; code đó cần các gói sau.
        "einops==0.8.0",
        "timm==1.0.13",
        "peft==0.14.0",
    )
    .env({"HF_HOME": "/cache", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir("src", remote_path="/root/src")
)


def _index_zips(wanted: set[str]) -> dict[str, dict[int, tuple[Path, str]]]:
    """`{video_id: {n: (zip, member)}}` đọc từ central directory — không giải nén."""
    import re
    import zipfile

    member_re = re.compile(r"(?:^|/)(?P<vid>L\d+_V\d+)/(?P<n>\d+)\.webp$")
    idx: dict[str, dict[int, tuple[Path, str]]] = {}
    for zp in sorted(Path("/data").glob(KEYFRAME_ZIP_GLOB)):
        with zipfile.ZipFile(zp) as zf:
            for m in zf.namelist():
                g = member_re.search(m)
                if g and g.group("vid") in wanted:
                    idx.setdefault(g.group("vid"), {})[int(g.group("n"))] = (zp, m)
    return idx


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    cpu=4.0,
    volumes={"/data": data, "/cache": cache},
    timeout=60 * 60 * 4,
    retries=2,
)
def encode_shard(video_ids: list[str], *, batch_size: int = 32,
                 benchmark_frames: int = 0) -> dict:
    """
    Mã hoá các video được giao. `benchmark_frames > 0` ⟹ chỉ ĐO, không ghi gì.
    """
    import sys
    import zipfile

    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModel

    sys.path.insert(0, "/root")
    from src.ingestion.jina_encoder import truncate_and_normalize
    from src.ingestion.vector_index import save_video_shard, video_is_encoded

    t_load = time.time()
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = model.to("cuda").eval()
    load_s = time.time() - t_load

    index = _index_zips(set(video_ids))
    out_root = Path("/data") / OUT_SUBDIR
    handles: dict[Path, zipfile.ZipFile] = {}
    stats = {"videos": 0, "frames": 0, "skipped": 0, "load_seconds": round(load_s, 1)}

    def read(vid: str, n: int):
        zp, mem = index[vid][n]
        zf = handles.get(zp) or handles.setdefault(zp, zipfile.ZipFile(zp))
        return Image.open(io.BytesIO(zf.read(mem))).convert("RGB")

    def encode(imgs) -> "np.ndarray":
        with torch.inference_mode():
            v = model.encode_image(imgs, batch_size=len(imgs))
        return truncate_and_normalize(np.asarray(v, dtype=np.float32), STORE_DIM)

    # ---------- chế độ ĐO ----------
    if benchmark_frames:
        pairs = [(v, n) for v in video_ids for n in sorted(index.get(v, {}))]
        pairs = pairs[:benchmark_frames]
        if not pairs:
            return {**stats, "error": "không tìm thấy khung nào"}
        warm = [read(v, n) for v, n in pairs[:batch_size]]
        encode(warm)                      # lô khởi động — KHÔNG tính vào đồng hồ
        t0 = time.time()                  # đặt SAU khi nạp + khởi động; xem docstring
        done = 0
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i:i + batch_size]
            encode([read(v, n) for v, n in chunk])
            done += len(chunk)
        el = time.time() - t0
        for zf in handles.values():
            zf.close()
        return {**stats, "frames": done, "seconds": round(el, 2),
                "fps": round(done / el, 3), "gpu": torch.cuda.get_device_name(0)}

    # ---------- chế độ GHI ----------
    t0 = time.time()
    for vid in video_ids:
        ns = sorted(index.get(vid, {}))
        if not ns:
            continue
        # Checkpoint: "xong" là ĐỌC ĐƯỢC VÀ ĐÚNG, không phải "file có tồn tại" — container
        # spot có thể chết giữa lúc ghi. Bốn điều kiện ở `video_is_encoded`.
        done, _reason = video_is_encoded(out_root, vid, ns, STORE_DIM)
        if done:
            stats["skipped"] += 1
            continue
        vecs = []
        for i in range(0, len(ns), batch_size):
            vecs.append(encode([read(vid, n) for n in ns[i:i + batch_size]]))
        mat = np.concatenate(vecs) if vecs else np.zeros((0, STORE_DIM), dtype=np.float32)
        save_video_shard(out_root, vid, ns, mat)
        data.commit()
        stats["videos"] += 1
        stats["frames"] += len(ns)

    for zf in handles.values():
        zf.close()
    stats["seconds"] = round(time.time() - t0, 1)
    return stats


@app.local_entrypoint()
def main(benchmark: int = 0, trial: int = 0, batch_size: int = 32, max_usd: float = 8.0,
         usd_per_hour: float = 0.59, gpu: str = "", fps: float = 0.0, groups: str = ""):
    import csv
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.index.shard_plan import balance_shards, projected_cost

    weights = []
    for mp in sorted(Path().glob("data/Framme/*/metadata/*.csv")):
        lines = open(mp, encoding="utf-8-sig").readlines()
        i = next(j for j, l in enumerate(lines) if l.strip())
        weights.append((mp.stem, sum(1 for r in csv.DictReader(lines[i:]) if r.get("frame_idx"))))
    # `--groups L21,L22` giới hạn theo tiền tố video. Chạy từng nhóm để chi phí và thời
    # gian nhỏ, kiểm được kết quả rồi mới đi tiếp — thay vì đặt cược 173.426 khung một lần.
    if groups:
        pre = tuple(g.strip() + "_" for g in groups.split(",") if g.strip())
        weights = [(v, n) for v, n in weights if v.startswith(pre)]
        if not weights:
            raise SystemExit(f"không video nào khớp {groups!r}")
    total = sum(n for _, n in weights)
    print(f"{len(weights)} video · {total:,} keyframe"
          f"{f' · nhóm {groups}' if groups else ''}")

    if benchmark:
        vids = [v for v, _ in weights[:6]]
        print(f"ĐO trên {benchmark} khung của {len(vids)} video, 1 container"
              f"{f' · GPU {gpu}' if gpu else ''}…")
        # `with_options` chứ không sửa decorator: decorator ghim GPU lúc ĐỊNH NGHĨA, nên
        # một cờ `--gpu` không có dòng này sẽ bị bỏ qua ÂM THẦM — lỗi đã mắc ở OWLv2.
        fn = encode_shard.with_options(gpu=gpu) if gpu else encode_shard
        st = fn.remote(vids, batch_size=batch_size, benchmark_frames=benchmark)
        if "fps" not in st:
            raise SystemExit(f"đo hỏng: {st}")
        fps = st["fps"]
        c = projected_cost(total, fps, usd_per_hour, MAX_CONTAINERS)
        print(f"\n  GPU {st['gpu']} · nạp model {st['load_seconds']}s (không tính vào đo)")
        print(f"  {st['frames']} khung / {st['seconds']}s = **{fps} khung/s** mỗi container")
        print(f"\n  CHIẾU cho {total:,} khung với {MAX_CONTAINERS} container:")
        print(f"    ${c['usd']:.2f} · {c['wall_clock_seconds']/60:.0f} phút đồng hồ")
        print(f"    cổng ${max_usd:.2f} ⟹ {'TRONG cổng' if c['usd'] <= max_usd else 'VƯỢT cổng'}")
        return

    # ---------- THỬ ĐƯỜNG GHI: vài video, ghi thật, đọc lại kiểm ----------
    #
    # Tồn tại vì `--benchmark` KHÔNG chạm đường ghi — nó `return` trước nhánh đó. Một bug
    # tên file tạm (`np.save` tự thêm `.npy`) đã sống sót qua cả benchmark lẫn test đơn vị
    # và chỉ lộ ra khi chạy thật. Không lượt đủ nào nên đi trước lượt thử này.
    if trial:
        vids = [v for v, _ in weights[:trial]]
        fn = encode_shard.with_options(gpu=gpu) if gpu else encode_shard
        st = fn.remote(vids, batch_size=batch_size)
        print(f"\nGHI THẬT: {st}")
        print(f"  kiểm bằng `modal volume ls aic-data-vol {OUT_SUBDIR}`")
        return

    # ---------- lượt đủ: BẮT BUỘC có thông lượng ĐO ĐƯỢC ----------
    if fps <= 0:
        raise SystemExit(
            "TỪ CHỐI chạy đủ khi chưa có phép đo. Chạy `--benchmark 400` trước rồi truyền "
            "`--fps <số đo được>` — ước lượng FLOP ở tầng detection từng sai 8,4×."
        )
    c = projected_cost(total, fps, usd_per_hour, MAX_CONTAINERS)
    print(f"\nchiếu từ {fps} khung/s ĐO ĐƯỢC: ${c['usd']:.2f} · "
          f"{c['wall_clock_seconds']/60:.0f} phút · cổng ${max_usd:.2f}")
    if c["usd"] > max_usd:
        raise SystemExit(f"TỪ CHỐI: ${c['usd']:.2f} vượt cổng ${max_usd:.2f}")

    shards = balance_shards(weights, MAX_CONTAINERS)
    parts = [(s.video_ids,) for s in shards if s.video_ids]
    fn = encode_shard.with_options(gpu=gpu) if gpu else encode_shard
    print(f"{len(parts)} phần song song · bắt đầu")
    tot = {"videos": 0, "frames": 0, "skipped": 0}
    t0, done = time.time(), 0
    for st in fn.starmap(parts, kwargs={"batch_size": batch_size}):
        for k in tot:
            tot[k] += st.get(k, 0)
        done += 1
        print(f"  [{'█' * (done * 3)}{'·' * ((len(parts) - done) * 3)}] "
              f"{done}/{len(parts)} phần · {tot['frames']:,}/{total:,} khung · "
              f"{(time.time()-t0)/60:.0f} phút", flush=True)
    wall = time.time() - t0
    print(f"\n✓ {tot['frames']:,} khung · {tot['videos']} video · {tot['skipped']} bỏ qua "
          f"· {wall/60:.0f} phút · ≈${wall/3600*usd_per_hour*MAX_CONTAINERS:.2f}")
    print(f"  → /data/{OUT_SUBDIR}/{{video_id}}.npy — tải về rồi chạy 3_build_index.py")
