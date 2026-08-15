#!/usr/bin/env python3
"""
Soi hành vi THẬT của một encoder trên ảnh THẬT — trước khi tiêu tiền cho nó
===========================================================================

Không đọc tài liệu rồi tin. Chạy model trên keyframe thật của bộ dữ liệu này và đo
sáu thứ mà mọi quyết định hạ nguồn dựa vào:

    1. `encode_image` nhả ra bao nhiêu chiều, đã chuẩn hoá chưa
    2. có tham số `truncate_dim` không, và nó có KHỚP với phép cắt tay của ta không
    3. fp16 nhanh hơn bao nhiêu, và đổi kết quả bao nhiêu
    4. thông lượng thật (khung/s) — đầu vào của cổng chi phí
    5. tiếng Việt và tiếng Anh cùng nghĩa có gần nhau không
    6. THANG của cosine chữ↔ảnh so với chữ↔chữ

Điểm 6 là thứ dễ bỏ sót nhất và ảnh hưởng thẳng tới tầng ③: nếu hai thang lệch nhau
một bậc độ lớn thì cộng thẳng vào `S` là sai, phải chuẩn hoá trước.

[ĐO 2026-08-14, jina-clip-v2, A10G, 64 keyframe thật]

    chiều mặc định        1024, norm ≈ 1,0 (model tự chuẩn hoá)
    truncate_dim          có · lệch 0,0016 so cắt tay ⟹ TƯƠNG ĐƯƠNG
    fp32 → fp16           12,70 → 13,39 khung/s (+5%), cosine 0,99979 ⟹ KHÔNG đáng
    tiếng Việt ↔ Anh      0,933 cùng nghĩa · 0,302 khác nghĩa
    cosine chữ↔ảnh        ~0,05  ·  chữ↔chữ ~0,93   ⟹ LỆCH ~19×, phải chuẩn hoá z

Chạy:
    modal run scripts/embed/00_probe_model.py
    modal run scripts/embed/00_probe_model.py --model jinaai/jina-embeddings-v4
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import modal

app = modal.App("aic-probe-encoder")
data = modal.Volume.from_name("aic-data-vol", create_if_missing=False)
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "transformers==4.48.0",
                 "pillow==11.1.0", "numpy==1.26.4", "einops==0.8.0",
                 "timm==1.0.13", "peft==0.14.0")
    .env({"HF_HOME": "/cache", "TOKENIZERS_PARALLELISM": "false"})
)

# Câu tiếng Việt/Anh cùng nghĩa và một câu khác nghĩa — để đo cả căn chỉnh đa ngữ lẫn
# khả năng PHÂN BIỆT. Chỉ đo cái đầu thì một model gán mọi câu về một điểm cũng "đạt".
PROBES_VI = ["một người phụ nữ đang nấu ăn", "một chiếc xe hơi màu đỏ"]
PROBES_EN = ["a woman cooking", "a red car"]


# Ảnh THỨ HAI cho model cần transformers mới (jina-embeddings-v4 dựa trên Qwen2.5-VL,
# chỉ có từ 4.49). KHÔNG nâng pin của ảnh đang chạy tốt: đổi phiên bản dưới chân một
# đường đã đo là cách chắc chắn để mất chính phép đo đó.
image_new = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "transformers==4.52.4",
                 "pillow==11.1.0", "numpy==1.26.4", "einops==0.8.0",
                 "timm==1.0.13", "peft==0.15.2", "accelerate==1.5.2")
    .env({"HF_HOME": "/cache", "TOKENIZERS_PARALLELISM": "false"})
)


@app.function(image=image_new, gpu="A10G", cpu=4.0,
              volumes={"/data": data, "/cache": cache}, timeout=3600)
def probe_new(model_name: str, n_images: int = 64, batch_size: int = 32) -> dict:
    return _body(model_name, n_images, batch_size)


@app.function(image=image, gpu="A10G", cpu=4.0, volumes={"/data": data, "/cache": cache},
              timeout=3600)
def probe(model_name: str, n_images: int = 64, batch_size: int = 32) -> dict:
    return _body(model_name, n_images, batch_size)


def _body(model_name: str, n_images: int, batch_size: int) -> dict:
    import inspect
    import re
    import time

    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModel

    def to_np(x):
        """
        Về numpy bất kể model trả cái gì. jina-clip-v2 trả `np.ndarray`, còn
        jina-embeddings-v4 trả **tensor CUDA** — probe phải độc lập model, nếu không nó
        chỉ soi được đúng model nó đã được viết cho.
        """
        if hasattr(x, "detach"):
            return x.detach().cpu().float().numpy()
        if isinstance(x, (list, tuple)) and x and hasattr(x[0], "detach"):
            return torch.stack(list(x)).detach().cpu().float().numpy()
        return np.asarray(x, dtype=np.float32)

    def norm(a, d=512):
        a = to_np(a)[:, :d]
        return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)

    out: dict = {"model": model_name}
    t0 = time.time()
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to("cuda").eval()
    out["load_seconds"] = round(time.time() - t0, 1)
    # jina-embeddings-v4 đòi chọn nhiệm vụ trước khi mã hoá (`retrieval`/`text-matching`/
    # `code`). jina-clip-v2 không có thuộc tính này — đặt có điều kiện, không giả định.
    if hasattr(model, "task"):
        model.task = "retrieval"
        out["task"] = model.task
    out["params_B"] = round(sum(p.numel() for p in model.parameters()) / 1e9, 2)

    zp = next(Path("/data").glob("Framme/*/*.zip"))
    imgs = []
    with zipfile.ZipFile(zp) as zf:
        for m in zf.namelist():
            if re.search(r"/\d+\.webp$", m):
                imgs.append(Image.open(io.BytesIO(zf.read(m))).convert("RGB"))
                if len(imgs) == n_images:
                    break
    out["image_size"] = list(imgs[0].size)

    with torch.inference_mode():
        v = to_np(model.encode_image(imgs[:8], batch_size=8))
    out["dim"] = v.shape[1]
    out["norm_default"] = round(float(np.linalg.norm(v, axis=1).mean()), 4)

    sig = inspect.signature(model.encode_image)
    out["has_truncate_dim"] = "truncate_dim" in sig.parameters
    if out["has_truncate_dim"]:
        with torch.inference_mode():
            v512 = to_np(model.encode_image(imgs[:8], batch_size=8, truncate_dim=512))
        out["truncate_vs_manual_maxdiff"] = round(float(np.abs(norm(v) - v512).max()), 6)

    def bench(fn, n=3):
        with torch.inference_mode():
            fn()
        t = time.time()
        for _ in range(n):
            with torch.inference_mode():
                fn()
        return (time.time() - t) / n

    out["fps_fp32"] = round(n_images / bench(
        lambda: model.encode_image(imgs, batch_size=batch_size)), 2)
    # `.half()` sửa model TẠI CHỖ và trả về chính nó — `m16 is model`. Nên nếu nhánh này
    # hỏng giữa chừng mà không khôi phục, model kẹt ở fp16 và MỌI phép đo sau đó chạy sai
    # kiểu dữ liệu. Đã mắc đúng lỗi này: phần đo văn bản nổ ở tận dưới, thông báo lỗi
    # không liên quan gì tới fp16. `finally` là chỗ duy nhất đúng để hoàn nguyên.
    try:
        m16 = model.half()
        out["fps_fp16"] = round(n_images / bench(
            lambda: m16.encode_image(imgs, batch_size=batch_size)), 2)
        with torch.inference_mode():
            v16 = to_np(m16.encode_image(imgs[:8], batch_size=8))
        out["cos_fp16_vs_fp32"] = round(float((norm(v16) * norm(v)).sum(axis=1).min()), 5)
    except Exception as e:
        out["fp16_error"] = f"{type(e).__name__}: {e}"[:160]
    finally:
        model = model.float()

    with torch.inference_mode():
        t_vi = norm(model.encode_text(PROBES_VI))
        t_en = norm(model.encode_text(PROBES_EN))
    out["cos_vi_en_same_meaning"] = [round(float(t_vi[i] @ t_en[i]), 4) for i in range(2)]
    out["cos_vi_vi_diff_meaning"] = round(float(t_vi[0] @ t_vi[1]), 4)
    out["cos_text_image_mean"] = round(float((norm(v) @ t_vi[0]).mean()), 4)
    out["scale_gap_text_text_vs_text_image"] = round(
        float(t_vi[0] @ t_en[0]) / max(abs(float((norm(v) @ t_vi[0]).mean())), 1e-9), 1)
    return out


@app.local_entrypoint()
def main(model: str = "jinaai/jina-clip-v2", n_images: int = 64, batch_size: int = 32,
         new_transformers: bool = False):
    fn = probe_new if new_transformers else probe
    r = fn.remote(model, n_images=n_images, batch_size=batch_size)
    print(json.dumps(r, ensure_ascii=False, indent=1))
    if "fps_fp32" in r:
        n = 173426
        for label, fps in (("fp32", r["fps_fp32"]), ("fp16", r.get("fps_fp16") or 0)):
            if fps:
                hrs = n / fps / 3600
                print(f"  chiếu {label}: {hrs:.1f} giờ-container · "
                      f"${hrs * 1.10:.2f} trên A10G · {hrs * 60 / 10:.0f} phút với 10 container")
