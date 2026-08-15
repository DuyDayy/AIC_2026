#!/usr/bin/env python3
"""
Mã hoá truy vấn eval bằng tháp VĂN BẢN — chạy MỘT LẦN, cache lại
================================================================

Bước duy nhất của eval cần GPU. Nó nhỏ: vài chục câu tiếng Việt, vài giây. Sau khi có
`data/embed/eval_queries.npz` thì `04_eval_index.py` chạy **$0, lặp bao nhiêu cũng được**.

Câu truy vấn **không bịa**: lấy đúng cụm tiếng Việt mà chỉ mục detection phát ra cho
từng lớp (`entities_vi`), rồi bọc bằng một khuôn tối giản. Nhờ vậy phép đo ở bước 04 là
"cùng một khái niệm, hai model độc lập có đồng thuận không", chứ không phải "tôi diễn
đạt câu hỏi hay tới đâu".

Chạy:
    modal run scripts/embed/03_encode_queries.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

MODEL_NAME = "jinaai/jina-clip-v2"
OUT = Path("data/embed/eval_queries.npz")

app = modal.App("aic-embed-queries")
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "transformers==4.48.0", "pillow==11.1.0",
                 "numpy==1.26.4", "einops==0.8.0", "timm==1.0.13", "peft==0.14.0")
    .env({"HF_HOME": "/cache", "TOKENIZERS_PARALLELISM": "false"})
)


@app.function(image=image, gpu="A10G", volumes={"/cache": cache}, timeout=1800)
def encode(texts: list[str]) -> list[list[float]]:
    """Trả vector ĐỦ 1024 chiều — bước 04 tự cắt xuống các chiều cần đo."""
    import numpy as np
    import torch
    from transformers import AutoModel

    m = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to("cuda").eval()
    with torch.inference_mode():
        v = m.encode_text(texts, batch_size=32)
    v = np.asarray(v, dtype=np.float32)
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    return v.tolist()


@app.local_entrypoint()
def main(objects: str = "data/objects-full", groups: str = ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import numpy as np

    # Chọn lớp ngay tại đây thay vì import từ bước 04: tên module bắt đầu bằng số nên
    # không import được bằng cú pháp thường, và logic này ngắn — nhân đôi rõ hơn là lách.
    from collections import Counter
    root = Path(objects)
    pref = tuple(g.strip() + "_" for g in groups.split(",") if g.strip()) or None
    labels, vi = {}, {}
    for vd in sorted(p for p in root.iterdir() if p.is_dir()):
        if pref and not vd.name.startswith(pref):
            continue
        for f in vd.glob("*.json"):
            d = json.loads(f.read_text(encoding="utf-8"))
            labels[(vd.name, int(f.stem))] = set(d["classes"])
            for e, v in zip(d["detection_class_entities"], d["entities_vi"]):
                vi.setdefault(e, v)
    n = len(labels) or 1
    cnt = Counter(c for s in labels.values() for c in s)
    picked = [(c, k / n) for c, k in cnt.most_common()
              if k >= 150 and k / n <= 0.35 and c in vi]
    print(f"{len(labels):,} khung · {len(cnt)} lớp · chọn {len(picked)} lớp để eval")

    # Khuôn tối giản: lấy từ CHÍNH cụm mà chỉ mục phát ra, chỉ thêm "một" phía trước nếu
    # cụm chưa có loại từ. Không viết lại bằng LLM — eval phải đo encoder, không đo prompt.
    texts, classes = [], []
    for c, _ in picked:
        term = vi[c].split(" / ")[0]
        texts.append(term if term.startswith(("một", "các", "nhiều")) else f"một {term}")
        classes.append(c)
    for t, c in list(zip(texts, classes))[:8]:
        print(f"   {c:<20} → {t!r}")

    vecs = np.array(encode.remote(texts), dtype=np.float32)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, vecs=vecs, classes=np.array(classes), texts=np.array(texts))
    print(f"\n✓ {OUT} · {vecs.shape}")
