#!/usr/bin/env python3
"""
SINH BỘ TRUY VẤN ĐỘC LẬP để kiểm chéo — chống thiên lệch có chủ đích
=====================================================================

    modal run scripts/eval/make_queries.py --n 120 --out queries_holdout

=============================================================================
VÌ SAO CẦN, VÀ VÌ SAO KHÔNG DÙNG LẠI HAI BỘ ĐANG CÓ
=============================================================================

Mọi trọng số và kết luận kiến trúc hiện tại đều **chỉnh trên bộ 100 câu gán nhãn tay**.
Đo lại trên chính bộ đó thì con số nào cũng đẹp — đó là khớp quá, không phải bằng chứng.

Bộ 226 câu có sẵn **không thay thế được**: model viết câu **khi đang nhìn khung đích**,
nên nó thổi điểm lên ~1,5 lần, và đo được **0% câu cần OCR/ASR** — tức nó không kiểm được
đúng những kết luận quan trọng nhất (δ ASR ×2, rổ ứng viên, dung hợp).

=============================================================================
BỐN CHỐT CHỐNG THIÊN LỆCH
=============================================================================

1. **Lấy mẫu khung NGẪU NHIÊN**, không theo điểm nhúng. Chọn theo điểm là tự lọc ra
   khung mà jina-clip vốn đã thích, rồi đo lại chính nó.

2. **Loại mọi video xuất hiện trong bộ 100 câu.** Không chỉ loại khung — loại cả video,
   vì hai khung cùng video chia sẻ OCR, lời nói và bối cảnh.

3. **Phân tầng CÓ CHỦ ĐÍCH** thay vì để model tự chọn viết gì: chia đều bốn loại
   `vision` · `vision+ocr` · `vision+asr` · `vision+ocr+asr`. Bộ 226 câu lệch hẳn về
   thuần thị giác chính vì không có bước này.

4. **Bắt DIỄN GIẢI LẠI, cấm trích nguyên văn** nội dung OCR/ASR. Trích nguyên văn thì
   BM25 khớp tầm thường và ta sẽ thổi phồng giá trị của hai nguồn đó — đúng lỗi mà bộ
   226 câu mắc ở chiều ngược lại.

=============================================================================
THIÊN LỆCH CÒN LẠI — PHẢI BIẾT KHI ĐỌC SỐ
=============================================================================

Model vẫn **nhìn khung đích** lúc viết. Nên điểm tuyệt đối trên bộ này vẫn là **cận
trên**, y như bộ 226. Thứ dùng được là **so sánh tương đối** giữa các cấu hình, vì mọi
cấu hình đều chịu cùng một thiên lệch.

Muốn số tuyệt đối đáng tin thì cần người **không nhìn khung** viết đề — không tự động
hoá được.
"""

from __future__ import annotations

import base64
import io
import json
import random
import re
import sys
from pathlib import Path

import modal

# Mã cấp module chạy CẢ HAI phía: ở máy file nằm tại `scripts/eval/`, còn Modal copy nó
# thành `/root/make_queries.py` — nơi chỉ có hai cấp cha, nên `parents[2]` ném IndexError
# và giết cả lần chạy trước khi vào hàm nào.
_here = Path(__file__).resolve()
ROOT = _here.parents[2] if len(_here.parents) > 2 else _here.parent
sys.path.insert(0, str(ROOT))

QA_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
KEYFRAME_ROOTS = ("data/Framme/L21-L25/Keyframes L21-L25",
                  "data/Framme/L26/L26",
                  "data/Framme/L27-L30/DATA")

app = modal.App("aic-mkq")
cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
vl_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "transformers==4.51.3", "pillow==11.1.0",
                 "accelerate==1.6.0", "qwen-vl-utils==0.0.10")
    .env({"HF_HOME": "/cache", "TOKENIZERS_PARALLELISM": "false"})
)

# Một hướng dẫn cho MỖI tầng. Khác nhau ở chỗ được phép dựa vào bằng chứng nào.
GUIDE = {
    "vision": (
        "Viết MỘT câu truy vấn tiếng Việt để tìm lại đúng cảnh này trong kho video.\n"
        "CHỈ mô tả những gì NHÌN THẤY: người, vật, hành động, bối cảnh, màu sắc.\n"
        "TUYỆT ĐỐI KHÔNG nhắc tới chữ hiện trên màn hình, tên kênh, hay lời người nói."),
    "vision+ocr": (
        "Viết MỘT câu truy vấn tiếng Việt để tìm lại đúng cảnh này.\n"
        "Mô tả cảnh NHÌN THẤY, VÀ nhắc tới ý của dòng chữ hiện trên màn hình.\n"
        "DIỄN GIẢI LẠI dòng chữ bằng lời của bạn — KHÔNG chép nguyên văn."),
    "vision+asr": (
        "Viết MỘT câu truy vấn tiếng Việt để tìm lại đúng cảnh này.\n"
        "Mô tả cảnh NHÌN THẤY, VÀ nhắc tới NỘI DUNG người trong video đang nói.\n"
        "DIỄN GIẢI LẠI lời nói bằng lời của bạn — KHÔNG chép nguyên văn."),
    "vision+ocr+asr": (
        "Viết MỘT câu truy vấn tiếng Việt để tìm lại đúng cảnh này.\n"
        "Mô tả cảnh NHÌN THẤY, VÀ nhắc tới cả ý dòng chữ trên màn hình lẫn nội dung "
        "lời nói.\nDIỄN GIẢI LẠI cả hai — KHÔNG chép nguyên văn."),
}
COMMON = ("\n\nViết như một người ĐÃ XEM video và đang nhớ lại khoảnh khắc đó, "
          "không phải như người đang mô tả từng chi tiết trong ảnh.\n"
          "Một câu, 15–35 từ, KHÔNG xuống dòng, KHÔNG giải thích thêm, "
          "KHÔNG mở đầu bằng 'Câu truy vấn:'.")


@app.function(image=vl_image, gpu="A10G", volumes={"/cache": cache_vol}, timeout=3600)
def write_queries(jobs: list[dict]) -> list[str]:
    """`[{image_b64, guide, ocr, asr}]` → `[câu truy vấn]`, ĐÚNG thứ tự đầu vào."""
    import io as _io

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    proc = AutoProcessor.from_pretrained(QA_MODEL)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QA_MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    out = []
    for j in jobs:
        im = Image.open(_io.BytesIO(base64.b64decode(j["image_b64"]))).convert("RGB")
        ctx = ""
        if j.get("ocr"):
            ctx += f"\n\nChữ hiện trên màn hình: {j['ocr'][:300]}"
        if j.get("asr"):
            ctx += f"\n\nNgười trong video đang nói: {j['asr'][:400]}"
        msg = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": j["guide"] + ctx + COMMON}]}]
        text = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[text], images=[im], return_tensors="pt").to("cuda")
        with torch.inference_mode():
            g = model.generate(**inp, max_new_tokens=90, do_sample=False)
        s = proc.batch_decode(g[:, inp.input_ids.shape[1]:],
                              skip_special_tokens=True)[0]
        out.append(" ".join(s.split()))
    return out


def frame_path(vid: str, n: int) -> Path | None:
    for root in KEYFRAME_ROOTS:
        for pat in (f"{vid}/{n:03d}.webp", f"{vid}/{n:04d}.webp",
                    f"{vid}/{n:03d}.jpg", f"{vid}/{n:04d}.jpg"):
            p = Path(root) / pat
            if p.exists():
                return p
    hits = list(Path(".").glob(f"data/Framme/**/{vid}/{n:0>3}.*"))
    return hits[0] if hits else None


@app.local_entrypoint()
def main(n: int = 120, out: str = "queries_holdout", seed: int = 20260816,
         bench: str = "export_for_fusion/benchmark_queries.json"):
    import numpy as np
    from PIL import Image

    from src.ingestion.vector_index import load_flat_index
    from src.retrieval.sources import load_asr_segments, load_frame_ms, load_ocr_text

    idx = load_flat_index(Path("data/embed"), dim=512)
    ocr = load_ocr_text("data/OCR/ocr.jsonl")
    fms = load_frame_ms()
    asr = load_asr_segments("data/ASR")

    # CHỐT 2 — loại cả VIDEO, không chỉ khung: hai khung cùng video chia sẻ OCR,
    # lời nói và bối cảnh, nên giữ lại là rò rỉ tập tune.
    banned = {r["video_id"] for r in json.loads(Path(bench).read_text(encoding="utf-8"))}
    print(f"loại {len(banned)} video có trong bộ tune")

    def asr_at(vid: str, nn: int) -> str:
        ms = fms.get((vid, nn))
        if ms is None:
            return ""
        return " ".join(t for a, b, t in asr.get(vid, [])
                        if a - 4000 <= ms <= b + 4000)[:500]

    rng = random.Random(seed)
    rows = [i for i, (v, _) in enumerate(idx.ids) if v not in banned]
    rng.shuffle(rows)

    per = n // 4
    want = {"vision": per, "vision+ocr": per, "vision+asr": per,
            "vision+ocr+asr": n - 3 * per}
    picked: list[dict] = []
    seen_vid: dict[str, int] = {}
    for i in rows:
        if not want or all(v == 0 for v in want.values()):
            break
        vid, nn = idx.ids[i]
        if seen_vid.get(vid, 0) >= 2:        # ≤2 câu mỗi video, tránh dồn cụm
            continue
        o = (ocr.get((vid, nn)) or "").strip()
        a = asr_at(vid, nn)
        kind = ("vision+ocr+asr" if len(o) >= 25 and len(a) >= 60 else
                "vision+ocr" if len(o) >= 25 and len(a) < 20 else
                "vision+asr" if len(a) >= 60 and len(o) < 10 else
                "vision" if len(o) < 10 and len(a) < 20 else None)
        if kind is None or want.get(kind, 0) <= 0:
            continue
        p = frame_path(vid, nn)
        if p is None:
            continue
        want[kind] -= 1
        seen_vid[vid] = seen_vid.get(vid, 0) + 1
        picked.append({"video_id": vid, "n": int(nn), "frame_idx": int(idx.frame_idx[i]),
                       "query_type": kind, "ocr": o, "asr": a, "path": str(p)})
    print(f"lấy mẫu {len(picked)} khung: " +
          " · ".join(f"{k}={sum(1 for x in picked if x['query_type']==k)}"
                     for k in GUIDE))
    if want and any(v > 0 for v in want.values()):
        print(f"⚠ thiếu: {dict((k, v) for k, v in want.items() if v > 0)}")

    jobs = []
    for x in picked:
        im = Image.open(x["path"]).convert("RGB")
        im.thumbnail((896, 896))
        b = io.BytesIO()
        im.save(b, "JPEG", quality=90)
        jobs.append({"image_b64": base64.b64encode(b.getvalue()).decode(),
                     "guide": GUIDE[x["query_type"]],
                     # CHỐT 4 — chỉ đưa OCR/ASR cho tầng ĐƯỢC PHÉP dùng chúng
                     "ocr": x["ocr"] if "ocr" in x["query_type"] else "",
                     "asr": x["asr"] if "asr" in x["query_type"] else ""})

    print(f"sinh {len(jobs)} câu bằng Qwen2.5-VL…", flush=True)
    texts: list[str] = []
    B = 40
    parts = [jobs[i:i + B] for i in range(0, len(jobs), B)]
    for got in write_queries.map(parts):
        texts += got
        print(f"   {len(texts)}/{len(jobs)}", flush=True)
    if len(texts) != len(jobs):
        raise SystemExit(f"nhận {len(texts)} câu cho {len(jobs)} việc — lệch thì gán "
                         f"nhầm ground truth, dừng")

    odir = Path(out)
    odir.mkdir(parents=True, exist_ok=True)
    for f in odir.glob("*.txt"):
        f.unlink()
    gt = []
    kept = 0
    for x, t in zip(picked, texts):
        t = re.sub(r'^["\'\s]*(?:câu truy vấn|truy vấn)\s*[:：]\s*', "", t,
                   flags=re.I).strip(' "\'')
        if len(t.split()) < 8:               # câu quá ngắn là model hỏng, bỏ
            continue
        qid = f"h_{x['video_id']}_{x['n']}"
        (odir / f"{kept + 1:03d}-kis-{qid}.txt").write_text(t + "\n", encoding="utf-8")
        gt.append({"query_id": qid, "query_text": t, "query_type": x["query_type"],
                   "video_id": x["video_id"], "frame_id": x["n"],
                   "frame_idx": x["frame_idx"]})
        kept += 1
    (odir / "_gt.json").write_text(json.dumps(gt, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    print(f"\n✓ {odir}/ · {kept} truy vấn + _gt.json")
    for k in GUIDE:
        print(f"   {k:<16} {sum(1 for g in gt if g['query_type'] == k)}")
    for g in gt[:3]:
        print(f"\n  [{g['query_type']}] {g['query_text']}")
