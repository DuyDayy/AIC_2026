#!/usr/bin/env python3
"""
CHẠY TOÀN BỘ — thả file .txt vào thư mục rồi gọi một lệnh
============================================================

    modal run scripts/run.py                      # đọc ./queries, ghi ./submission
    modal run scripts/run.py --dir de --out nop   # đổi thư mục
    modal run scripts/run.py --no-rerank          # bỏ ⑤, không cần lượt GPU thứ hai
    modal run scripts/run.py --light              # chỉ thị giác, cho máy thiếu RAM

Mỗi truy vấn là **một file `.txt`**, tên file thành `query_id`. Bổ sung truy vấn =
thêm file, không sửa gì trong mã.

Loại đề đoán theo, ưu tiên từ trên xuống:
  1. tên file chứa `trake` / `qa` / `kis`
  2. nội dung có `E1:` `E2:` ⟹ TRAKE
  3. còn lại ⟹ Textual KIS

=============================================================================
VÌ SAO PHẢI CHẠY QUA `modal run`
=============================================================================

Tháp VĂN BẢN của jina-clip-v2 cần GPU, nên không mã hoá truy vấn tại máy được. Mọi thứ
CÒN LẠI chạy tại máy và **$0**: chỉ mục, BM25, DP, khử trùng, ghi bài nộp.

Vector đã mã hoá được **cache theo băm nội dung** ở `data/embed/query_cache.npz`. Chạy
lại với cùng bộ truy vấn ⟹ không tốn GPU lần nào nữa. Thêm một file mới ⟹ chỉ mã hoá
đúng file đó.

=============================================================================
TẦNG NÀO CHẠY, VÀ ĐIỀU GÌ ĐÃ ĐO
=============================================================================

    ① probe hoá   tách mốc E1:/E2:, rút trích dẫn
    ② bốn nguồn   thị giác · OCR · vật thể · lời nói
    ③ ma trận S   chuẩn hoá z trong tập CÓ dữ liệu, trọng số 1/0,1/0,1/0,1 (TẠM)
    ④ DANTE       DP thứ tự thời gian; λ=0 (đo được mọi λ>0 tệ hơn)
    ⑤ rerank      bậc 1 mảnh cắt jina-clip (đúng màu 54% → 77%)
                  bậc 2 VLM chấm P(khớp) — TẮT mặc định, bật bằng --vlm-top-k
    ⑥ đầu đọc     CHỈ đề Q&A: Qwen2.5-VL đọc khung + OCR + lời nói → sinh `answer`
    ⑦ nộp         RẢI khung vào khe giữa keyframe; KHÔNG khử trùng

=============================================================================
KHÔNG KHỬ TRÙNG — VÀ ĐÂY LÀ MỘT KẾT LUẬN ĐÃ BỊ ĐẢO
=============================================================================

Khử trùng theo cảnh từng đo được **+2,0pp** (thắng 23/thua 0). Nhưng phép đo đó giả
định `đáp án = keyframe của ta`, mà thể lệ bác bỏ điều đó. Đo lại theo mô hình đúng —
mốc thật lệch ngẫu nhiên trong khe:

    khử trùng  rải     L=9      L=11     L=21
    không        7   0,2317   0,2476   0,2788   ← mặc định
    cảnh         7   0,1889   0,2037   0,2408
    không        5   0,2120   0,2406   0,2974
    không        1   0,0831   0,1022   0,1844

Khử trùng cảnh **mất 4,3pp** (−23% tương đối). Khử trùng theo video còn tệ hơn nữa.

**Cơ chế:** phép rải ĐÃ TỰ LO việc chống trùng. Mỗi keyframe chỉ rải trong **nửa khe
của chính nó**, nên các dải rải **lát kề nhau, không chồng lên nhau**. Khử trùng sau đó
chỉ bỏ đi phần PHỦ mà không bỏ được phần TRÙNG nào — vì không còn phần trùng nào.

Bật lại bằng `--dedup` nếu cần so.

=============================================================================
VÌ SAO PHẢI RẢI KHUNG — ĐÁP ÁN KHÔNG PHẢI KEYFRAME BTC CẤP
=============================================================================

Thể lệ: *"khung hình ngữ nghĩa … KHÁC VỚI I-Frame là khung hình kỹ thuật … đã được
cung cấp cho các đội thi"*, và đoạn đáp án *"thường rất ngắn, thông thường là dưới 10
frame"*. Ví dụ KIS trong thể lệ: `[500, 510]` = 11 khung.

[ĐO] xác suất một cửa sổ `L=9` đặt ngẫu nhiên **chứa sẵn** một keyframe của ta:
**23,5%**. Đó là **trần cứng** của cách nộp thuần keyframe, bất kể truy xuất tốt đến đâu.

[ĐO] Final trên 226 truy vấn, mốc thật lệch ngẫu nhiên trong khe, ngân sách 100:

    100 mốc × 1 khung   L=9 0,0608   L=11 0,0758   L=15 0,1002   L=21 0,1384
    20 mốc × 5 khung    L=9 0,1624   L=11 0,1877   L=15 0,2137   L=21 0,2349
    14 mốc × 7 khung    L=9 0,1742   L=11 0,1882   L=15 0,2038   L=21 0,2184  ← mặc định

**Gấp ~2,5 lần.** Chọn `spread=7` chứ không phải 5 vì hai lý do cộng lại: nó tốt nhất ở
`L=9` — điểm vận hành nhiều khả năng nhất, do thể lệ nói *"thường dưới 10 frame"* — và
7 khung rải trong khe trung vị 48 cho **bước 8 khung**, nên theo Định lý 1 nó **BẢO
ĐẢM** trúng khi `L ≥ 8`, không còn là xác suất. `--spread 5` nhỉnh hơn nếu `L ≥ 15`.

Đổi bằng `--spread`; `--spread 1` quay về cách nộp thuần keyframe.

Bài nộp ghi **`frame_idx`** — số khung thật. `n` chỉ là số thứ tự keyframe, và đo được
0/173.426 khung có hai giá trị bằng nhau.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CACHE = Path("data/embed/query_cache.npz")
KEYFRAME_ROOTS = ("data/Framme/L21-L25/Keyframes L21-L25",
                  "data/Framme/L26/L26",
                  "data/Framme/L27-L30/DATA")
RERANK_TOP_K = 100

# Trọng số bậc 2. Điểm VLM nằm trong [0,1] còn điểm nền là z-score, nhưng `fuse` chuẩn
# hoá z cả hai nên trọng số chỉ diễn đạt MỨC TIN CẬY. 1,0 = tin VLM ngang thị giác.
# CHƯA QUÉT trên dữ liệu thật — xem README, mục "còn phải làm".
VLM_WEIGHT = 1.0
CROP_BATCH = 400          # ~10 MB base64 mỗi lượt gọi Modal

app = modal.App("aic-run")
cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "transformers==4.48.0", "pillow==11.1.0",
                 "numpy==1.26.4", "einops==0.8.0", "timm==1.0.13", "peft==0.14.0")
    .env({"HF_HOME": "/cache", "TOKENIZERS_PARALLELISM": "false"})
)
MODEL = "jinaai/jina-clip-v2"

# Qwen2.5-VL cần transformers ≥ 4.49; jina-clip-v2 lại chốt ở 4.48. Hai image riêng
# thay vì ép chung một phiên bản — ép chung là cách nhanh nhất để hỏng cả hai.
vl_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "transformers==4.51.3", "pillow==11.1.0",
                 "accelerate==1.6.0", "qwen-vl-utils==0.0.10")
    .env({"HF_HOME": "/cache", "TOKENIZERS_PARALLELISM": "false"})
)


@app.function(image=image, gpu="A10G", volumes={"/cache": cache_vol}, timeout=3600)
def encode_text(texts: list[str]) -> list[list[float]]:
    import numpy as np
    import torch
    from transformers import AutoModel

    m = AutoModel.from_pretrained(MODEL, trust_remote_code=True).to("cuda").eval()
    with torch.inference_mode():
        v = np.asarray(m.encode_text(texts, batch_size=16), dtype=np.float32)
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    return v.tolist()


@app.function(image=image, gpu="A10G", volumes={"/cache": cache_vol}, timeout=3600)
def encode_crops(blobs: list[str]) -> list[list[float]]:
    """Mã hoá mảnh cắt bằng tháp ẢNH — cùng không gian với tháp văn bản."""
    import base64 as b64
    import io as _io

    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModel

    m = AutoModel.from_pretrained(MODEL, trust_remote_code=True).to("cuda").eval()
    ims = [Image.open(_io.BytesIO(b64.b64decode(b))).convert("RGB") for b in blobs]
    with torch.inference_mode():
        v = np.asarray(m.encode_image(ims, batch_size=32), dtype=np.float32)
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    return v.tolist()


QA_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


@app.function(image=vl_image, gpu="A10G", volumes={"/cache": cache_vol}, timeout=3600)
def score_frames_vlm(jobs: list[dict]) -> list[float]:
    """
    Tầng ⑤ bậc 2 — VLM chấm `P(khung khớp mô tả)` cho từng khung.

    Ép model trả **một ký tự** `1` hoặc `0`, rồi đọc **softmax trên đúng hai token đó**.
    Không sinh chuỗi, không phân tích văn bản: điểm luôn xác định, liên tục, tất định.

    Bậc này thấy thứ mảnh cắt không thấy — quan hệ giữa các vật, hành động, phủ định —
    nên nó bổ sung chứ không thay thế bậc 1.
    """
    import base64 as b64
    import io as _io

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    proc = AutoProcessor.from_pretrained(QA_MODEL)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QA_MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    tok = proc.tokenizer
    # Lấy id của "1" và "0". Nếu vốn từ tách chúng thành nhiều token thì phép đọc xác
    # suất mất nghĩa — chặn ngay thay vì trả điểm rác.
    ids = {}
    for ch in ("1", "0"):
        t = tok.encode(ch, add_special_tokens=False)
        if len(t) != 1:
            raise RuntimeError(f"ký tự {ch!r} tách thành {len(t)} token — không đọc "
                               f"được xác suất một token")
        ids[ch] = t[0]

    out = []
    for j in jobs:
        im = Image.open(_io.BytesIO(b64.b64decode(j["image_b64"]))).convert("RGB")
        msg = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text":
             f"Mô tả cần tìm: {j['query']}\n\nKhung hình này có khớp mô tả trên không? "
             f"Chỉ trả lời một ký tự: 1 nếu khớp, 0 nếu không."}]}]
        text = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[text], images=[im], return_tensors="pt").to("cuda")
        with torch.inference_mode():
            logits = model(**inp).logits[0, -1]
        p1, p0 = logits[ids["1"]].float(), logits[ids["0"]].float()
        out.append(float(torch.softmax(torch.stack([p0, p1]), dim=0)[1]))
    return out


@app.function(image=vl_image, gpu="A10G", volumes={"/cache": cache_vol}, timeout=3600)
def read_answer(jobs: list[dict]) -> list[str]:
    """
    Tầng ⑥ — đọc khung + chứng cứ chữ rồi sinh chuỗi `answer` cho đề Q&A.

    Không có tầng này thì câu Q&A **được 0 điểm** dù tìm đúng khung: thể lệ đòi
    `aᵢ = GTₐ` ngoài `vᵢ = GTᵥ` và `idᵢ ∈ [s,e]`.

    Nhận `[{image_b64, question, ocr, asr}]`, trả `[answer]` cùng thứ tự. Đưa kèm OCR
    và lời nói vì nhiều câu hỏi (tên riêng, con số, ngày tháng) **chỉ đọc được từ chữ**,
    không nhìn ra từ ảnh.
    """
    import base64 as b64
    import io as _io

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    proc = AutoProcessor.from_pretrained(QA_MODEL)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QA_MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    out = []
    for j in jobs:
        im = Image.open(_io.BytesIO(b64.b64decode(j["image_b64"]))).convert("RGB")
        ctx = ""
        if j.get("ocr"):
            ctx += f"\nChữ hiện trên khung: {j['ocr'][:300]}"
        if j.get("asr"):
            ctx += f"\nLời nói quanh khung: {j['asr'][:400]}"
        msg = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text":
             f"{j['question']}{ctx}\n\nTrả lời NGẮN GỌN bằng tiếng Việt, chỉ nêu đáp "
             f"án, không giải thích. Nếu là số thì viết bằng chữ số."}]}]
        text = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[text], images=[im], return_tensors="pt").to("cuda")
        with torch.inference_mode():
            g = model.generate(**inp, max_new_tokens=48, do_sample=False)
        ans = proc.batch_decode(g[:, inp.input_ids.shape[1]:],
                                skip_special_tokens=True)[0].strip()
        out.append(" ".join(ans.split())[:120])
    return out


# ─────────────────────────── phần chạy tại máy, $0 ───────────────────────────

def read_queries(d: Path) -> list[dict]:
    """`[{id, text, kind}]` từ mọi `.txt` trong thư mục, sắp theo tên."""
    from src.retrieval.probe import EVENT_MARKER

    out = []
    for p in sorted(d.glob("*.txt")):
        text = " ".join(p.read_text(encoding="utf-8").split())
        if not text:
            print(f"  ⚠ bỏ qua {p.name}: file rỗng", file=sys.stderr)
            continue
        low = p.stem.lower()
        kind = ("trake" if "trake" in low else "qa" if "qa" in low
                else "kis" if "kis" in low
                else "trake" if EVENT_MARKER.search(text) else "kis")
        out.append({"id": p.stem, "text": text, "kind": kind})
    return out


def load_cache() -> dict[str, list[float]]:
    if not CACHE.exists():
        return {}
    import numpy as np
    z = np.load(CACHE, allow_pickle=True)
    return {str(k): v for k, v in zip(z["keys"], z["vecs"])}


def save_cache(c: dict) -> None:
    import numpy as np
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, keys=np.array(list(c)), vecs=np.array(list(c.values())))


def key_of(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def frame_path(video_id: str, n: int) -> Path | None:
    for r in KEYFRAME_ROOTS:
        p = Path(r) / video_id / f"{n:03d}.webp"
        if p.exists():
            return p
    return None


def make_crops(refs, ids) -> list[str]:
    """`[CropRef]` → ảnh JPEG base64, bỏ mảnh quá nhỏ để không nội suy ra hư vô."""
    from PIL import Image

    out = []
    for c in refs:
        p = frame_path(c.video_id, c.n)
        if p is None:
            out.append(None)
            continue
        im = Image.open(p).convert("RGB")
        box = c.pixel_box(*im.size)
        crop = im.crop(box)
        if min(crop.size) < 24:
            out.append(None)
            continue
        crop.thumbnail((336, 336))
        b = io.BytesIO()
        crop.save(b, "JPEG", quality=88)
        out.append(base64.b64encode(b.getvalue()).decode())
    return out


@app.local_entrypoint()
def main(dir: str = "queries", out: str = "submission", index: str = "data/embed",
         top_k: int = 100, rerank: bool = True, light: bool = False, dim: int = 512,
         spread: int = 7, vlm_top_k: int = 0, dedup: bool = False):
    import numpy as np

    from src.ingestion.jina_encoder import truncate_and_normalize
    from src.retrieval.dante import dante_over_videos
    from src.retrieval.probe import build_probes
    from src.retrieval.rerank import collect_crops, crop_scores, vlm_scores
    from src.retrieval.score_matrix import DEFAULT_WEIGHTS, fuse
    from src.submission.coverage import spread_in_gap
    from src.ingestion.vector_index import load_flat_index
    from src.retrieval.sources import (
        AsrSource, SourceScores, TextSource, VisualSource, load_asr_segments,
        load_frame_ms, load_object_text, load_ocr_text, load_shot_id,
    )

    qdir, odir = Path(dir), Path(out)
    if not qdir.is_dir():
        raise SystemExit(f"không thấy thư mục {qdir} — tạo nó rồi thả file .txt vào")
    queries = read_queries(qdir)
    if not queries:
        raise SystemExit(f"{qdir} không có file .txt nào")
    print(f"{len(queries)} truy vấn từ {qdir}/")

    # ① probe hoá → danh sách văn bản cần mã hoá
    for q in queries:
        q["probes"] = [p.text for p in build_probes(q["text"])] or [q["text"]]
        if q["kind"] != "trake" and len(q["probes"]) > 1:
            print(f"  ⚠ {q['id']}: tách ra {len(q['probes'])} mốc nhưng loại là "
                  f"{q['kind']} — đổi sang TRAKE", file=sys.stderr)
            q["kind"] = "trake"

    cache = load_cache()
    need = sorted({t for q in queries for t in q["probes"] if key_of(t) not in cache})
    if need:
        print(f"mã hoá {len(need)} đoạn mới trên GPU (đã cache {len(cache)})…")
        for i, v in zip(need, encode_text.remote(need)):
            cache[key_of(i)] = v
        save_cache(cache)
    else:
        print(f"toàn bộ đã có trong cache — không tốn GPU")
    QV = {t: truncate_and_normalize(np.asarray([cache[key_of(t)]], np.float32), dim)[0]
          for q in queries for t in q["probes"]}

    # ②③ nạp nguồn
    t0 = time.time()
    idx = load_flat_index(Path(index), dim=dim if dim < 1024 else None)
    vis = VisualSource(idx)
    tsrc = []
    if not light:
        tsrc = [TextSource("ocr", idx.ids, load_ocr_text("data/OCR/ocr.jsonl")),
                TextSource("object", idx.ids, load_object_text("data/objects-full")),
                AsrSource(idx.ids, load_frame_ms(), load_asr_segments("data/ASR"))]
    W = dict(DEFAULT_WEIGHTS) if not light else {"visual": 1.0}
    shot = load_shot_id()
    fms = load_frame_ms()
    times = np.array([fms.get(k, 0.0) for k in idx.ids], dtype=np.float64)
    # Hàng xóm THỜI GIAN trong cùng video — `spread_in_gap` cần nó để rải theo mật độ
    # cục bộ. Khe đo được p10=19, p50=48, p90=105 khung nên bước cố định là sai.
    FI = np.asarray(idx.frame_idx)
    nbr: dict[int, tuple[int, int]] = {}
    for v, (lo, hi) in idx.ranges.items():
        o = np.argsort(FI[lo:hi])
        f = FI[lo:hi][o]
        for j, r in enumerate(o):
            nbr[lo + int(r)] = (int(f[j - 1]) if j else int(f[j]) - 71,
                                int(f[j + 1]) if j + 1 < len(f) else int(f[j]) + 71)
    print(f"nạp {time.time() - t0:.0f}s · {idx.n_frames:,} khung · {idx.dim} chiều "
          f"· trọng số {W}")

    # ②③④ chấm điểm
    for q in queries:
        rows = [fuse([vis.score(QV[t])] + [s.score(t) for s in tsrc], W)
                for t in q["probes"]]
        q["S"] = np.vstack(rows).astype(np.float32)

    # ⑤ rerank — gom mảnh cắt của MỌI truy vấn rồi mã hoá một lượt
    if rerank:
        allrefs, span = [], {}
        for q in queries:
            base = q["S"].max(axis=0)
            cand = [int(r) for r in np.argsort(-base)[:min(top_k, RERANK_TOP_K)]]
            refs = collect_crops(cand, idx.ids, "data/objects-full")
            span[q["id"]] = (len(allrefs), len(allrefs) + len(refs))
            allrefs.extend(refs)
        blobs = make_crops(allrefs, idx.ids)
        ok = [i for i, b in enumerate(blobs) if b]
        print(f"⑤ rerank: {len(ok)}/{len(blobs)} mảnh cắt · "
              f"ước tính ${len(ok) * 4.32 / 173426:.2f}")
        vecs = np.zeros((len(blobs), idx.dim), dtype=np.float32)
        for s in range(0, len(ok), CROP_BATCH):
            part = ok[s:s + CROP_BATCH]
            got = np.asarray(encode_crops.remote([blobs[i] for i in part]), np.float32)
            vecs[part] = truncate_and_normalize(got, dim)
            print(f"   {min(s + CROP_BATCH, len(ok))}/{len(ok)}")
        for q in queries:
            lo, hi = span[q["id"]]
            refs = [allrefs[i] for i in range(lo, hi) if blobs[i]]
            vv = np.array([vecs[i] for i in range(lo, hi) if blobs[i]],
                          dtype=np.float32).reshape(len(refs), -1)
            new = []
            for i, t in enumerate(q["probes"]):
                cs = crop_scores(idx.n_frames, refs, vv, QV[t])
                base = SourceScores("visual", q["S"][i],
                                    np.ones(idx.n_frames, dtype=bool))
                new.append(fuse([base, cs], {"visual": 1.0, "crop": 1.0}))
            q["S"] = np.vstack(new).astype(np.float32)

    # ⑥ đầu đọc QA — chỉ cho đề `qa`, một lượt suy luận trên khung top-1 mỗi đề
    qa_jobs, qa_of = [], {}
    if any(q["kind"] == "qa" for q in queries):
        ocr_raw, asr_raw = {}, {}
        with open("data/OCR/ocr.jsonl", encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                t = (d.get("text_normalized") or d.get("text_raw") or "").strip()
                if t:
                    ocr_raw[(d["video_id"], int(d["n"]))] = t
        for f in Path("data/ASR").glob("*/results/*.json"):
            d = json.loads(f.read_text(encoding="utf-8"))
            asr_raw[d["video_id"]] = [(s_["start_ms"], s_["end_ms"], s_["text"])
                                      for s_ in d.get("segments", [])]
        for q in queries:
            if q["kind"] != "qa":
                continue
            r = int(np.argmax(q["S"][0]))
            vid, n = idx.ids[r]
            fp = frame_path(vid, n)
            if fp is None:
                continue
            from PIL import Image as _Im
            im = _Im.open(fp).convert("RGB")
            im.thumbnail((896, 896))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=90)
            ms = fms.get((vid, n))
            asr = " ".join(t for a, b, t in asr_raw.get(vid, [])
                           if ms is not None and a - 5000 <= ms <= b + 5000)[:400]
            qa_of[q["id"]] = len(qa_jobs)
            qa_jobs.append({"image_b64": base64.b64encode(buf.getvalue()).decode(),
                            "question": q["text"], "ocr": ocr_raw.get((vid, n), ""),
                            "asr": asr})
        if qa_jobs:
            print(f"⑥ đầu đọc QA: {len(qa_jobs)} đề")
            answers = read_answer.remote(qa_jobs)
            for q in queries:
                if q["id"] in qa_of:
                    q["answer"] = answers[qa_of[q["id"]]]
                    print(f"   {q['id']}: {q['answer']!r}")

    # ⑤ bậc 2 — VLM chấm top-K sau bậc mảnh cắt
    if vlm_top_k > 0:
        vjobs, vspan = [], {}
        for q in queries:
            if q["kind"] == "trake":
                continue          # TRAKE chấm theo ĐƯỜNG, không theo khung lẻ
            base = q["S"].max(axis=0)
            cand = [int(r) for r in np.argsort(-base)[:vlm_top_k]]
            vspan[q["id"]] = (len(vjobs), len(vjobs) + len(cand), cand)
            for r in cand:
                vid, n = idx.ids[r]
                fp = frame_path(vid, n)
                if fp is None:
                    continue
                from PIL import Image as _Im2
                im = _Im2.open(fp).convert("RGB")
                im.thumbnail((640, 640))
                b = io.BytesIO()
                im.save(b, "JPEG", quality=85)
                vjobs.append({"image_b64": base64.b64encode(b.getvalue()).decode(),
                              "query": q["text"]})
        if vjobs:
            print(f"⑤ bậc 2 (VLM): {len(vjobs)} khung")
            probs = []
            for s_ in range(0, len(vjobs), 200):
                probs += score_frames_vlm.remote(vjobs[s_:s_ + 200])
                print(f"   {min(s_ + 200, len(vjobs))}/{len(vjobs)}")
            for q in queries:
                if q["id"] not in vspan:
                    continue
                lo, hi, cand = vspan[q["id"]]
                vs = vlm_scores(idx.n_frames, cand[:hi - lo], probs[lo:hi])
                base = SourceScores("visual", q["S"][0], np.ones(idx.n_frames, bool))
                q["S"] = fuse([base, vs], {"visual": 1.0, "vlm": VLM_WEIGHT}
                              )[None, :].astype(np.float32)

    # ④⑦ ra đáp án
    odir.mkdir(parents=True, exist_ok=True)
    report = []
    for q in queries:
        if q["kind"] == "trake":
            res = dante_over_videos(q["S"], idx.ranges, times, videos=None)
            lines = []
            for vid, p in res[:top_k]:
                lo, _ = idx.ranges[vid]
                lines.append([vid] + [int(idx.frame_idx[lo + c]) for c in p.cols])
        else:
            s = q["S"][0]
            pool = np.argpartition(-s, min(top_k * 30, len(s) - 1))[:top_k * 30]
            order = pool[np.argsort(-s[pool])]
            seen, moments = set(), []
            for r in order:
                r = int(r)
                vid, n = idx.ids[r]
                # Khử trùng cảnh TẮT mặc định — đo được nó HẠI. Xem khối bình luận
                # ở `DEDUP_NOTE` phía trên.
                g = (vid, shot.get((vid, n), ("n", n))) if dedup else r
                if g in seen:
                    continue
                seen.add(g)
                moments.append(r)
                if len(moments) * spread >= top_k * spread:
                    break
            lines = []
            for r in moments:
                vid, _ = idx.ids[r]
                p_, n_ = nbr[r]
                for f in spread_in_gap(int(FI[r]), p_, n_, spread):
                    lines.append([vid, int(f)])
                    if len(lines) >= top_k:
                        break
                if len(lines) >= top_k:
                    break
        if q["kind"] == "qa":
            # Thể lệ: `<video_id>, <frame_id>, <answer>`. Thiếu answer ⟹ 0 điểm.
            a = q.get("answer", "")
            lines = [r + [a] for r in lines]
        p = odir / f"{q['id']}.csv"
        p.write_text("\n".join(",".join(str(x) for x in r) for r in lines) + "\n",
                     encoding="utf-8")
        report.append({"id": q["id"], "kind": q["kind"], "n_probes": len(q["probes"]),
                       "n_answers": len(lines), "spread": spread if q["kind"] != "trake" else 1,
                       "dedup": bool(dedup),
                       "answer": q.get("answer"), "top1": lines[0] if lines else None})
        print(f"  {q['id']:<20} {q['kind']:<6} {len(lines):>3} đáp án → {p}")
    (odir / "_report.json").write_text(
        json.dumps({"weights": W, "dim": idx.dim, "rerank": bool(rerank),
                    "spread": spread,
                    "queries": report}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n✓ {odir}/ · {len(queries)} file + _report.json")
    print("  cột: video_id, frame_idx[…]  — frame_idx là số khung THẬT, không phải n")
