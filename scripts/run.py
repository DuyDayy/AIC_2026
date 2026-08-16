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
    ③ ma trận S   chuẩn hoá z trong tập CÓ dữ liệu, trọng số 1/0,089/0,132/0,106
    ⑤a rổ         mỗi nguồn đề cử TOP RIÊNG 40 → rổ ~155 khung, là bộ lọc CỨNG
    ⑤b bậc 1      mảnh cắt vật thể qua jina-clip, quét CẢ rổ (đúng màu 54% → 77%)
    ⑤c bậc 2      VLM chấm P(khớp) trên top 30 sau bậc 1 — BẬT mặc định
    ④ kbest       DP thứ tự thời gian, k-best mỗi video rồi sắp toàn cục; λ=0
    ⑥ đầu đọc     CHỈ đề Q&A: Qwen2.5-VL đọc khung + OCR + lời nói → sinh `answer`
    ⑦ nộp         KHÔNG khử trùng; `--spread` quyết có rải khung vào khe hay không

=============================================================================
RỔ CHỌN AI VÀO VÒNG TRONG, RERANKER QUYẾT THỨ HẠNG
=============================================================================

Thị giác mang hệ số `1,0` còn OCR/ASR/vật thể mang `0,09–0,13`, nên một khung chỉ có
bằng chứng **thuần OCR** gần như không bao giờ nổi lên trong điểm đã hợp. [ĐO] 10/100 câu
có video đúng **vắng mặt hoàn toàn** khỏi bài nộp, mà 9/10 nằm ở hạng mốc 16–62.

⑤a cho mỗi nguồn tự đề cử top 40 của riêng nó. Trên đúng 10 câu đó:

    kiến trúc               video đúng vào được   hạng
    hợp điểm + ⑤ (cũ)              4/10           71 · 85 · 92 · 98
    rổ + ⑤b + ⑤c (mới)             8/10           3 · 7 · 13 · 14 · 21 · 42 · 48 · 48

⚠️ Xếp hạng **bằng** top riêng (xen kẽ vòng tròn giữa các nguồn) đã thử và **bác bỏ**:
−0,1430 Final. Rổ và xếp hạng là hai câu hỏi khác nhau — rổ chỉ quyết ai vào vòng trong.

⚠️ **DUNG HỢP 4 NGUỒN VẪN LÀ XƯƠNG SỐNG, KHÔNG BỎ ĐƯỢC.** Khoá `fused4` trong
`RERANK_WEIGHTS` chính là điểm dung hợp của ③ — không phải riêng nguồn thị giác. Đặt nó
về 0 để "kéo điểm hoàn toàn từ reranker" thì **mất 0,045**:

    dung hợp 4 nguồn   mảnh cắt   VLM     Final
          0,0             0,0     0,5    0,4248   ← thứ hạng thuần reranker
          0,0             1,0     1,0    0,3943
          1,0             0,0     0,5    0,4700   ← chốt

Reranker **bổ sung** cho dung hợp chứ không thay thế nó.

=============================================================================
KIS CHÍNH LÀ TRAKE VỚI N=1 — MỘT ĐƯỜNG MÃ, KHÔNG HAI
=============================================================================

`N=1` làm `DP[i,t] = S[i,t] + max_{τ<t}(DP[i−1,τ] − λ(t−τ))` suy biến thành `max_t S[0,t]`.
⑦ lấy `k` đường tốt nhất MỖI video rồi **sắp toàn cục**: `N=1, k` lớn cho đúng KIS,
`N>1, k=1` cho đúng TRAKE. `tests/test_kis_is_trake_n1.py` khoá tính tương đương lại.

⚠️ Hai nhánh cũ KHÔNG tương đương: nhánh TRAKE gọi `dante_over_videos`, vốn trả **một
đường mỗi video** — tức khử trùng theo video, đo được **−12,0pp**.

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

=============================================================================
VÌ SAO PHẢI RẢI KHUNG — ĐÁP ÁN KHÔNG PHẢI KEYFRAME BTC CẤP
=============================================================================

Thể lệ: *"khung hình ngữ nghĩa … KHÁC VỚI I-Frame là khung hình kỹ thuật … đã được
cung cấp cho các đội thi"*, và đoạn đáp án *"thường rất ngắn, thông thường là dưới 10
frame"*. Ví dụ KIS trong thể lệ: `[500, 510]` = 11 khung.

[ĐO] xác suất một cửa sổ `L=9` đặt ngẫu nhiên **chứa sẵn** một keyframe của ta:
**23,5%**. Đó là **trần cứng** của cách nộp thuần keyframe, bất kể truy xuất tốt đến đâu.

[ĐO] Final trên 226 truy vấn, mốc thật lệch ngẫu nhiên trong khe, ngân sách 100:

    rải   số mốc     L=9      L=11     L=15     L=21
      1      100   0,1091   0,1296   0,1750   0,2354   ← nộp thuần keyframe
      5       20   0,2990   0,3386   0,3789   0,4134
      7       14   0,3241   0,3460   0,3701   0,3906
      9       11   0,3274   0,3410   0,3588   0,3769

**Nộp thuần keyframe kém 3 lần**, và trên bài nộp thật của 100 truy vấn gán nhãn tay:

    rải 7 → 0,4477        rải 1 → 0,1366        (−3,3 lần)

🔴 **Mặc định hiện tại là `--spread 1` THEO YÊU CẦU.** Đây không phải vấn đề xếp hạng nên
reranker không cứu được: trần hình học của bài nộp thuần keyframe là ~23,5%, mà rải 1 đã
ở 0,1366 — dùng 58% của trần. Bản rải 7 ở 0,4477, **cao gấp đôi cái trần ấy**.
Bật lại bằng `--spread 7`.

Trong vùng còn rải, `6` tới `9` **không phân biệt được** (0,4464–0,4488; `6` hơn `7` đúng
+0,0021 với KTC95 [−0,0092, +0,0136]). Chọn 7 vì nằm giữa vùng phẳng.

⚠️ Lý lẽ cũ ở đây SAI và đã bị thay: tôi từng phá hoà 7/9 bằng hai mô hình `chặt ±4` và
`cửa sổ khe`, nhưng **cả hai đều ôm keyframe ground truth**, nên chúng tự thưởng cho
phương án phát ra nhiều keyframe hơn (14 so với 11). Chỉ mô hình **mốc lệch** so được.

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
# ⑤ RỔ ỨNG VIÊN — mỗi nguồn đề cử top riêng của nó, hợp lại.
# Vì sao không lấy top của điểm ĐÃ HỢP: thị giác mang hệ số 1,0 còn OCR/ASR/vật thể mang
# 0,09–0,13, nên khung chỉ có bằng chứng thuần OCR gần như không bao giờ nổi lên. Đo được
# 10/100 câu có video đúng VẮNG MẶT khỏi bài nộp, mà 9/10 nằm ở hạng 16–62.
POOL_PER_SOURCE = 40      # 4 nguồn × 40 ⟹ rổ ≤ 160 khung sau khi khử trùng
POOL_CAP = 200            # chặn trên, cắt theo điểm nền SAU khi mỗi nguồn đã có suất

# Hạn ngạch mỗi nguồn được đề cử tối đa bao nhiêu khung của CÙNG một video.
# [ĐO] Đây là CHỐT AN TOÀN, KHÔNG phải phép tối ưu — 10 đo được Δ=+0,0014 với
# KTC95 [−0,0019, +0,0061], tức bằng 0. Siết chặt hơn thì HẠI: 5 → −0,0184, 2 → −0,0456,
# đổi lại chỉ vớt được nhóm A từ 0,0000 lên 0,0250. Giữ 10 vì nó chặn được ca bệnh lý
# (một nguồn cho điểm phẳng cả video chiếm sạch 40 suất bằng nhiễu) mà chi phí bằng 0.
POOL_PER_VIDEO = 10

# Rổ là bộ lọc CỨNG: chỉ khung trong rổ mới được nộp. Cộng biên này để chúng luôn đứng
# trên mọi khung ngoài rổ mà vẫn giữ được thứ tự nội bộ phần ngoài — DANTE của TRAKE cần
# ma trận dày, nên không thể đặt −inf.
POOL_MARGIN = 1000.0

# Trọng số hợp điểm SAU rerank. Cả ba đi qua cùng phép chuẩn hoá z của ③ nên trọng số
# chỉ diễn đạt MỨC TIN CẬY.
#
# [ĐO] Quét đủ 63 tổ hợp trên 100 truy vấn gán nhãn tay, mô hình mốc lệch, so với nền
# «hợp điểm, không rerank» = 0,4485:
#
#   dung hợp  mảnh cắt   VLM     Final       Δ
#     1,0      0,0     0,5    0,4700   +0,0215   KTC95 [+0,0040, +0,0408]  22T/8Th  ✓
#     1,0      0,0     1,0    0,4593   +0,0108   KTC95 [−0,0128, +0,0351]           ✗
#     1,0      0,0     0,0    0,4513   +0,0028                                      ✗
#     0,3      1,0     1,0    0,4184   −0,0301                                      ✗
#     0,0      2,0     0,0    0,2331   −0,2154                                      ✗
#
# **BẬC 1 (mảnh cắt) PHÁ ĐIỂM.** Mọi tổ hợp có `crop ≥ 0,5` đều tụt, và càng nặng càng
# tệ. Nên `crop = 0,0`: bỏ hẳn bậc đó, vừa tốt hơn vừa rẻ hơn $0,76 và ~8 phút mỗi lượt.
#
# ⚠️ Điều này KHÔNG mâu thuẫn phép đo cũ "màu đúng 54% → 77%": phép đo đó chấm bằng mắt
# trên 8 truy vấn THUẦN MÀU, còn bộ 100 câu này gần như không có câu nào phụ thuộc màu.
# Mảnh cắt nhiều khả năng vẫn lợi ở loại đề đó — bật lại bằng cách đặt `crop > 0`.
#
# ─── KIỂM CHÉO: lấy đỉnh là KHỚP QUÁ, và đo được khớp quá bao nhiêu ────────────
# Quét rồi lấy đỉnh trên CHÍNH bộ đo là thiên lệch chọn lọc. Kiểm chéo 5 lớp × 20 lần
# xáo, chọn trên 80 câu và chấm trên 20 câu chưa thấy:
#
#     đỉnh trên bộ đầy đủ (2,0 / 0,0 / 0,25)   Δ = +0,0241   ← THIÊN LỆCH
#     kiểm chéo                                Δ = +0,0175   KTC95 [+0,0147, +0,0201] ✓
#     lượng KHỚP QUÁ                               +0,0066   (27% mức lợi biểu kiến)
#
# Lợi ích sống sót qua kiểm chéo, nhưng **con số thật là +0,0175, không phải +0,0241**.
#
# `crop = 0` VỮNG: 88/100 lần chia lớp đều chọn tổ hợp có `crop = 0,0`.
#
# VÙNG PHẲNG rộng — 0,4767 xuống 0,4739 trải khắp `fused4 ∈ [0,5; 2,0]` và
# `vlm ∈ [0,25; 1,0]`, tức tỉ lệ 2:1 tới 8:1 đều như nhau. Chỉ TỈ LỆ có nghĩa (phép
# chuẩn hoá z đã đưa mọi nguồn về cùng đơn vị), nên chuẩn hoá `fused4 = 1` rồi lấy
# `vlm = 0,25` — giá trị được chọn ở 63/100 lớp. Đó là điểm ỔN ĐỊNH giữa vùng phẳng,
# KHÔNG phải đỉnh.
RERANK_WEIGHTS = {"fused4": 1.0, "crop": 0.0, "vlm": 0.25}

VLM_TOP_K = 30            # bậc 2 chỉ chấm top-K SAU bậc 1 — thác nước, không quét cả rổ

# TRAKE: mỗi video nộp bao nhiêu ĐƯỜNG. `1` = chỉ đường DP tốt nhất, đúng hành vi cũ.
# CHƯA CHỈNH ĐƯỢC và sẽ không chỉnh được cho tới khi có bộ eval TRAKE: hiện chỉ có 6 truy
# vấn, p=0,69. Đánh đổi ở đây cùng dạng với `spread` — nhiều đường/ít video so với ít
# đường/nhiều video — và mọi đánh đổi cùng dạng đã đo đều rơi vào vùng phẳng. Đặt 1 vì đó
# là hành vi đã chạy, không vì có bằng chứng nó tốt hơn.
TRAKE_K_PER_VIDEO = 1
CROP_BATCH = 400          # ~10 MB base64 mỗi lượt gọi Modal
VLM_BATCH = 200

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
         spread: int = 1, vlm_top_k: int = VLM_TOP_K):
    import numpy as np

    from src.ingestion.jina_encoder import truncate_and_normalize
    from src.retrieval.dante import DEFAULT_LAMBDA
    from src.retrieval.pool import union_pool
    from src.retrieval.probe import build_probes
    from src.retrieval.rerank import collect_crops, crop_scores, vlm_scores
    from src.retrieval.score_matrix import DEFAULT_WEIGHTS, fuse
    from src.submission.coverage import spread_in_window
    from src.submission.kbest import k_best_alignments
    from src.ingestion.vector_index import load_flat_index
    from src.retrieval.sources import (
        AsrSource, SourceScores, TextSource, VisualSource, load_asr_segments,
        load_frame_ms, load_object_text, load_ocr_text, load_shot_bounds,
        load_video_last_frame,
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
    fms = load_frame_ms()
    times = np.array([fms.get(k, 0.0) for k in idx.ids], dtype=np.float64)
    # Hàng xóm THỜI GIAN trong cùng video — cần để dựng cửa sổ rải theo mật độ
    # cục bộ. Khe đo được p10=19, p50=48, p90=105 khung nên bước cố định là sai.
    FI = np.asarray(idx.frame_idx)
    # Cửa sổ rải của mỗi keyframe = NỬA KHE tới hàng xóm, GIAO với biên CẢNH của nó,
    # rồi kẹp vào số khung thật của video. Hai phép giao sau không phải trang trí:
    # thiếu chúng thì bài nộp vắt qua ranh giới cảnh và vượt cuối video — cả hai đã
    # bị `writer.validate_all` bắt trên bài nộp thật.
    shot_b = load_shot_bounds()
    last_f = load_video_last_frame()
    win: dict[int, tuple[int, int]] = {}
    for v, (lo, hi) in idx.ranges.items():
        o = np.argsort(FI[lo:hi])
        f = FI[lo:hi][o]
        vmax = last_f.get(v, int(f[-1]))
        for j, r in enumerate(o):
            row = lo + int(r)
            c = int(f[j])
            prev = int(f[j - 1]) if j else c
            nxt = int(f[j + 1]) if j + 1 < len(f) else c
            a, b = c - (c - prev) // 2, c + (nxt - c) // 2
            ss, se = shot_b.get(idx.ids[row], (a, b))
            win[row] = (max(0, a, ss), min(b, se, vmax))
    # ĐỒNG HỒ TỪNG TẦNG. Kỳ thi chỉ có 2 giờ 30, nên "chạy được" chưa đủ — phải biết
    # mỗi tầng ăn bao nhiêu phút để cắt đúng chỗ khi thiếu giờ, chứ không cắt mò.
    T: dict[str, float] = {"nạp": time.time() - t0}
    _tick = [time.time()]

    def lap(name: str) -> None:
        now = time.time()
        T[name] = now - _tick[0]
        _tick[0] = now
        print(f"    ⏱ {name}: {T[name]:.0f}s", flush=True)

    print(f"nạp {time.time() - t0:.0f}s · {idx.n_frames:,} khung · {idx.dim} chiều "
          f"· trọng số {W}")

    # ②③ chấm điểm — GIỮ LẠI điểm từng nguồn, không hợp rồi vứt.
    # Điểm hợp `q["S"]` vẫn cần: ④ DANTE chạy trên nó, và nó là điểm nền phá hoà trong
    # rổ. Nhưng việc CHỌN AI VÀO VÒNG TRONG thì do từng nguồn tự đề cử.
    for q in queries:
        rows, per_probe = [], []
        for t in q["probes"]:
            ss = [vis.score(QV[t])] + [s.score(t) for s in tsrc]
            per_probe.append(ss)
            rows.append(fuse(ss, W))
        q["S"] = np.vstack(rows).astype(np.float32)
        q["sources"] = per_probe
    lap("②③ chấm 4 nguồn")

    # ⑤a RỔ ỨNG VIÊN — hợp top riêng của từng nguồn, gộp qua mọi probe của truy vấn
    for q in queries:
        got: set[int] = set()
        prov: dict[int, tuple[str, ...]] = {}
        for i, ss in enumerate(q["sources"]):
            pr = union_pool(ss, per_source=POOL_PER_SOURCE, base=q["S"][i], cap=POOL_CAP,
                            ranges=idx.ranges, per_video=POOL_PER_VIDEO)
            got.update(pr.rows.tolist())
            for r, src_names in pr.provenance.items():
                prov[r] = tuple(sorted(set(prov.get(r, ())) | set(src_names)))
        q["pool"] = np.array(sorted(got), dtype=np.int64)
        q["prov"] = prov
    npool = [len(q["pool"]) for q in queries]
    print(f"⑤a rổ ứng viên: trung vị {int(np.median(npool))} khung/truy vấn "
          f"(min {min(npool)}, max {max(npool)}) · {POOL_PER_SOURCE}/nguồn")
    lap("⑤a dựng rổ")

    # ⑤b bậc 1 — mảnh cắt vật thể, mã hoá bằng jina-clip. Gom mảnh của MỌI truy vấn rồi
    # mã hoá một lượt để đỡ vòng gọi Modal.
    # Trọng số 0 ⟹ **KHÔNG TÍNH LUÔN**, chứ không tính rồi bỏ: bậc này tốn $0,76 và ~8
    # phút cho 100 truy vấn, mà đo được nó phá điểm (xem `RERANK_WEIGHTS`).
    if rerank and RERANK_WEIGHTS.get("crop", 0.0) > 0.0:
        allrefs, span = [], {}
        for q in queries:
            cand = [int(r) for r in q["pool"]]
            refs = collect_crops(cand, idx.ids, "data/objects-full")
            span[q["id"]] = (len(allrefs), len(allrefs) + len(refs))
            allrefs.extend(refs)
        blobs = make_crops(allrefs, idx.ids)
        ok = [i for i, b in enumerate(blobs) if b]
        print(f"⑤ rerank: {len(ok)}/{len(blobs)} mảnh cắt · "
              f"ước tính ${len(ok) * 4.32 / 173426:.2f}")
        lap("⑤b cắt mảnh")
        vecs = np.zeros((len(blobs), idx.dim), dtype=np.float32)
        # `.map` chứ KHÔNG phải vòng lặp `.remote`: 30.397 mảnh chia lô 400 là 76 lượt
        # gọi, và mỗi lượt tải ~10 MB base64. Chạy tuần tự trên MỘT container thì nút cổ
        # chai là ĐƯỜNG TRUYỀN, không phải GPU — đo được ~2,5 phút mỗi lô, tức ~3 giờ cho
        # 100 truy vấn. `.map` để Modal bung nhiều container, các lượt tải chồng lên nhau.
        parts = [ok[s:s + CROP_BATCH] for s in range(0, len(ok), CROP_BATCH)]
        # KHÔNG `zip` thẳng vào `.map()`: `zip` xong là thả tham chiếu, Python đóng
        # generator lúc nó còn chạy ⟹ `RuntimeError: aclose(): asynchronous generator is
        # already running`. Lỗi đó không làm sập, và đó mới là chỗ nguy: nếu `zip` dừng
        # sớm thì các hàng chưa mã hoá **ở nguyên số 0** và sai lặng lẽ. Vét cạn trước.
        got_all = []
        for got in encode_crops.map([[blobs[i] for i in p] for p in parts]):
            got_all.append(got)
            print(f"   {sum(len(g) for g in got_all)}/{len(ok)}", flush=True)
        if len(got_all) != len(parts):
            raise SystemExit(f"⑤b nhận {len(got_all)} lô cho {len(parts)} lô gửi — "
                             f"thiếu lô nào thì vector của nó ở nguyên 0, dừng")
        for part, got in zip(parts, got_all):
            if len(got) != len(part):
                raise SystemExit(f"⑤b lô lệch cỡ: gửi {len(part)}, nhận {len(got)}")
            vecs[part] = truncate_and_normalize(np.asarray(got, np.float32), dim)
        for q in queries:
            lo, hi = span[q["id"]]
            refs = [allrefs[i] for i in range(lo, hi) if blobs[i]]
            vv = np.array([vecs[i] for i in range(lo, hi) if blobs[i]],
                          dtype=np.float32).reshape(len(refs), -1)
            # Giữ điểm bậc 1 lại chứ không hợp ngay: bậc 2 còn phải cộng vào, và hợp
            # hai lần liên tiếp sẽ chuẩn hoá z chồng lên nhau — sai thang.
            q["crop"] = [crop_scores(idx.n_frames, refs, vv, QV[t])
                         for t in q["probes"]]

    # ── HỢP ĐIỂM SAU RERANK ─────────────────────────────────────────────────
    # Một chỗ duy nhất dựng `q["S"]` cuối. Gọi hai lần: sau bậc 1 để bậc 2 biết chấm
    # khung nào, rồi sau bậc 2 để ra điểm cuối. Hợp một lần từ nguồn gốc chứ không hợp
    # chồng lên điểm đã hợp — chuẩn hoá z hai lượt là sai thang.
    def combine(q: dict) -> np.ndarray:
        rows = []
        for i in range(len(q["probes"])):
            parts = [SourceScores("fused4", q["S0"][i],
                                  np.ones(idx.n_frames, dtype=bool))]
            if "crop" in q:
                parts.append(q["crop"][i])
            if "vlm" in q and i == 0:
                parts.append(q["vlm"])
            s = fuse(parts, RERANK_WEIGHTS)
            # Rổ là bộ lọc CỨNG. Biên giữ nguyên thứ tự nội bộ hai phía nên DANTE vẫn
            # chạy được trên ma trận dày.
            s[q["pool"]] += POOL_MARGIN
            rows.append(s)
        return np.vstack(rows).astype(np.float32)

    for q in queries:
        q["S0"] = q["S"]                      # điểm nền của ④ nguồn, giữ nguyên bản
        q["S"] = combine(q)
    lap("⑤b mã hoá + hợp")

    # ⑤c bậc 2 — VLM chấm top-K SAU bậc 1. Thác nước: bậc 1 rẻ quét cả rổ, bậc 2 đắt
    # chỉ chấm phần đầu bậc 1 đã lọc.
    if rerank and vlm_top_k > 0:
        vjobs, vspan = [], {}
        for q in queries:
            if q["kind"] == "trake":
                continue          # TRAKE chấm theo ĐƯỜNG, không theo khung lẻ
            base = q["S"].max(axis=0)
            cand = [int(r) for r in np.argsort(-base)[:vlm_top_k]]
            got = []
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
                got.append(r)
            # `got` chứ không phải `cand`: khung thiếu file ảnh bị bỏ, và nếu span đếm
            # theo `cand` thì mọi truy vấn sau đó lệch một — lỗi im lặng, hỏng cả bài.
            vspan[q["id"]] = (len(vjobs) - len(got), len(vjobs), got)
        if vjobs:
            print(f"⑤c bậc 2 (VLM): {len(vjobs)} khung · "
                  f"top {vlm_top_k} sau bậc 1")
            # `.map` giữ NGUYÊN thứ tự đầu vào — `vspan` đánh chỉ số theo vị trí trong
            # `vjobs` nên đảo thứ tự là gán nhầm điểm cho truy vấn khác, im lặng.
            probs = []
            vparts = [vjobs[s_:s_ + VLM_BATCH]
                      for s_ in range(0, len(vjobs), VLM_BATCH)]
            for got in score_frames_vlm.map(vparts):
                probs += got
                print(f"   {len(probs)}/{len(vjobs)}", flush=True)
            if len(probs) != len(vjobs):
                raise SystemExit(f"⑤c trả {len(probs)} điểm cho {len(vjobs)} khung — "
                                 f"lệch thì `vspan` gán nhầm truy vấn, dừng")
            for q in queries:
                if q["id"] not in vspan:
                    continue
                lo, hi, got = vspan[q["id"]]
                if not got:
                    continue
                q["vlm"] = vlm_scores(idx.n_frames, got, probs[lo:hi])
                q["S"] = combine(q)

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

    # ④⑦ ra đáp án
    odir.mkdir(parents=True, exist_ok=True)
    report = []
    for q in queries:
        # ─────────────────────────────────────────────────────────────────────
        # MỘT đường duy nhất cho MỌI loại đề. TRAKE với N=1 CHÍNH LÀ KIS.
        #
        # Trước đây đây là hai nhánh, và chúng KHÔNG tương đương ở khâu nộp dù tương
        # đương ở khâu DP: nhánh TRAKE cũ gọi `dante_over_videos`, vốn trả **một đường
        # mỗi video** — tức đã khử trùng theo video, mà khử trùng video đo được −12,0pp.
        # Hợp nhất đúng cách là lấy `k` đường tốt nhất MỖI video rồi **sắp toàn cục**:
        #   N=1, k lớn  ⟹ đúng bằng KIS (mọi khung trong rổ, sắp theo điểm)
        #   N>1, k=1    ⟹ đúng bằng TRAKE cũ
        # Rổ là bộ lọc cứng nên ứng viên đã ít, `k_best_alignments` chạy trên vài chục
        # khung mỗi video chứ không trên cả lát.
        # ─────────────────────────────────────────────────────────────────────
        n_mom = len(q["probes"])          # N mốc: KIS/QA là 1, TRAKE là N
        k_per_video = 10**9 if n_mom == 1 else TRAKE_K_PER_VIDEO
        by_video: dict[str, list[int]] = {}
        for r in q["pool"].tolist():
            by_video.setdefault(idx.ids[r][0], []).append(r)
        scored: list[tuple[float, str, tuple[int, ...], int]] = []
        for vid, rows in by_video.items():
            # Sắp theo `frame_idx`: thể lệ đòi bộ khung TĂNG DẦN, và `frame_idx` mới là
            # số khung thật. Sắp theo `n` là sai — 20/873 video có `pts_time` tụt khi
            # `n` tăng. Khử trùng `frame_idx` vì `k_best_alignments` đòi tăng NGHIÊM NGẶT.
            rows = sorted(set(rows), key=lambda r: (int(FI[r]), r))
            seen_f, keep = set(), []
            for r in rows:
                f = int(FI[r])
                if f in seen_f:
                    continue
                seen_f.add(f)
                keep.append(r)
            if len(keep) < n_mom:
                continue
            cand = [int(FI[r]) for r in keep]
            row_of = {f: r for f, r in zip(cand, keep)}
            sc = q["S"][:, keep]
            # ④ DANTE nằm ở ĐÂY. `k_best_alignments` là bản k-best của đúng phép quy
            # hoạch động `DP[i,t] = S[i,t] + max_{τ<t}(DP[i−1,τ] − λ(t−τ))`; với k=1 nó
            # trùng khớp `dante()`. Truyền λ để giữ nguyên ngữ nghĩa ④, không đánh rơi.
            for al in k_best_alignments(cand, sc.tolist(),
                                        k=min(k_per_video, len(keep)),
                                        pacing_penalty=DEFAULT_LAMBDA):
                scored.append((float(al.score), vid, tuple(int(f) for f in al.frames),
                               row_of[int(al.frames[0])]))
        # Phá hoà bằng `(vid, frames)` để hai lần chạy cho cùng một bài nộp.
        scored.sort(key=lambda x: (-x[0], x[1], x[2]))

        if n_mom > 1:
            lines = [[vid] + list(fr) for _s, vid, fr, _r in scored[:top_k]]
        else:
            moments = [r for _s, _v, _f, r in scored]
            # KHÔNG khử trùng cảnh — đo được nó hại 4,3pp, xem docstring.
            # Hai mốc KỀ NHAU rải chạm nhau ở điểm giữa khe, nên có thể phát ra
            # cùng một (video, frame). Nộp trùng là **phí slot**: hàm chấm lấy
            # `R@k = max`, hai dòng giống hệt mua đúng MỘT cơ hội.
            # [ĐO] không lọc: 266/10.000 dòng trùng (2,7%), tệ nhất 8 dòng một bài,
            # chỉ 7/100 bài sạch. `writer.validate_all` bắt được lỗi này.
            lines, emitted = [], set()
            for r in moments:
                vid, _ = idx.ids[r]
                wlo, whi = win[r]
                c = int(FI[r])
                for f in spread_in_window(c, min(wlo, c), max(whi, c), spread):
                    k = (vid, int(f))
                    if k in emitted:
                        continue
                    emitted.add(k)
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
                       "answer": q.get("answer"), "top1": lines[0] if lines else None})
        print(f"  {q['id']:<20} {q['kind']:<6} {len(lines):>3} đáp án → {p}")
    # ── LƯU ĐIỂM TRUNG GIAN CỦA ⑤ ───────────────────────────────────────────
    # Không có bản lưu này thì mỗi lần quét `RERANK_WEIGHTS`/`VLM_WEIGHT` phải chạy lại
    # GPU. Chỉ lưu phần TRONG RỔ nên rất nhẹ: ~156 khung × 100 truy vấn.
    if rerank:
        dump: dict[str, np.ndarray] = {}
        for q in queries:
            p = q["pool"]
            dump[f"{q['id']}/rows"] = p.astype(np.int64)
            dump[f"{q['id']}/base"] = q["S0"][:, p].astype(np.float32)
            if "crop" in q:
                dump[f"{q['id']}/crop"] = np.vstack(
                    [c.scores[p] for c in q["crop"]]).astype(np.float32)
                dump[f"{q['id']}/crop_cov"] = np.vstack(
                    [c.covered[p] for c in q["crop"]])
            if "vlm" in q:
                dump[f"{q['id']}/vlm"] = q["vlm"].scores[p].astype(np.float32)
                dump[f"{q['id']}/vlm_cov"] = q["vlm"].covered[p]
            # Số nguồn đã đề cử mỗi khung — tín hiệu tin cậy, KHÔNG dùng làm điểm
            # (đo được cộng vào điểm thì hại 0,0208; xem README mục ③).
            dump[f"{q['id']}/agree"] = np.array(
                [len(q["prov"].get(int(r), ())) for r in p], dtype=np.int8)
        np.savez_compressed(odir / "_rerank_scores.npz", **dump)
        print(f"  đã lưu điểm ⑤ trung gian → {odir}/_rerank_scores.npz "
              f"({(odir / '_rerank_scores.npz').stat().st_size / 1e6:.1f} MB)")

    (odir / "_report.json").write_text(
        json.dumps({"weights": W, "dim": idx.dim, "rerank": bool(rerank),
                    "spread": spread,
                    "pool_per_source": POOL_PER_SOURCE, "pool_cap": POOL_CAP,
                    "rerank_weights": RERANK_WEIGHTS,
                    "vlm_top_k": vlm_top_k if rerank else 0,
                    "queries": report}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n✓ {odir}/ · {len(queries)} file + _report.json")
    print("  cột: video_id, frame_idx[…]  — frame_idx là số khung THẬT, không phải n")
