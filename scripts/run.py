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

    ① probe hoá   tách mốc E1:/E2:, rút trích dẫn; Q&A: câu hỏi → câu MÔ TẢ
    ② bốn nguồn   thị giác · OCR · vật thể · lời nói
    ③ ma trận S   chuẩn hoá z trong tập CÓ dữ liệu, trọng số 1/0,089/0,132/0,213
    ⑤a rổ         mỗi nguồn đề cử TOP RIÊNG 40 → rổ 158 khung, là bộ lọc CỨNG
    ⑤b bậc 1      mảnh cắt vật thể — BỎ: trống cả ở nhóm ≥2 màu, tốn 655s + $0,77
    ⑤c bậc 2      VLM chấm P(khớp) trên CẢ RỔ — BẬT mặc định, trọng số 0,25
    ④ kbest       DP thứ tự thời gian, k-best mỗi video rồi sắp toàn cục; λ=0
    ⑥ đầu đọc     CHỈ đề Q&A: Qwen2.5-VL đọc khung + OCR + lời nói → sinh `answer`
    ⑦ nộp         MỘT khung mỗi mốc (`frame_idx` thật); KHÔNG rải, KHÔNG khử trùng cảnh

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
⑦ NỘP MỘT KHUNG MỖI MỐC — THIẾT KẾ, KHÔNG PHẢI TUỲ CHỌN
=============================================================================

⑦ nộp **đúng một dòng cho mỗi keyframe được chọn**: `frame_idx` của chính nó. Không có
phép rải, và **không có cờ để bật lại** — đây là hình dạng của tầng nộp, không phải một
chế độ trong nhiều chế độ.

VÌ SAO KHÔNG ĐỂ CỜ. Một cờ `--spread` sẽ nói dối về bản chất bài toán. Rải không phải
cách xếp hạng tốt hơn; nó là cách **tiêu 100 suất ngân sách để phủ khoảng trống giữa các
keyframe**. Khoảng trống đó là thuộc tính của **bộ keyframe**, không phải của tầng truy
vấn. Nên chỗ giải nó là lúc CẮT KHUNG, và một cờ ở đây chỉ mời người ta lấy ngân sách
đắp cho dữ liệu thiếu — rồi tưởng mình đang chỉnh mô hình.

ĐÁNH ĐỔI, ĐO ĐƯỢC, KHÔNG GIẤU. Ở mật độ keyframe hiện tại thiết kế này **kém hơn**:

    [ĐO] GT v2 100 câu, L=11, cùng cấu hình còn lại, cùng hạt giống:
         một khung mỗi mốc → 0,1115   ·   rải 7 → 0,4496          (−75%)
    [ĐO] bộ giữ kín 110 câu:  rải 1 → 0,0857   ·   rải thích ứng → 0,2334   (−63%)

    Cơ chế đọc theo CỘT R@k: R@1 gần như y nhau (0,0546 so với 0,0500) rồi bản một-khung
    ĐỨNG IM từ R@5. Tức 99/100 suất ngân sách gần như vô giá trị ở cấp khung — chúng là
    keyframe cách nhau ~48 khung, mà chỉ một cửa sổ ~11 khung được tính.

ĐIỀU KIỆN ĐỂ THIẾT KẾ NÀY ĐÚNG. Trần phủ là hàm của mật độ keyframe:

    khe keyframe            trần ở L=9    trần ở L=11
    48 (hiện tại)               23,5%         28,6%
    24 (×2 dày)                 45,7%         53,1%
    12 (×4 dày)                 73,5%         81,1%
    10 (mỗi khung thứ 10)       90,0%        100%

Ở ×4 dày, mỗi keyframe tự phủ cửa sổ của nó và rải mất hết ý nghĩa. Cắt dày hơn **hoá
giải đúng cái trần** mà rải sinh ra để vượt — nên hai thứ không cộng dồn, chúng thay nhau.

🔴 CHỖ HỔNG THẬT KHÔNG PHẢI XẾP HẠNG. `R@1` cấp shot đo được **0,5600** so với `R@1` cấp
khung **0,0546**: hệ tay đúng shot ở hạng 1 cho 56% câu, và khung hạng 1 lệch keyframe
ground truth **trung vị 5 khung**. 50/100 câu mất điểm dù xếp hạng đã hoàn hảo. Đó là lý
lẽ định lượng cho việc cắt dày hơn, và là lý do không nên tiêu ngân sách để đắp.

KHÔNG XOÁ MÃ PHÂN TÍCH. `src/submission/coverage.py` vẫn giữ `spread_in_window`,
`adaptive_m` và phép tính trần ở trên — nhưng nó **không được import từ đường chạy**, và
`tests/test_coverage.py` khoá hành vi đó lại. Cần đo lại đánh đổi này thì dùng module đó,
đừng thêm cờ vào đây.

Bài nộp ghi **`frame_idx`** — số khung thật. `n` chỉ là số thứ tự keyframe, và đo được
0/173.426 khung có hai giá trị bằng nhau.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
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

# ③ DUNG HỢP — RRF CÓ TRỌNG SỐ HAI TẦNG, không còn chuẩn hoá z.
#
#     Score(d) = Σ_m alpha_m · ( Σ_j beta_mj / (k + rank_mj(d)) )
#     Σ alpha_m = 1  ·  Σ_j beta_mj = 1 cho từng modality
#
# Trọng số áp vào ĐÓNG GÓP RRF, không áp vào raw score: cosine và BM25 không bao giờ
# được cộng trực tiếp. Chuẩn hoá beta trong từng modality ngăn modality nhiều expansion
# hơn tự có nhiều quyền vote hơn — [ĐO] cấu hình phẳng tụt 0,4427 so với 0,5281.
RRF_K = 60.0

# alpha ĐỀU, và đó là kết quả ĐO chứ không phải chỗ chưa tinh chỉnh. E4 quét trọng số
# toàn cục trên 210 câu tuning rồi báo trên 100 câu held-out chưa đụng tới:
#     alpha* tìm được = (0,30 · 0,35 · 0,35) ≈ đều
#     held-out: đều 0,5281 → có trọng số 0,4884
#     ΔMRR bắt cặp −0,0398 · KTC95 [−0,0783, −0,0056] · thắng 10 / thua 18
# Khoảng tin cậy nằm TRỌN dưới 0: trọng số toàn cục kém hơn CÓ Ý NGHĨA.
ALPHA = {"visual": 1 / 3, "ocr": 1 / 3, "asr": 1 / 3}

# beta trong từng modality: (câu gốc, câu mở rộng). [ĐO] E3 trên bench_kis:
#     bỏ bản gốc, chỉ QE          0,4867
#     giữ cả hai, 0,5/0,5   ★     0,5281
# Expansion BỔ SUNG cho câu gốc chứ không thay thế nó.
def _beta_for(n_runs: int) -> tuple[float, ...]:
    """Trọng số TRONG một modality: mọi nhánh đóng góp ĐỀU, kể cả câu gốc.

    Đều là mức có căn cứ, không phải mức mặc định cho tiện. E3 đo trên bench_kis với hai
    nhánh mỗi modality (gốc + một expansion) và `(0,5 · 0,5)` — tức ĐỀU — là cấu hình
    thắng: MRR 0,5281 so với 0,4867 khi bỏ câu gốc.

    Ở đây từng có `(0,5 · 0,25 · 0,25)` cho ba nhánh, ưu ái câu gốc. Nó lấy từ hạt giống
    ở playbook mục 7.1 chứ KHÔNG phải từ phép đo nào — và ưu ái một nhánh mà không có
    bằng chứng thì chỉ là một giả định đội lốt hằng số. Đều thì trùng đúng giá trị E3 đo
    được ở `n=2`, nên nó vừa là mặc định trung tính vừa nhất quán với số liệu.

    Muốn lệch khỏi đều thì phải ĐO trên tập giữ kín trước — E4 đã cho thấy trọng số toàn
    cục lệch khỏi đều làm HẠI có ý nghĩa (ΔMRR −0,0398, KTC95 trọn dưới 0).
    """
    return (1.0 / n_runs,) * max(n_runs, 1)


# ⑤a RỔ ỨNG VIÊN — top-K của điểm ĐÃ HỢP
#
# 🔁 ĐÃ ĐẢO. Bản cũ cho MỖI RUN một hạn ngạch top-40 riêng rồi hợp lại (`union_pool`).
# Nó ra đời để chữa phép hợp CŨ: thị giác hệ số 1,0 còn OCR/ASR 0,09–0,13 nên khung
# thuần OCR không nổi lên trong điểm hợp — đo được 10/100 câu có video đúng VẮNG MẶT
# khỏi bài nộp, 9/10 nằm ở hạng 16–62, và cấp suất riêng lãi +0,0153.
#
# Tiền đề ấy đã tan. ③ nay là RRF phân cấp, `alpha` chia ĐỀU (OCR nắm đúng 1/3 lá phiếu,
# ngang thị giác) và cộng theo HẠNG nên thang điểm thô của từng nguồn không còn nghĩa.
# Khung đứng #1 bảng OCR nay nổi lên TRONG CHÍNH điểm hợp — đúng thứ hạn ngạch phải đi
# vòng để đạt. Giữ hạn ngạch lúc này là chữa bệnh đã khỏi, mà giá thì còn nguyên:
#   · rổ không còn là top-K của thứ gì ⟹ `POOL_CAP` cắt theo một trật tự KHÁC trật tự
#     đưa vào. Đó chính là chỗ nó cắt câm 25% rổ ở mọi truy vấn mà không ai thấy.
#   · nguồn yếu vẫn tiêu đủ 40 suất, bất kể yếu tới đâu — hạn ngạch không biết nhường
#   · số ứng viên nở theo SỐ RUN: thêm một bản mở rộng là thêm 40 suất, dù nó gần trùng
#
# `POOL_CAP` nay là phép cắt CHỦ ĐỘNG (rổ = đúng top-`POOL_CAP`), không còn là chốt an
# toàn — nên nó thành một tham số THẬT, cần quét, chứ không phải một con số đặt cho to.
POOL_CAP = 300

# 🔴 POOL_PER_VIDEO ĐÃ RỜI ĐƯỜNG CHẠY — hằng số giữ lại chỉ để `union_pool` còn dùng
# được trong nghiên cứu. Nó là hạn ngạch "một video được góp tối đa bao nhiêu khung",
# sinh ra để chặn một NGUỒN PHẲNG đổ cả trăm khung cùng điểm của một video vào rổ.
#
# Với rổ = top-K của điểm đã hợp thì không còn nguồn phẳng nào để chặn: thứ tự trong rổ
# LÀ thứ hạng của ③, nên khung thứ 11 của một video vào rổ đúng khi ③ thật sự xếp nó
# trên khung đầu của video khác. [ĐO] bench_kis 100 câu, độ phủ khung GT trong rổ 300:
#     không hạn ngạch  98,0%   ← ĐÚNG BẰNG trần lý thuyết (98/100 câu có GT trong top-300)
#     per_video = 40   98,0%
#     per_video = 20   95,0%
#     per_video = 10   92,0%   ← giá của hạn ngạch: −6,0pp, đổi lại KHÔNG được gì
# Ở `union_pool` hạn ngạch áp cho TỪNG NGUỒN nên 7 run × 10 = tối đa 70 khung/video;
# bê nguyên số 10 sang phép hợp một lần là siết chặt gấp 7 lần trong im lặng.
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
#
# ⚠️ GỐC GÁC CỦA `0,25` ĐÃ HẾT HIỆU LỰC. Con số đó rút khi ⑤ hợp bằng chuẩn hoá z, nơi
# `fused4` và `vlm` đều được đưa về `E=0, sd=1` trước khi cộng. Nay ⑤ hợp bằng RRF, nên
# `0,25` là trọng số TẦNG BA của RRF — cùng đơn vị với `alpha` của ③, khác đơn vị với
# thứ nó được quét ra. Phân bố điểm nền đổi hẳn, nên phải QUÉT LẠI trước khi tin.
# Quét lại cần một lượt Modal; đến lúc đó `0,25` chỉ là giá trị mang sang, không phải
# giá trị đã đo trong kiến trúc hiện tại.
RERANK_WEIGHTS = {"fused4": 1.0, "crop": 0.0, "vlm": 0.25}

# ⑤c chấm top-K của rổ. [ĐO] trên bộ GIỮ KÍN 110 câu (chưa dùng để chọn gì), bắt cặp
# theo truy vấn, bootstrap 4000:
#
#     L      bỏ VLM     K=30     K=160    Δ bỏ→160              KTC95      T/H/Th
#     9      0,2309   0,2493   0,2651     +0,0342   [+0,0077,+0,0618]    18/85/7  ✓
#     11     0,2445   0,2635   0,2802     +0,0356   [+0,0091,+0,0659]    18/85/7  ✓
#     21     0,2713   0,2907   0,3101     +0,0388   [+0,0111,+0,0711]    18/84/8  ✓
#     51     0,3118   0,3344   0,3564     +0,0447   [+0,0145,+0,0791]    18/84/8  ✓
#
# ⚠️ Bốn cột SỐ TUYỆT ĐỐI ở bảng trên đo bằng mô hình cửa sổ CHƯA chặn biên shot, nên
# thấp hơn thực tế (bộ này ra 0,2605 thay vì 0,2778 ở L=11). Δ bắt cặp giữ được vì cả
# hai nhánh dùng cùng cửa sổ; muốn số tuyệt đối đúng thì chạy lại cả sweep.
#
# Bỏ VLM mất ~13% điểm tương đối. K=30 mua 0,019; nới lên 160 mua thêm 0,017 — chưa
# thấy bão hoà, nên chấm CẢ RỔ. `VLM_TOP_K` ≥ `POOL_CAP` ⟹ phủ hết rổ.
#
# ⚠️ **VLM là cú can thiệp MẠNH và HIẾM, không phải bộ chỉnh êm.** 85/110 câu nó không
# đụng tới; toàn bộ khoản lợi đến từ **18 câu thắng, 7 câu thua** — tỉ lệ 2,6 ăn 1. Hệ
# quả: với đề 35 câu thì kỳ vọng chỉ ~8 câu bị đụng, nên `+0,0356` là kỳ vọng DÀI HẠN,
# một kỳ thi đơn lẻ lệch khá xa cả hai chiều được.
#
# ⚠️ Vì thắng/thua là 2,6:1 chứ không phải luôn đúng, `RERANK_WEIGHTS["vlm"] = 0,25`
# đang giữ nó đúng vai trò CHỈNH chứ không LẬT. Nâng trọng số là khuếch đại cả 18 lần
# đúng lẫn 7 lần sai.
#
# Chi phí: ~7 giây/câu ⟹ ~4 phút cho 35 câu, tức ~3% ngân sách 2h30.
VLM_TOP_K = 160

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
#: GHIM REVISION. `from_pretrained` không ghim sẽ kéo bản MỚI NHẤT trên HuggingFace —
#: đã quan sát: nó tự tải bản mã remote mới ngay lần gọi đầu ở một script khác. Vector
#: KHUNG mã hoá bằng cặp sha này (`artifacts/embed/embed/manifest.json`, và
#: `PipelineConfig` của frame_extracting dùng đúng cặp đó). Truy vấn mã hoá bằng bản
#: khác thì hai bên nằm ở hai không gian hơi lệch — cosine vẫn ra số, thứ hạng vẫn sắp
#: được, và KHÔNG có gì báo.
MODEL_SHA = "e10d47f5691d0454a0fb5d13f46f2199b74cb436"
CODE_SHA = "39e6a55ae971b59bea6e44675d237c99762e7ee2"

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

    m = AutoModel.from_pretrained(MODEL, trust_remote_code=True,
                                  revision=MODEL_SHA,
                                  code_revision=CODE_SHA).to("cuda").eval()
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

    m = AutoModel.from_pretrained(MODEL, trust_remote_code=True,
                                  revision=MODEL_SHA,
                                  code_revision=CODE_SHA).to("cuda").eval()
    ims = [Image.open(_io.BytesIO(b64.b64decode(b))).convert("RGB") for b in blobs]
    with torch.inference_mode():
        v = np.asarray(m.encode_image(ims, batch_size=32), dtype=np.float32)
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    return v.tolist()


QA_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


# ⚠️ ĐÃ THỬ `modal.Cls` + `@modal.enter()` ĐỂ NẠP MODEL MỘT LẦN MỖI CONTAINER — HOÀN NGUYÊN.
# [ĐO] bản này nạp lại Qwen 7B ở MỖI lượt gọi: 110 truy vấn × 160 khung = 88 lô, đếm
# được **176 lần nạp checkpoint**, ~13 phút, trong đó ~90% là nạp chứ không phải suy
# luận. `modal.Cls` lẽ ra chữa đúng chỗ đó.
# Ở quy mô thật (88 lô) nó chết bằng `TimeoutError` phía client kèm
# `App state is APP_STATE_STOPPED`; smoke test 4 lô thì chạy được.
#
# ⚠️ **TÔI ĐÃ QUY KẾT SAI NGUYÊN NHÂN.** Sau đó hai lần chạy KHÁC — không liên quan gì
# tới `modal.Cls` — chết với cùng `App state is APP_STATE_STOPPED`, và lần thứ hai lộ ra
# nguyên nhân thật: `ConnectionError: [Errno 8] nodename nor servname provided`, tức
# **mất DNS/mạng**. Vậy `APP_STATE_STOPPED` là HẬU QUẢ của mất mạng, không phải bằng
# chứng chống lại refactor.
#
# Nên lý do hoàn nguyên còn lại là **chưa chứng minh được ở quy mô thật**, chứ KHÔNG phải
# "đã chứng minh là hỏng". Muốn thử lại thì chạy ≥88 lô khi mạng ổn định; nếu chạy được
# thì nó cắt được ~90% thời gian bậc 2.
# GIỮ BẢN NÀY vì nó **đã chạy được ở đúng quy mô này**. Đường ống ngày thi chạy MỘT lần
# và không sửa được; đổi lấy ~3 phút trong ngân sách 150 phút không đáng rủi ro đó.
# Muốn làm lại thì phải kiểm ở ≥88 lô TRƯỚC, không phải ở 4 lô.
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
        # ⚠️ ĐÃ THỬ đưa kèm OCR + lời nói vào prompt (như ⑥ vẫn làm) — HOÀN NGUYÊN.
        # Lý do nghe rất thuyết phục: ảnh gửi 640px nên chữ lower-third khó đọc, vậy
        # với câu cần chữ thì VLM đang chấm bằng thông tin nó không có.
        # [ĐO] GT v2, 100 câu, L=11: 0,4408 → 0,4304. Và theo loại đề thì NGƯỢC hẳn
        # kỳ vọng — nó hại đúng loại cần OCR:
        #     vision          0,2800 → 0,2960   (+0,016)
        #     vision+ocr      0,5360 → 0,5040   (−0,032)
        #     vision+asr      0,5920 → 0,6320   (+0,040)
        #     vision+ocr+asr  0,6720 → 0,6000   (−0,072)
        # Cơ chế: **giá trị của VLM là làm tín hiệu ĐỘC LẬP.** Nguồn OCR ở ② đã khai
        # thác chính văn bản đó; đưa lại vào đây khiến hai tầng ĐẾM TRÙNG một nguồn
        # nhiễu (ticker tin khác, OCR sai), đổi lại mất phán đoán thị giác độc lập —
        # thứ duy nhất ⑤c đóng góp. ASR ngược lại có lợi (+0,040) vì nguồn ASR ở ②
        # chấm theo cửa sổ ±5s khá thô, nên lời nói ở đây vẫn thêm thông tin mới.
        # Muốn thử lại thì chỉ đưa ASR, đừng đưa OCR.
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
         vlm_top_k: int = VLM_TOP_K, crop_w: float = -1.0,
         expansions: str = "expansions"):
    import numpy as np

    from src.ingestion.jina_encoder import truncate_and_normalize
    from src.retrieval.dante import DEFAULT_LAMBDA
    from src.retrieval.pool import fused_pool
    from src.retrieval.probe import build_probes, declarativize
    from src.retrieval.rerank import collect_crops, crop_scores, vlm_scores
    from src.retrieval.score_matrix import hierarchical_rrf
    from src.submission.kbest import k_best_alignments
    from src.submission.writer import (SubmissionError, TaskSubmission,
                                       pack_submission_zip,
                                       task_type_from_filename, write_task_csv)
    from src.submission.writer import MAX_ANSWER_CHARS
    from src.ingestion.vector_index import load_flat_index
    from src.retrieval.sources import (
        AsrSource, SourceScores, TextSource, VisualSource, load_asr_segments,
        load_frame_ms, load_ocr_text,
    )

    # Ghi đè trọng số bậc 1 từ dòng lệnh. Có cờ này vì phép đo trước tôi làm bằng cách
    # SỬA hằng số rồi phục hồi — nếu lần chạy chết giữa đường thì repo đọng lại trạng
    # thái đo, không phải trạng thái vận hành. `-1` = giữ `RERANK_WEIGHTS`.
    RW = dict(RERANK_WEIGHTS)
    if crop_w >= 0.0:
        RW["crop"] = crop_w
        print(f"⚠ ghi đè trọng số bậc 1: crop = {crop_w}", flush=True)

    qdir, odir = Path(dir), Path(out)
    if not qdir.is_dir():
        raise SystemExit(f"không thấy thư mục {qdir} — tạo nó rồi thả file .txt vào")
    queries = read_queries(qdir)
    if not queries:
        raise SystemExit(f"{qdir} không có file .txt nào")
    print(f"{len(queries)} truy vấn từ {qdir}/")

    # ① probe hoá → danh sách văn bản cần mã hoá
    for q in queries:
        # Đề Q&A: mã hoá CÂU MÔ TẢ, không mã hoá câu hỏi. "…cầm ly màu gì?" hỏi về thứ
        # ta chưa biết — cụm "màu gì" là chỗ trống, không mô tả khung hình nào. Ý lấy từ
        # NII-UIT @ VBS2025 mục 2.7. `q["text"]` giữ NGUYÊN cho ⑥: đầu đọc cần biết đang
        # được hỏi gì mới trả lời được.
        src_text = declarativize(q["text"]) if q["kind"] == "qa" else q["text"]
        if q["kind"] == "qa" and src_text != q["text"]:
            print(f"  ① {q['id']}: probe ← {src_text!r}")
        q["probes"] = [p.text for p in build_probes(src_text)] or [src_text]
        if q["kind"] != "trake" and len(q["probes"]) > 1:
            # KHÔNG đổi loại. Thể lệ: "Quy ước tên file truy vấn — hậu tố kis/qa/trake",
            # nên HẬU TỐ là thứ duy nhất nói loại, và nó quyết định luôn SỐ CỘT của file
            # nộp. Đổi `kind` ở đây từng khiến `query-1-kis.csv` được ghi 4 cột trong khi
            # bộ chấm đọc nó theo luật KIS 2 cột — file vẫn ghi ra, vẫn mở được, và điểm
            # về 0. Phép tách mốc là suy đoán CỦA TA; nó không được phép sửa thể lệ.
            #
            # Các mốc tách ra vẫn dùng, nhưng làm BẰNG CHỨNG: ③ chấm từng mốc rồi ④ gộp
            # bằng `max` (ngay dưới), đúng cách rổ đã gộp qua probe.
            print(f"  ⚠ {q['id']}: tách ra {len(q['probes'])} mốc nhưng hậu tố là "
                  f"{q['kind']} — giữ {q['kind']}, gộp các mốc bằng max ở ④",
                  file=sys.stderr)

    cache = load_cache()
    need = sorted({t for q in queries for t in q["probes"] if key_of(t) not in cache})
    if need:
        print(f"mã hoá {len(need)} đoạn mới trên GPU (đã cache {len(cache)})…")
        for i, v in zip(need, encode_text.remote(need)):
            cache[key_of(i)] = v
        save_cache(cache)
    else:
        print("toàn bộ đã có trong cache — không tốn GPU")
    QV = {t: truncate_and_normalize(np.asarray([cache[key_of(t)]], np.float32), dim)[0]
          for q in queries for t in q["probes"]}

    # ① MỞ RỘNG TRUY VẤN — đọc từ file, KHÔNG gọi LLM lúc thi.
    #
    # Sinh expansion cần một lượt gọi LLM mỗi truy vấn. Trong 2h30 không có lần hai, và
    # proxy đã sập HAI LẦN chỉ trong một buổi làm việc — mỗi lần hơn mười phút, mọi model
    # đều timeout. Nên đường chạy thi chỉ ĐỌC file đã sinh sẵn; sinh là việc làm trước,
    # bằng `scripts/research/generate_expansions.py`.
    #
    # Thiếu file, hoặc thiếu một truy vấn trong file, thì lùi về CÂU GỐC và báo ra. Lùi
    # không trung tính — nhánh expansion khi đó trùng nhánh gốc — nên nó phải nhìn thấy
    # được, không được im lặng.
    # Chấp nhận CẢ HAI dạng, vì chúng phục vụ hai việc khác nhau:
    #   · THƯ MỤC `<id>.<modality><n>.txt` — sửa bằng tay được, grep được, diff được,
    #     đúng cách `queries/` đang làm. Đây là mặc định.
    #   · file JSON — đầu ra thô của `generate_expansions.py`, tiện cho nhánh nghiên cứu.
    # Lấy TẤT CẢ nhánh, không chỉ nhánh đầu: [ĐO] hai expansion của cùng modality có
    # Jaccard token chỉ 0,148 (OCR) và 0,155 (ASR), 0/24 trùng chuỗi — chúng là hai GÓC
    # NHÌN khác nhau, nên bỏ cái thứ hai là vứt đúng phần đa dạng vừa trả tiền để có.
    EXP: dict[str, dict[str, list[str]]] = {}
    exp_path = Path(expansions)
    by_id: dict[str, dict[str, list[str]]] = {}
    if exp_path.is_dir():
        for f in sorted(exp_path.glob("*.txt")):
            # `<query_id>.<modality><n>.txt`; `query_id` có thể chứa dấu chấm nên tách
            # từ PHẢI sang, đúng một lần.
            stem, _, tag = f.stem.rpartition(".")
            mod = tag.rstrip("0123456789")
            if not stem or mod not in ("ocr", "asr"):
                continue
            text = f.read_text(encoding="utf-8").strip()
            if text:
                by_id.setdefault(stem, {}).setdefault(mod, []).append(text)
    elif exp_path.is_file():
        raw = json.loads(exp_path.read_text(encoding="utf-8"))
        items = raw.get("queries", raw) if isinstance(raw, dict) else raw
        for r in items:
            e = r.get("expansions", r)
            got: dict[str, list[str]] = {}
            for name in ("ocr", "asr"):
                val = e.get(name)
                xs = [val] if isinstance(val, str) else list(val or [])
                xs = [x for x in xs if isinstance(x, str) and x.strip()]
                if xs:
                    got[name] = xs
            if got:
                by_id[r.get("id") or r.get("query_id")] = got

    for q in queries:
        got = by_id.get(q["id"])
        if not got:
            continue
        for t in q["probes"]:
            EXP.setdefault(t, {}).update(got)
    n_probe = sum(len(q["probes"]) for q in queries)
    if by_id:
        n_o = max((len(v.get("ocr", [])) for v in EXP.values()), default=0)
        n_a = max((len(v.get("asr", [])) for v in EXP.values()), default=0)
        print(f"① mở rộng: {len(EXP)}/{n_probe} probe · {n_o} nhánh OCR + {n_a} nhánh ASR "
              f"· nguồn {exp_path}")
    else:
        print(f"① mở rộng: KHÔNG đọc được {exp_path} — mọi nhánh dùng câu gốc")

    # ②③ nạp nguồn
    t0 = time.time()
    idx = load_flat_index(Path(index), dim=dim if dim < 1024 else None)
    vis = VisualSource(idx)
    tsrc = []
    if not light:
        tsrc = [TextSource("ocr", idx.ids, load_ocr_text("data/OCR/ocr.jsonl")),
                AsrSource(idx.ids, load_frame_ms(), load_asr_segments("data/ASR"))]
    # `DEFAULT_WEIGHTS` (z-norm) KHÔNG còn dung hợp gì — ③ nay là `hierarchical_rrf`.
    # Không giữ lại một biến `W` để in và ghi báo cáo: một bản ghi nêu cấu hình mà đường
    # chạy không dùng còn tệ hơn không ghi gì, vì nó làm mọi phép so sau này sai gốc.
    fms = load_frame_ms()          # ⑥ dùng để gắn lời nói quanh khung cho đầu đọc QA
    # Ở đây từng có thêm `times = [frame_ms cho từng khung]` — một vòng lặp qua 173.426
    # khung nữa. Nó phục vụ
    # `dante(times_ms=…)`. Sau khi ⑦ hợp nhất về `k_best_alignments`, hàm đó KHÔNG nhận
    # trục thời gian nữa — nó ràng buộc thứ tự bằng chỉ số ứng viên và `min_gap`.
    # Mất trục thời gian không đổi kết quả CHỈ VÌ `DEFAULT_LAMBDA = 0`: λ là người tiêu thụ
    # duy nhất của trục ấy. Bật λ > 0 trở lại thì phải dựng lại `times` VÀ đổi sang `dante()`.
    # `FI[row]` = số khung THẬT của keyframe. ⑦ nộp đúng giá trị này, nên đây là toàn bộ
    # thông tin thời gian mà tầng nộp cần.
    #
    # Ở đây từng có khối dựng `win[row]` — nửa khe tới hàng xóm thời gian, giao biên cảnh,
    # kẹp vào cuối video. Nó chỉ tồn tại để phục vụ phép rải. Sau khi ⑦ chốt một khung mỗi
    # mốc thì `win` được DỰNG RỒI KHÔNG AI ĐỌC: 0,73 giây mỗi lượt (0,37 nạp biên cảnh +
    # 0,22 nạp độ dài video + 0,14 vòng qua 173.426 khung) cho một dict chết. Tệ hơn con
    # số đó là nó nói sai thiết kế — người đọc thấy bộ máy "cửa sổ rải" thì tưởng hệ có
    # rải. Cửa sổ đó vẫn cần cho việc CHẤM ĐIỂM, nên nó sống ở `scripts/eval/*`, đúng chỗ.
    FI = np.asarray(idx.frame_idx)
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
          f"· ③ RRF phân cấp k={RRF_K:g} · alpha {ALPHA}")

    # ②③ chấm điểm — GIỮ LẠI điểm từng RUN, không hợp rồi vứt.
    # Điểm hợp `q["S"]` vẫn cần: ④ DANTE chạy trên nó, và nó là điểm nền phá hoà trong
    # rổ. Nhưng việc CHỌN AI VÀO VÒNG TRONG thì do từng run tự đề cử.
    #
    # Mỗi modality văn bản chấm HAI lần: câu gốc và câu mở rộng. Rổ ⑤a nhận cả 5 run
    # (1 thị giác + 2 OCR + 2 ASR) làm 5 nguồn đề cử độc lập — một expansion là một GÓC
    # NHÌN khác trên cùng modality, nên nó xứng đáng có suất riêng, đúng lý lẽ đã cho
    # `+0,0153` khi mỗi nguồn được suất riêng.
    n_fallback = 0
    for q in queries:
        rows, per_probe = [], []
        for t in q["probes"]:
            v = vis.score(QV[t])
            runs: dict[str, list] = {"visual": [v]}
            flat = [v]
            for src in tsrc:
                texts = EXP.get(t, {}).get(src.name) or []
                if not texts:
                    n_fallback += 1
                group = [src.score(t)] + [src.score(x) for x in texts]
                runs[src.name] = group
                flat += group
            per_probe.append(flat)
            # alpha CHỈ trên các modality thật sự có mặt, rồi chuẩn hoá lại. `--light`
            # bỏ hết nguồn văn bản, nên nếu bê nguyên ALPHA ba khoá thì Visual nhận 1/3
            # và tổng không còn bằng 1 — không đổi thứ hạng khi chỉ có một modality,
            # nhưng nó làm bản ghi nói sai cấu hình đã chạy.
            present = {m: ALPHA[m] for m in runs}
            tot = sum(present.values())
            alpha = {m: w / tot for m, w in present.items()}
            beta = {m: _beta_for(len(r)) for m, r in runs.items()}
            rows.append(hierarchical_rrf(runs, alpha=alpha, beta=beta, k=RRF_K))
        S = np.vstack(rows).astype(np.float32)
        # KIS và Q&A nộp ĐÚNG MỘT khung mỗi dòng (thể lệ mục 1 và 2), nên ma trận điểm
        # phải có đúng một hàng trước khi tới ④ — `k_best_alignments` sinh N khung theo
        # SỐ HÀNG của nó. Gộp bằng `max`: một mốc khớp mạnh là đủ để khung vào cuộc.
        if q["kind"] in ("kis", "qa") and S.shape[0] > 1:
            S = S.max(axis=0, keepdims=True)
        q["S"] = S
        q["sources"] = per_probe
    if n_fallback:
        # Lùi về câu gốc KHÔNG trung tính: nhánh expansion khi đó TRÙNG nhánh gốc, nên
        # modality ấy tự nhân đôi quyền vote của câu gốc. Báo ra thay vì giấu.
        print(f"⚠ {n_fallback} nhánh expansion lùi về câu gốc (thiếu trong file mở rộng)")
    lap("②③ chấm 3 nguồn × 5 run")

    # ⑤a RỔ ỨNG VIÊN — top-K của điểm ĐÃ HỢP, KHÔNG đề cử riêng theo thành phần
    #
    # Trước đây mỗi RUN tự đề cử top-40 rồi hợp lại. Cách ấy sinh ra để chữa phép hợp
    # CŨ, nơi thị giác mang hệ số 1,0 còn OCR/ASR mang 0,09–0,13 nên khung thuần OCR
    # không bao giờ nổi lên. ③ nay là RRF với `alpha` chia ĐỀU và cộng theo HẠNG, nên
    # khuyết tật ấy không còn — và ba cái giá của hạn ngạch riêng thì còn nguyên:
    # rổ không còn là top-K của bất cứ thứ gì (nên `cap` cắt theo một trật tự khác với
    # trật tự đưa vào — chính là chỗ POOL_CAP=200 cắt câm 25% rổ), nguồn yếu vẫn tiêu
    # đủ 40 suất, và số ứng viên nở theo SỐ RUN chứ không theo nhu cầu.
    # Lý lẽ đầy đủ: `src/retrieval/pool.fused_pool`.
    #
    # `max` qua mọi probe, không lấy probe đầu: với TRAKE, khung chỉ hợp cho mốc thứ ba
    # vẫn phải được giữ. Đây cũng là chỗ `cap` phải áp SAU khi gộp probe — cap trong
    # vòng lặp thì TRAKE N mốc sinh rổ tới N×POOL_CAP. (OPTIMIZATION_PLAN mục 3.4)
    for q in queries:
        q["pool"] = fused_pool(q["S"], POOL_CAP)
    npool = [len(q["pool"]) for q in queries]
    print(f"⑤a rổ ứng viên: {int(np.median(npool))} khung/truy vấn "
          f"= top-{POOL_CAP} của điểm ③ đã hợp")
    lap("⑤a dựng rổ")

    # ⑤b bậc 1 — mảnh cắt vật thể, mã hoá bằng jina-clip. Gom mảnh của MỌI truy vấn rồi
    # mã hoá một lượt để đỡ vòng gọi Modal.
    # Trọng số 0 ⟹ **KHÔNG TÍNH LUÔN**, chứ không tính rồi bỏ: bậc này tốn $0,76 và ~8
    # phút cho 100 truy vấn, mà đo được nó phá điểm (xem `RERANK_WEIGHTS`).
    if rerank and RW.get("crop", 0.0) > 0.0:
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
    # chồng lên điểm đã hợp — hợp hai lượt là sai thang.
    #
    # RRF ở đây nữa, KHÔNG chuẩn hoá z. Đây là TẦNG THỨ BA của cùng một cấu trúc: ③ hợp
    # expansion trong modality rồi hợp modality; ⑤ hợp *giai đoạn* — điểm nền với điểm
    # VLM. Dùng z ở riêng tầng này thì đường chạy có hai hệ thang khác nhau, và trọng số
    # của tầng nào cũng không đọc được cùng đơn vị với tầng kia.
    #
    # Nó cũng giải đúng chỗ mà z sinh ra để giải: `fused4` là điểm RRF cỡ 0–0,016 còn
    # `vlm` là xác suất 0–1, cộng thẳng thì VLM nuốt trọn. Xếp hạng cả hai rồi mới cộng
    # làm hai thang gặp nhau mà không cần biết biên độ của bên nào.
    def combine(q: dict) -> np.ndarray:
        rows = []
        for i in range(len(q["probes"])):
            parts = {"fused4": [SourceScores("fused4", q["S0"][i],
                                             np.ones(idx.n_frames, dtype=bool))]}
            if "crop" in q:
                parts["crop"] = [q["crop"][i]]
            if "vlm" in q and i == 0:
                parts["vlm"] = [q["vlm"]]
            # Chỉ giữ trọng số của giai đoạn CÓ MẶT rồi chuẩn hoá lại, để `_report.json`
            # ghi đúng cấu hình đã chạy thay vì cấu hình đã khai báo.
            present = {k: RW[k] for k in parts if RW.get(k, 0.0) > 0.0}
            if not present:
                present = {"fused4": 1.0}
                parts = {"fused4": parts["fused4"]}
            tot = sum(present.values())
            s = hierarchical_rrf({k: parts[k] for k in present},
                                 alpha={k: v / tot for k, v in present.items()},
                                 k=RRF_K)
            # Rổ là bộ lọc CỨNG. Biên giữ nguyên thứ tự nội bộ hai phía nên DANTE vẫn
            # chạy được trên ma trận dày.
            s[q["pool"]] += POOL_MARGIN
            rows.append(s)
        return np.vstack(rows).astype(np.float32)

    for q in queries:
        q["S0"] = q["S"]                      # điểm nền của ④ nguồn, giữ nguyên bản
        q["S"] = combine(q)
    lap("⑤b mã hoá + hợp")

    # Chứng cứ CHỮ thô, nạp MỘT LẦN dùng chung cho ⑤c và ⑥. Trước đây chỉ ⑥ nạp, nên
    # ⑤c chấm khung mà không thấy chữ trên màn — bất nhất trong chính hệ.
    def _load_raw_text():
        o, a = {}, {}
        with open("data/OCR/ocr.jsonl", encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                t = (d.get("text_normalized") or d.get("text_raw") or "").strip()
                if t:
                    o[(d["video_id"], int(d["n"]))] = t
        for f in Path("data/ASR").glob("*/results/*.json"):
            d = json.loads(f.read_text(encoding="utf-8"))
            a[d["video_id"]] = [(x["start_ms"], x["end_ms"], x["text"])
                                for x in d.get("segments", [])]
        return o, a

    vlm_ocr: dict[tuple[str, int], str] = {}
    vlm_asr: dict[str, list] = {}
    if any(q["kind"] == "qa" for q in queries):
        vlm_ocr, vlm_asr = _load_raw_text()

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
    lap("⑤c VLM bậc 2")

    # ⑥ đầu đọc QA — chỉ cho đề `qa`, một lượt suy luận trên khung top-1 mỗi đề
    qa_jobs, qa_of = [], {}
    if any(q["kind"] == "qa" for q in queries):
        ocr_raw, asr_raw = (vlm_ocr, vlm_asr) if vlm_ocr else _load_raw_text()
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
    report, fmt_notes, failed = [], [], []
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
        # N mốc lấy từ SỐ HÀNG của `S`, không từ `len(probes)`: ③ đã gộp probe của
        # KIS/QA về một hàng, nên hai con số đó khác nhau đúng ở chỗ dễ nhầm nhất.
        # `k_best_alignments` sinh đúng `S.shape[0]` khung mỗi đường ⟹ số cột CSV đúng
        # theo thể lệ mà không phải kiểm lại lần nữa.
        n_mom = q["S"].shape[0]
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
            # MỘT khung mỗi mốc — `frame_idx` thật của keyframe. Không rải.
            # Vẫn khử trùng `(video, frame)`: hàm chấm lấy `R@k = max`, hai dòng giống
            # hệt mua đúng MỘT cơ hội. Sau khi bỏ rải thì trùng gần như không còn xảy
            # ra, nhưng bỏ phép kiểm là mở đường cho nó quay lại lặng lẽ.
            lines, emitted = [], set()
            for r in moments:
                vid, _ = idx.ids[r]
                k = (vid, int(FI[r]))
                if k in emitted:
                    continue
                emitted.add(k)
                lines.append([vid, int(FI[r])])
                if len(lines) >= top_k:
                    break
        # ⑦ GHI THEO THỂ LỆ. Không nối chuỗi bằng `","` nữa: `write_task_csv` dùng
        # `csv.writer` (QUOTE_MINIMAL), tự bọc ngoặc kép khi đáp án Q&A có dấu phẩy /
        # ngoặc kép / xuống dòng, rồi ĐỌC LẠI file bằng `csv.reader` để chắc bộ chấm
        # tách ra đúng số cột. Nối tay thì `"Năm người, gồm nam và nữ"` thành hai cột
        # và cả dòng lệch — thể lệ liệt kê đúng lỗi này trong "5 lỗi thường gặp nhất".
        sub = TaskSubmission(
            task_id=q["id"],
            task_type=task_type_from_filename(q["id"]),
            answers=tuple((vid, tuple(int(f) for f in fr), q.get("answer"))
                          for vid, *fr in lines),
            n_moments=n_mom,
        )
        # MỘT câu hỏng KHÔNG được giết cả gói. `validate_task` ném khi truy vấn ra 0
        # đáp án hoặc lệch số mốc — đúng ra là ném, vì đó là lỗi thật. Nhưng ngày thi
        # thì gói có 30–40 câu và chỉ 3 lượt nộp: để một exception cuốn đi 39 câu còn
        # lại là đổi sai chiều. Ghi ĐƯỢC câu nào giữ câu đó, hỏng thì kêu to và đi tiếp.
        try:
            p, notes = write_task_csv(sub, odir, budget=top_k)
        except SubmissionError as e:
            print(f"  ✗ {q['id']}: KHÔNG ghi được — {e}", file=sys.stderr)
            fmt_notes.append(f"[{q['id']}] BỎ QUA: {e}")
            failed.append(q["id"])
            continue
        for m in notes:
            print(f"  ⚠ {m}", file=sys.stderr)
        fmt_notes.extend(notes)
        report.append({"id": q["id"], "kind": q["kind"], "n_probes": len(q["probes"]),
                       "n_answers": len(lines),
                       "answer": q.get("answer"), "top1": lines[0] if lines else None})
        print(f"  {q['id']:<20} {q['kind']:<6} {len(lines):>3} đáp án → {p}")

    # ⑦b ĐÓNG GÓI. Thể lệ đòi `<zip>/submission/*.csv`; "thiếu thư mục submission" là
    # lỗi số 2 trong danh sách của BTC và nó chỉ lộ ra SAU khi đã tiêu một lượt nộp —
    # mỗi gói chỉ có 3 lượt. `pack_submission_zip` chỉ nhận `.csv`, nên `_report.json`
    # và `_rerank_scores.npz` (hàng chục MB) không lọt vào bài nộp.
    if failed:
        print(f"\n  ✗ {len(failed)}/{len(queries)} truy vấn KHÔNG có file: "
              f"{', '.join(failed)}", file=sys.stderr)
    zip_path = pack_submission_zip(
        odir, odir.parent / f"{odir.name}.zip",
        expected=[q["id"] for q in queries if q["id"] not in failed])
    print(f"  đóng gói → {zip_path} ({zip_path.stat().st_size / 1e3:.0f} KB)")
    lap("⑦ phát bài nộp")
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
            # `agree`/`prov` ĐÃ GỠ. Chúng đếm "mấy nguồn đề cử khung này", và phép
            # đề cử riêng không còn tồn tại — ⑤a nay lấy top-K của điểm đã hợp. Ghi ra
            # một hằng số rồi gọi nó là lai lịch thì tệ hơn là không ghi: phần chẩn
            # đoán đọc npz sẽ thấy một cột hợp lệ về kiểu và vô nghĩa về nội dung.
        np.savez_compressed(odir / "_rerank_scores.npz", **dump)
        print(f"  đã lưu điểm ⑤ trung gian → {odir}/_rerank_scores.npz "
              f"({(odir / '_rerank_scores.npz').stat().st_size / 1e6:.1f} MB)")

    (odir / "_report.json").write_text(
        json.dumps({"fusion": {
                        "method": "hierarchical_rrf",
                        "rrf_k": RRF_K,
                        "alpha": ALPHA,
                        "beta_rule": "ĐỀU — mọi nhánh trong một modality bằng nhau, kể cả câu gốc",
                        "runs_per_modality": {
                            "visual": 1,
                            "ocr": 1 + max((len(v.get("ocr", [])) for v in EXP.values()),
                                           default=0),
                            "asr": 1 + max((len(v.get("asr", [])) for v in EXP.values()),
                                           default=0),
                        },
                        "expansion_source": str(exp_path),
                        "probes_with_expansion": len(EXP),
                    },
                    "dim": idx.dim, "rerank": bool(rerank),
                    "pool": {"method": "fused_top_k", "cap": POOL_CAP},
                    # Định dạng nộp bài theo "Hướng dẫn nộp bài sơ tuyển".
                    "format": {"csv": "QUOTE_MINIMAL · UTF-8 · không header · LF",
                               "zip": f"{odir.name}.zip → submission/*.csv",
                               "max_rows": top_k,
                               "max_answer_chars": MAX_ANSWER_CHARS,
                               "ghi_chu": fmt_notes,
                               "khong_ghi_duoc": failed},
                    "rerank_weights": RW,
                    "vlm_top_k": vlm_top_k if rerank else 0,
                    # Ngân sách thi 2h30 là ràng buộc CỨNG, nên thời gian từng tầng phải
                    # nằm trong bản ghi chứ không chỉ trong log terminal — hết log là
                    # mất căn cứ để quyết cắt tầng nào khi thiếu giờ.
                    "timing_s": {k: round(v, 1) for k, v in T.items()},
                    "queries": report}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n✓ {odir}/ · {len(queries)} file + _report.json")
    print("  cột: video_id, frame_idx[…]  — frame_idx là số khung THẬT, không phải n")
