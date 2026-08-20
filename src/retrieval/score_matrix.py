"""
Tầng ③ — MA TRẬN S: hợp bốn nguồn thành MỘT điểm cho mỗi (mốc, khung)
=======================================================================

    S[i,t] = α·z_visual + β·z_object + γ·z_ocr + δ·z_asr

`S` là toàn bộ những gì tầng ④ nhìn thấy. Với `N = 1` (KIS/QA), `max_t S[0,t]` chính là
kết quả cuối; với `N > 1` (TRAKE), DP chạy trên chính ma trận này.

=============================================================================
VÌ SAO CHUẨN HOÁ z, KHÔNG PHẢI RRF VÀ CŨNG KHÔNG PHẢI CỘNG THÔ
=============================================================================

**RRF hỏng, đo được.** `Σ 1/(k + rank_m)` cho mọi tài liệu CÓ modality `m` một khoản
dương *bất kể liên quan*. Đo trên 688 truy vấn: **−0,2927 FINAL**, mất 43% điểm, vì
nhóm video có OCR được nâng hạng có hệ thống. Đó là tiên nghiệm theo **độ phủ**, không
phải theo **độ liên quan**.

**Cộng điểm thô hỏng theo hai đường.** Thứ nhất là cùng lỗi trên: điểm BM25 ≥ 0 nên có
dữ liệu luôn tốt hơn không có. Thứ hai là **lệch thang**: đo được cosine chữ↔ảnh ~0,05
còn BM25 không có trần — cộng thẳng thì BM25 nuốt chửng thị giác.

**Chuẩn hoá z sửa cả hai**, và điều kiện là tính μ, σ **chỉ trên tập CÓ dữ liệu**:

    z_m(d) = (s_m(d) − μ_m) / σ_m     nếu d có modality m
    z_m(d) = 0                         nếu không có

    μ_m, σ_m tính trên các ứng viên CÓ m   ⟹   E[z_m | d có m] = 0

Kỳ vọng bằng đúng giá trị mà tài liệu KHÔNG có m nhận được. Nên **có dữ liệu không còn
là lợi thế**; chỉ *khớp hơn trung bình* mới là. Và vì mọi nguồn về cùng đơn vị độ lệch
chuẩn, trọng số chỉ còn diễn đạt **mức tin cậy tương đối**, không phải sửa thang.

=============================================================================
BM25 RẤT LỆCH, NÊN CÓ THÊM MỘT PHÉP CHUẨN HOÁ THỨ HAI ĐỂ SO
=============================================================================

Điểm BM25 **thưa và lệch nặng**: trong 169.409 khung có chữ, chỉ vài trăm khung khớp
một truy vấn, còn lại đúng 0. Khi đó σ bị chính vài giá trị lớn chi phối, nên khung khớp
có thể nhận z rất lớn và át hẳn thị giác dù β nhỏ.

`rank_normalize` là phương án thay thế: đổi điểm thành **thứ hạng phân vị trong tập có
dữ liệu**, rồi dịch về khoảng `[-0,5, 0,5]`. Nó giữ đúng tính chất `E = 0` nhưng **miễn
nhiễm với đuôi lệch**, đổi lại vứt bỏ độ lớn (khớp gấp đôi và khớp hơn một chút thành
như nhau).

=============================================================================
ĐÃ ĐO — VÀ `rank` THUA NẶNG. CHỐT `z`.
=============================================================================

226 truy vấn, hợp nhất đủ bốn nguồn:

    chuẩn hoá   R@10    R@100   Final
    z           0,460   0,708   0,4779   ← chốt
    rank        0,035   0,150   0,0717
                                          rank thắng 16 / thua 208 · p < 0,0001

**Cơ chế, không phải chuyện thang đo.** `rank` trải đều THEO ĐỊNH NGHĨA. Thị giác phủ
100% khung nên bị dàn đều khắp 173.426 vị trí, hai hạng liền nhau chỉ chênh
`1/173.425 ≈ 6·10⁻⁶`. Đo trên một truy vấn thật:

                    khoảng cách thị giác hạng 1↔1000   một cú khớp OCR đóng góp
    z                                          1,338                    +1,663
    rank                                       0,006                    +0,050

Với `z` hai nguồn cạnh tranh sòng phẳng (1,2×). Với `rank`, **một cú khớp OCR bằng
8,7 lần toàn bộ khoảng cách hạng 1→1000 của thị giác** — tức mọi khung có chữ khớp
nhảy qua 1.000 khung nhìn giống hơn. Sự thật "top-10 tốt hơn hẳn hạng 1000" bị xoá.

Suy ra `rank` sai với nguồn **DÀY**, nên đã thử **TRỘN**: `z` cho thị giác, `rank` cho
ba nguồn văn bản thưa, quét lại trọng số vì biên độ `rank` nhỏ hơn hẳn:

    trộn, w=0,10   R@10 0,416   Final 0,4602
    trộn, w=0,5    R@10 0,425   Final 0,4496
    trộn, w=1,0    R@10 0,416   Final 0,4230
    z hết          R@10 0,460   Final 0,4779   ← vẫn thắng

`rank_normalize` GIỮ LẠI trong mã như phương án đã bị bác bỏ có số liệu, không phải
như lựa chọn đang mở.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from src.retrieval.sources import SourceScores

# Nguồn nào có mặt thì phải có trọng số; thiếu là lỗi cấu hình, không phải mặc định 0.
#
# =============================================================================
# TRỌNG SỐ RÚT TỪ DỮ LIỆU — BOOTSTRAP 200 LẦN, KÈM KHOẢNG TIN CẬY
# =============================================================================
#
# Lấy mẫu lại 326 truy vấn của hai bộ eval 200 lần; mỗi lần cực đại `Final` trên lưới
# 0,0125 bằng hạ toạ độ; rồi TRUNG BÌNH 200 nghiệm. Trung bình các ước lượng nhiễu có
# phương sai nhỏ hơn từng cái, và nó hoàn toàn do dữ liệu quyết định.
#
#     trọng số     trung bình   trung vị        KTC 5–95%    xác định?
#     β vật thể        0,0891     0,0875   [0,049; 0,125]         có
#     γ OCR            0,1322     0,1250   [0,062; 0,163]        yếu
#     δ lời nói        0,1064     0,1250   [0,062; 0,125]         có
#
# ⚠️ **KHOẢNG TIN CẬY QUAN TRỌNG HƠN GIÁ TRỊ.** Mỗi trọng số chỉ được xác định trong
# khoảng rộng khoảng **2,5 lần**. Mọi giá trị trong khoảng đó đều tương thích với dữ
# liệu. Viết `0,0891` không có nghĩa là biết tới chữ số thứ tư — nó là điểm giữa của
# một khoảng rất rộng. Đừng chỉnh chữ số thứ ba; hãy mở rộng bộ eval.
#
# =============================================================================
# δ LỜI NÓI ĐÃ ĐỔI: 0,1064 → 0,2128. BỘ CŨ BỊ CHỈNH THIẾU, KHÔNG PHẢI KHỚP QUÁ
# =============================================================================
#
# Bộ trên fit trên **326 truy vấn GỘP hai bộ eval**. Nhưng hai bộ đó khác hẳn nhau:
#
#     226 câu do model viết KHI ĐANG NHÌN khung đích →  0% câu cần OCR/ASR
#     100 câu người gán nhãn tay mô phỏng đề thật     → 81% câu cần OCR/ASR
#
# Gộp lại thì 69% số câu là loại ASR vô dụng, nên nghiệm bị **kéo về 0**. Đó là bài toán
# HỖN HỢP, không phải khớp quá: trọng số fit ra là trung bình của hai chế độ, và tỉ lệ
# trộn trong bộ eval không khớp tỉ lệ trong đề thi.
#
# [ĐO] MRR theo hệ số nhân δ, cùng một thước trên cả hai bộ:
#
#     ×δ     Δ MRR 100 câu tay   Δ MRR 226 câu model   p hoà vốn
#     1,5          +0,0000              −0,0234           99,9%
#     2,0          +0,0215              −0,0330           60,6%
#     2,5          +0,0438              −0,0479           52,3%
#     3,0          +0,0347              −0,0648           65,1%
#
# `p` = tỉ lệ câu trong đề thi thật cần OCR/ASR. Kiểm chéo 5 lớp × 20 xáo trên bộ 100
# câu cho **Δ = +0,0149, KTC95 [+0,0108; +0,0192] ✓**, và 99/100 lớp chọn ×2,0 hoặc ×2,5
# — nên đây là tín hiệu, không phải nhiễu chọn lọc.
#
# **Chọn ×2,0 (δ = 0,2128)**: đi đúng hướng, giữ được nửa mức lợi, và chỉ cần `p > 61%`
# là có lãi — trong khi bộ gán nhãn tay đo được `p = 81%`. Không lấy ×2,5 vì nó phụ thuộc
# nặng hơn vào một `p` mà ta **không biết chắc**.
#
# ⚠️ Giá trị mới nằm NGOÀI khoảng bootstrap [0,062; 0,125] ở bảng trên. Điều đó đúng và
# không mâu thuẫn: bảng đó là khoảng tin cậy **có điều kiện trên hỗn hợp 326 câu**. Đổi
# hỗn hợp thì đổi nghiệm. Muốn thu hẹp thật sự thì phải biết `p` của đề thi.
#
# =============================================================================
# BỐN CÁCH RÚT TỪ DỮ LIỆU, TẤT CẢ RƠI VÀO CÙNG MỘT VÙNG
# =============================================================================
#
# Kiểm chéo lồng 5-fold, chấm `Final` THẬT trên phần giữ kín:
#
#     lưới tròn 0,10/0,15/0,125     0,5546  (độ lệch 0,055)
#     một lần tìm trực tiếp         0,5515  (0,053)
#     bootstrap trung vị            0,5472  (0,048)   ← phương sai thấp nhất
#     bootstrap trung bình          0,5405  (0,047)
#     chỉ thị giác                  0,4779  (0,023)
#
# Bốn cách chênh nhau **1,4 điểm phần trăm**, nhỏ hơn độ lệch giữa các fold (5 điểm).
# Chúng không phân biệt được. Điều đáng nói là chúng **đồng thuận về vùng**: bootstrap
# độc lập rơi đúng chỗ điểm lưới đã chọn tay, nên số tròn kia không phải bịa.
#
# Hai cách khớp bằng giải tích thì THUA hẳn, và cả hai vì cùng một lý do — chúng tối
# ưu sai đại lượng:
#
#     tối ưu lồi (softmax NLL)      0,7020 so với 0,7760 của lưới  → tối ưu XÁC SUẤT
#     hàm thay thế trơn của Final   0,4787 so với 0,5548, thắng 0/5 fold
#
# `Final` là bậc thang của hạng. Gradient chỉ chảy từ truy vấn NẰM SÁT ngưỡng k, số đó
# ít, và tập những câu đó đổi theo `w` — bộ tối ưu đuổi một mẫu nhỏ và trôi. Mặt mục
# tiêu lại phẳng, nên phương sai chọn lấn át mọi khoản lợi.
#
# =============================================================================
# ĐIỀU DUY NHẤT THẬT SỰ MUA ĐƯỢC ĐIỂM
# =============================================================================
#
#     chỉ thị giác    0,4779
#     có hợp nhất     0,5546      +7,7 điểm phần trăm
#
# Chênh giữa các cách CHỌN trọng số: 1,4 điểm. Chênh giữa CÓ và KHÔNG hợp nhất: 7,7.
# Nếu còn thời gian, đổ vào mở rộng bộ eval chứ đừng đổ vào ba con số này.
# Nguồn `object` ĐÃ BỎ. Nó phủ 99,5% khung nên không phải vấn đề dữ liệu thiếu — nhưng
# MRR đứng riêng là 0,0003 (hạng đáp án cỡ 3.000), và bỏ nó khỏi dung hợp đo được trên
# 210 câu (gtv2 + holdout), bắt cặp, bootstrap 4000:
#
#     ΔMRR   +0,0064   KTC95 [−0,0067, +0,0205]   thắng 39 / thua 20 / HOÀ 151
#     ΔR@100 +0,0095   KTC95 [−0,0095, +0,0286]   thắng  3 / thua  1 / HOÀ 206
#
# Cả hai khoảng tin cậy chứa 0, và 151/210 câu KHÔNG ĐỔI GÌ. Nên đây là quyết định về chi
# phí chứ không phải về điểm: nguồn này bắt nạp 1,3 GB `data/objects-full` và thêm một
# nhánh vào mọi phép đo, để đổi lấy một hiệu ứng không đo được. Ước lượng điểm còn hơi
# DƯƠNG về phía bỏ, nên không có bằng chứng nào nói nó giúp.
DEFAULT_WEIGHTS = {"visual": 1.0, "ocr": 0.1322, "asr": 0.2128}


def z_normalize(scores: np.ndarray, covered: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Chuẩn hoá z **chỉ trên tập có dữ liệu**; khung không có dữ liệu ra đúng 0.

    Tính chất bảo đảm: `E[z | covered] = 0`, đúng bằng giá trị khung không phủ nhận
    được — nên độ phủ không còn tạo lợi thế. Xem docstring module.

    σ = 0 (mọi khung có dữ liệu cùng điểm) ⟹ trả 0 hết: nguồn đó **không phân biệt
    được gì** cho truy vấn này, và 0 là cách nói điều đó mà không thêm nhiễu.
    """
    out = np.zeros_like(scores, dtype=np.float32)
    if not covered.any():
        return out
    v = scores[covered]
    sd = float(v.std())
    if sd < eps:
        return out
    out[covered] = ((v - float(v.mean())) / sd).astype(np.float32)
    return out


def rank_normalize(scores: np.ndarray, covered: np.ndarray) -> np.ndarray:
    """
    Thứ hạng phân vị trong tập có dữ liệu, dịch về `[-0,5, 0,5]`; ngoài tập ra 0.

    Giữ `E ≈ 0` như `z_normalize` nhưng **miễn nhiễm với đuôi lệch** của BM25. Giá phải
    trả: vứt độ lớn — khớp áp đảo và khớp vừa phải thành gần như nhau.

    =========================================================================
    ĐIỂM BẰNG NHAU PHẢI NHẬN CÙNG MỘT THỨ HẠNG — THỨ HẠNG TRUNG BÌNH
    =========================================================================

    Bản đầu dùng `pct[order] = arange(n)` nên mỗi khung nhận một phân vị RIÊNG kể cả
    khi điểm y hệt nhau. Với BM25 thì đó là thảm hoạ, vì khối hoà chiếm gần hết:

        [ĐO] một truy vấn OCR thật — 169.409 khung có chữ, **151.615 (89,5%) điểm
        đúng bằng 0**. Bản cũ trải khối hoà đó từ −0,500 tới +0,395, khiến **66.910
        khung KHÔNG khớp gì nhận điểm DƯƠNG**, chỉ vì vị trí hàng.

    Chỉ mục sắp theo `(video_id, n)`, nên "vị trí hàng" nghĩa là **tên video** — video
    xếp sau bảng chữ cái được nâng hạng có hệ thống. Đúng loại thiên lệch theo độ phủ
    mà module này tồn tại để chặn, lọt vào bằng một cửa khác.

    Hậu quả đo được trên 226 truy vấn: R@10 **0,035** so với 0,460 của `z`.
    """
    out = np.zeros_like(scores, dtype=np.float32)
    n = int(covered.sum())
    if n == 0:
        return out
    v = scores[covered]
    order = np.argsort(v, kind="stable")
    sv = v[order]
    # Biên các nhóm hoà: phần tử mở đầu một nhóm là chỗ giá trị đổi.
    new = np.empty(n, dtype=bool)
    new[0] = True
    if n > 1:
        new[1:] = sv[1:] != sv[:-1]
    first = np.flatnonzero(new)
    last = np.append(first[1:], n) - 1
    mean_rank = (first + last) / 2.0            # thứ hạng TRUNG BÌNH của nhóm
    pct = np.empty(n, dtype=np.float32)
    pct[order] = (mean_rank[np.cumsum(new) - 1] / max(1, n - 1)).astype(np.float32)
    out[covered] = pct - 0.5
    return out


def rrf_normalize(scores: np.ndarray, covered: np.ndarray, k: float = 60.0) -> np.ndarray:
    """
    Reciprocal Rank Fusion.
    Chỉ tính trên tập có dữ liệu (covered). Điểm = 1 / (k + hạng).
    Xếp hạng giảm dần theo điểm số ban đầu.
    """
    out = np.zeros_like(scores, dtype=np.float32)
    n = int(covered.sum())
    if n == 0:
        return out
    v = scores[covered]
    
    # Sắp xếp giảm dần theo điểm (chú ý -v)
    order = np.argsort(-v, kind="stable")
    
    # Tạo mảng hạng bắt đầu từ 1
    ranks = np.empty(n, dtype=np.float32)
    ranks[order] = np.arange(1, n + 1, dtype=np.float32)
    
    out[covered] = 1.0 / (k + ranks)
    return out


def minmax_normalize(scores: np.ndarray, covered: np.ndarray) -> np.ndarray:
    """
    Min-Max normalization trên tập có dữ liệu, đưa về [0, 1].
    Ngoài tập trả về 0.
    """
    out = np.zeros_like(scores, dtype=np.float32)
    n = int(covered.sum())
    if n == 0:
        return out
    v = scores[covered]
    v_min = float(v.min())
    v_max = float(v.max())
    if v_max - v_min < 1e-9:
        return out
    out[covered] = ((v - v_min) / (v_max - v_min)).astype(np.float32)
    return out


NORMALIZERS = {"z": z_normalize, "rank": rank_normalize,
               "rrf": rrf_normalize, "minmax": minmax_normalize}

# Phép chuẩn hoá nào ĐẢM BẢO `E[·|covered] = 0` — bất biến trung tâm của module.
#
# Chỉ `z` và `rank` có tính chất đó. `rrf` và `minmax` cho điểm DƯƠNG cho mọi khung được
# phủ, nên "có dữ liệu" tự nó thành lợi thế: khung mà OCR chấm hạng bét vẫn hơn khung OCR
# không có chữ. Đây đúng là cơ chế từng làm RRF hỏng nặng ở kiến trúc hợp điểm có trọng số.
#
# Hai cái sau VẪN nằm trong `NORMALIZERS` vì chúng có ích thật — `hierarchical_rrf` đo được
# MRR 0,5281 trên bench_kis, cao nhất trong mọi cấu hình đã thử. Lý do chúng không hỏng ở
# đó là TÍNH CHẤT CỦA KHO, không phải của phép chuẩn hoá: OCR phủ 81–100% khung mỗi video
# và ASR phủ 96,9%, nên gần như không có khung nào "không được phủ" để hưởng lợi.
#
# ⚠️ Nghĩa là: dùng `rrf`/`minmax` với một nguồn phủ THƯA thì thiên lệch quay lại ngay.
ZERO_MEAN_NORMALIZERS = ("z", "rank")


def hierarchical_rrf(
    modalities: Mapping[str, Sequence[SourceScores]],
    alpha: Mapping[str, float] | None = None,
    beta: Mapping[str, Sequence[float]] | None = None,
    k: float = 60.0,
) -> np.ndarray:
    """RRF hai tầng: gộp expansion TRONG modality trước, rồi mới gộp giữa các modality.

        Score(d) = Σ_m alpha_m · ( Σ_j beta_mj / (k + rank_mj(d)) )
        ràng buộc: Σ_m alpha_m = 1  ·  Σ_j beta_mj = 1 cho từng m

    `modalities` là `{tên: [run gốc, run mở rộng, …]}`. Run đầu tiên của mỗi modality nên
    là câu GỐC — thứ tự đó là thứ `beta` mặc định giả định khi ai đó truyền beta lệch.

    =========================================================================
    VÌ SAO PHÂN CẤP, KHÔNG PHẲNG
    =========================================================================

    Ném cả năm run vào MỘT phép RRF phẳng thì modality nào có nhiều expansion hơn tự
    nhiên có nhiều quyền vote hơn — Visual một run bị OCR hai run lấn át chỉ vì đếm run.
    Chuẩn hoá beta trong từng modality chặn đúng điều đó.

    [ĐO] bench_kis 100 câu, k=60, so cùng một bộ run:

        cấu hình                                                MRR     R@100
        V + O_gốc + A_gốc                (không expansion)    0,4751     0,91
        V + O_qo + A_qa                  (bỏ bản gốc)         0,4867     0,93
        V + (O_gốc+O_qo)/2 + A_gốc       (QE chỉ OCR)         0,4989     0,92
        V + O_gốc + (A_gốc+A_qa)/2       (QE chỉ ASR)         0,4874     0,92
        V + (O_gốc+O_qo)/2 + (A_gốc+A_qa)/2   ← MẶC ĐỊNH     0,5281     0,93
        V + O_gốc + O_qo + A_gốc + A_qa  (PHẲNG)              0,4427     0,89

    Ba điều rút ra, và cả ba đều nằm trong mặc định của hàm này:
      · phân cấp hơn phẳng 0,085 MRR — khoản lớn nhất trong bảng;
      · giữ bản GỐC bên cạnh expansion hơn bỏ nó 0,041 — expansion bổ sung, không thay thế;
      · QE cho CẢ HAI nhánh văn bản mới đạt đỉnh, cao hơn tổng hai phần riêng lẻ.

    =========================================================================
    VÌ SAO `alpha` MẶC ĐỊNH LÀ ĐỀU
    =========================================================================

    Không phải vì chưa thử. E4 quét trọng số toàn cục hai tầng trên 210 câu tuning
    (gtv2 + holdout), rồi báo trên 100 câu held-out chưa đụng tới:

        alpha* tìm được = (0,30 · 0,35 · 0,35)   ≈ đều
        beta*  tìm được = (0,5 · 0,5) cho CẢ OCR lẫn ASR  ≈ đúng mặc định ở đây
        held-out: E3 đều MRR 0,5281  →  E4 có trọng số 0,4884
        ΔMRR bắt cặp −0,0398 · KTC95 [−0,0783, −0,0056] · thắng 10 / thua 18

    Khoảng tin cậy nằm TRỌN dưới 0: trọng số toàn cục **kém hơn có ý nghĩa**, không phải
    "không phân biệt được". Nên `alpha` đều là kết quả đo, không phải chỗ chưa tối ưu.

    ⚠️ Điều đó KHÔNG nói trọng số là vô ích, chỉ nói trọng số TOÀN CỤC là vô ích. Cùng
    bộ số ấy cho thấy alpha* hại nhóm `vision` (−0,0696 MRR) trong khi giúp
    `vision+ocr+asr` (+0,0266) — một hằng số không phục vụ được mọi loại đề.

    Raises:
        ValueError: modality rỗng · lệch số khung · beta lệch số run · trọng số âm.
    """
    if not modalities:
        raise ValueError("hierarchical_rrf cần ít nhất một modality")
    n = None
    for name, runs in modalities.items():
        if not runs:
            raise ValueError(f"modality {name!r} không có run nào")
        for r in runs:
            if n is None:
                n = r.scores.shape[0]
            elif r.scores.shape[0] != n:
                raise ValueError("các run lệch số khung — nguy cơ ghép nhầm vị trí")

    names = list(modalities)
    a = {m: 1.0 / len(names) for m in names} if alpha is None else dict(alpha)
    missing = [m for m in names if m not in a]
    if missing:
        # Thiếu trọng số phải là LỖI chứ không mặc định 0: thêm modality mới rồi quên nối
        # là im lặng mất nguồn, đúng lỗi mà `fuse` cũng chặn.
        raise ValueError(f"thiếu alpha cho {missing}")
    if any(a[m] < 0 for m in names):
        raise ValueError("alpha không được âm")

    out = np.zeros(n, dtype=np.float32)
    for m in names:
        runs = modalities[m]
        b = list(beta[m]) if beta and m in beta else [1.0 / len(runs)] * len(runs)
        if len(b) != len(runs):
            raise ValueError(f"beta[{m!r}] có {len(b)} phần tử cho {len(runs)} run")
        if any(x < 0 for x in b):
            raise ValueError(f"beta[{m!r}] không được âm")
        total = float(sum(b))
        if total <= 0:
            raise ValueError(f"beta[{m!r}] cộng lại bằng 0")
        inner = np.zeros(n, dtype=np.float32)
        for w, r in zip(b, runs):
            # Chuẩn hoá beta TẠI ĐÂY, không tin bên gọi đã chuẩn hoá: bất biến
            # "Σ beta = 1 trong từng modality" chính là thứ giữ cho modality nhiều
            # expansion không tự có thêm quyền vote.
            if w:
                inner += (w / total) * rrf_normalize(r.scores, r.covered, k)
        out += float(a[m]) * inner
    return out


def fuse(sources: Sequence[SourceScores], weights: Mapping[str, float],
         mode: str = "z") -> np.ndarray:
    """
    `[SourceScores]` + trọng số → một vector điểm cho mọi khung.

    Trọng số **chỉ diễn đạt mức tin cậy tương đối**, không phải sửa thang — phép chuẩn
    hoá đã đưa mọi nguồn về cùng đơn vị. Nên chỉ TỈ LỆ giữa các trọng số có nghĩa, và
    cố định `α = 1` là đủ khi quét.

    Raises:
        ValueError: `mode` lạ · nguồn thiếu trọng số · các nguồn lệch số khung.
    """
    if mode not in NORMALIZERS:
        raise ValueError(f"mode phải thuộc {sorted(NORMALIZERS)}, nhận {mode!r}")
    if not sources:
        return np.zeros(0, dtype=np.float32)
    n = sources[0].scores.shape[0]
    if any(s.scores.shape[0] != n for s in sources):
        raise ValueError("các nguồn lệch số khung — nguy cơ ghép nhầm vị trí")
    missing = [s.name for s in sources if s.name not in weights]
    if missing:
        raise ValueError(
            f"thiếu trọng số cho {missing} — bỏ sót một nguồn phải là LỖI, không phải "
            f"mặc định 0, nếu không thì thêm nguồn mới xong quên nối là im lặng"
        )
    f = NORMALIZERS[mode]
    out = np.zeros(n, dtype=np.float32)
    for s in sources:
        w = float(weights[s.name])
        if w:
            out += w * f(s.scores, s.covered)
    return out


def score_matrix(per_probe: Sequence[Sequence[SourceScores]],
                 weights: Mapping[str, float], mode: str = "z") -> np.ndarray:
    """
    `S[i,t]` — hàng `i` là mốc thứ `i`, cột `t` là khung. Đầu vào của tầng ④.

    `N = 1` ⟹ ma trận một hàng, và `max_t S[0,t]` chính là kết quả KIS. Không cần
    nhánh riêng cho KIS: DP ở ④ tự suy biến.
    """
    if not per_probe:
        return np.zeros((0, 0), dtype=np.float32)
    rows = [fuse(p, weights, mode) for p in per_probe]
    return np.vstack(rows).astype(np.float32)
