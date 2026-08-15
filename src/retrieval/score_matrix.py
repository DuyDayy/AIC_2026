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
# ⚠️ TẠM — chờ truy vấn gán nhãn tay để chỉnh lại. Cơ sở hiện tại:
#
#   [ĐO] quét 100 tổ hợp × 226 truy vấn tả cảnh + 1 ca sự kiện có đáp án:
#     β=γ=δ=0      R@10 0,407 · ca sự kiện hạng 67
#     β=γ=δ=0,10   R@10 0,460 · ca sự kiện hạng  1   ← đang dùng
#     β=0,10 một mình  thắng 98/thua 72, p=0,055 · ca sự kiện hạng 89
#     γ=0,10 một mình  thắng 51/thua 97, p=0,0002 → HẠI câu tả cảnh, nhưng
#                      kéo ca sự kiện 67 → 3
#
# Mâu thuẫn đó KHÔNG phải nghịch lý: 226 câu tả cảnh do model nhìn ảnh viết ra nên
# thuần thị giác, OCR không thể giúp mà chỉ thêm nhiễu. Đo trên 30 đề thi thật thì
# **40% có nhắc chữ trên khung** và 20% có tên riêng — tức bộ eval hiện tại ĐANG
# THIÊN LỆCH chống lại OCR/ASR, không phải OCR/ASR vô dụng.
#
# Nên 0,10 là mức thấp có chủ ý: đủ để cứu câu sự kiện (hạng 67 → 1), chưa đủ để
# gây hại đo được trên câu tả cảnh (thắng 99/thua 85, p=0,34 — không phân biệt được
# với ngẫu nhiên). Có nhãn tay rồi thì quét lại.
DEFAULT_WEIGHTS = {"visual": 1.0, "object": 0.10, "ocr": 0.10, "asr": 0.10}


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


NORMALIZERS = {"z": z_normalize, "rank": rank_normalize}


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
