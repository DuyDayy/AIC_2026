"""
Phủ lưới khung hình — cài đặt Định lý 1 và bài toán đặt lưới tối ưu (D5)
=======================================================================

ĐỊNH LÝ 1 (phủ lưới — pigeonhole).
    Cho A = {a + kΔ : k ∈ ℤ} và đoạn đáp án [s, e] với L = e − s + 1:

        A ∩ [s, e] ≠ ∅ với MỌI vị trí s   ⟺   Δ ≤ L

    Chứng minh. (⇐) L ≥ Δ số nguyên liên tiếp chứa đủ một hệ thặng dư đầy đủ
    mod Δ, nên chứa phần tử ≡ a (mod Δ).
    (⇒) Nếu Δ > L, chọn s ≡ a + 1 (mod Δ) thì [s, s+L−1] ⊂ (a+kΔ, a+(k+1)Δ),
    giao rỗng. ∎

Hệ quả vận hành: `m` khung hình nộp trên lưới bước Δ ≤ L phủ CHẮC CHẮN một
dải rộng `m·Δ` khung hình. Đây là mức sàn không cần mô hình nào — dùng khi
mọi toán tử sắc đều không áp dụng được.

MÔ HÌNH CỬA SỔ ĐỐI XỨNG. Các hàm dựa trên posterior giả định cửa sổ đáp án
[s, e] nằm ĐỐI XỨNG quanh khoảnh khắc ngữ nghĩa `t`, tức
    [s, e] = [t − ⌊(L−1)/2⌋, t + ⌊L/2⌋].
Khi đó "nộp khung hình f trúng khoảnh khắc t" ⟺ t nằm trong đoạn dài L đối
xứng quanh f. Giả định này chỉ ảnh hưởng phần TỐI ƯU HOÁ vị trí lưới; phần
bảo đảm của Định lý 1 (`uniform_grid`) không phụ thuộc nó.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# ĐỊNH LÝ 1 — LƯỚI ĐỀU CÓ BẢO ĐẢM
# ============================================================


def half_widths(window_length: int) -> tuple[int, int]:
    """
    Nửa bề rộng trái/phải của cửa sổ dài `L` đối xứng: (⌊(L−1)/2⌋, ⌊L/2⌋).

    Tổng luôn bằng L − 1, nên đoạn [f − lo, f + hi] có đúng L phần tử.
    """
    if window_length < 1:
        raise ValueError(f"window_length phải ≥ 1, nhận {window_length}")
    return (window_length - 1) // 2, window_length // 2


def uniform_grid(
    center: int,
    m: int,
    delta: int,
    *,
    lo: int = 0,
    hi: int | None = None,
) -> list[int]:
    """
    Lưới đều `m` điểm bước `delta`, căn giữa quanh `center`, kẹp vào [lo, hi].

    Đây là chiến lược sàn của Định lý 1: nếu `delta ≤ L` thì lưới phủ chắc
    chắn dải rộng `m·delta` khung hình quanh `center`.

    Args:
        center: khung hình trung tâm (ước lượng tốt nhất).
        m: số khung hình được phép nộp cho mốc này.
        delta: bước lưới. Phải ≤ L để có bảo đảm.
        lo, hi: biên khung hình hợp lệ của video (hi = None ⟹ không chặn trên).

    Returns:
        Danh sách `m` khung hình tăng dần, đã khử trùng lặp sau khi kẹp biên.
        Có thể ngắn hơn `m` nếu video quá ngắn.
    """
    if m < 1:
        raise ValueError(f"m phải ≥ 1, nhận {m}")
    if delta < 1:
        raise ValueError(f"delta phải ≥ 1, nhận {delta}")

    # Căn giữa: với m lẻ, `center` nằm chính giữa; với m chẵn, lệch trái nửa bước.
    start = center - ((m - 1) // 2) * delta
    frames = [start + i * delta for i in range(m)]

    clamped = [max(lo, f) if hi is None else min(hi, max(lo, f)) for f in frames]
    # Kẹp biên có thể tạo trùng lặp; giữ thứ tự tăng dần và khử trùng.
    return sorted(set(clamped))


def spread_in_window(center: int, lo: int, hi: int, m: int) -> list[int]:
    """
    `m` khung rải đều trong cửa sổ `[lo, hi]` — lưới THÍCH ỨNG, không bước cố định.

    Khác `uniform_grid` ở chỗ bước không cố định: khoảng cách giữa hai keyframe liền
    nhau đo được **p10 = 19, p50 = 48, p90 = 105 khung**, chênh hơn 5 lần. Bước cố định
    hoặc phủ thừa ở chỗ dày, hoặc hụt ở chỗ thưa.

    =========================================================================
    VÌ SAO CẦN — ĐÁP ÁN KHÔNG PHẢI KEYFRAME BTC CẤP
    =========================================================================

    Thể lệ nói rõ: *"khung hình ngữ nghĩa … KHÁC VỚI I-Frame là khung hình kỹ thuật
    trong các thuật toán nén video đã được cung cấp cho các đội thi"*, và đoạn đáp án
    *"thường rất ngắn, thông thường là dưới 10 frame"* (ví dụ KIS trong thể lệ:
    `[500, 510]` = 11 khung).

    [ĐO] xác suất một cửa sổ rộng `L` đặt ngẫu nhiên **chứa sẵn** một keyframe của ta:

        L=9  → 23,5%      L=25 → 57,6%      L=101 → 96,6%

    Nên ở `L ≈ 10`, nộp thuần keyframe có **trần cứng ~23%** bất kể truy xuất tốt đến
    đâu. Rải khung vào khe là cách duy nhất vượt trần đó mà không phải mã hoá lại video.

    [ĐO] Final trung bình trên 226 truy vấn, mốc thật lệch ngẫu nhiên trong khe, ngân
    sách 100 câu trả lời:

        cách nộp        L=9      L=11     L=15     L=21
        100 mốc × 1     0,0608   0,0758   0,1002   0,1384
        20 mốc × 5      0,1624   0,1877   0,2137   0,2349   ← chọn
        14 mốc × 7      0,1742   0,1882   0,2038   0,2184
        1 mốc × 100     0,0805   0,0827   0,0876   0,0962

    `20 × 5` tốt nhất trung bình trên cả dải; `14 × 7` nhỉnh hơn ở `L` hẹp nhất. Cả hai
    hơn cách nộp thuần keyframe **~2,5 lần**.

    =========================================================================
    CỬA SỔ DO CHỖ GỌI QUYẾT ĐỊNH, KHÔNG SUY RA Ở ĐÂY
    =========================================================================

    Bản đầu tự suy cửa sổ từ hai keyframe hàng xóm ("nửa khe"). Cách đó sinh hai lỗi
    mà `writer.validate_all` bắt được trên bài nộp thật:

    * **Vượt số khung video.** Keyframe cuối cùng không có hàng xóm phải, nên phải
      đoán bằng hằng số và dải rải chạy ra ngoài: *"frame_id 16994 ≥ số khung 16993"*.
    * **Vắt qua ranh giới CẢNH.** Nửa khe không biết gì về cảnh, nên nó rải sang nội
      dung hoàn toàn khác.

    Chỗ gọi có đủ dữ liệu để tính đúng — biên cảnh ở `sources.load_shot_bounds`, chặn
    trên video ở `load_video_last_frame` — nên cửa sổ vào đây là tham số, không phải
    suy đoán.

    Args:
        center: khung của keyframe được chọn, phải nằm trong `[lo, hi]`.
        lo, hi: biên cửa sổ, đã giao với biên cảnh và chặn trên của video.
        m: số khung được phép nộp cho mốc này.

    =========================================================================
    THỨ TỰ TRẢ VỀ: KEYFRAME TRƯỚC, RỒI TOẢ RA HAI BÊN
    =========================================================================

    Tập khung không đổi, nhưng **thứ tự thì tính điểm**: `Final` cộng `R@k` ở
    `k ∈ {1,5,20,50,100}`, nên câu trả lời đầu tiên được tính vào cả năm mức còn câu
    thứ 51 chỉ một mức. Đưa khung có xác suất cao nhất — chính keyframe — lên đầu.

    Bản đầu trả về **tăng dần theo số khung**, khiến keyframe rơi vào GIỮA (hạng 4/7)
    còn hạng 1 là khung ở rìa khe. [ĐO] so cặp trên 100 truy vấn, bài nộp THẬT:

        mô hình cửa sổ    tăng dần   keyframe trước   tốt/tệ         p
        chặt ±4             0,4620           0,5400    35/0    <10⁻⁶
        cửa sổ khe          0,6040           0,5920     2/6     0,29
        trung bình          0,5330           0,5660

    Ở mô hình chặt — nơi vị trí thật sự quan trọng — nó tốt hơn **35 câu, tệ hơn 0**.
    Ở mô hình khe, mọi khung trong khe đều trúng nên thứ tự gần như không đổi gì; chênh
    −1,2pp đến từ **mốc cuối bị cắt** khi hết ngân sách 100, và p = 0,29 nói đó là nhiễu.

    Returns:
        Tối đa `m` khung, đã khử trùng, **sắp theo khoảng cách tới `center`** (gần
        trước). `m = 1` trả đúng `[center]` — không đoán mò khi ngân sách chỉ đủ một
        khung.

    Raises:
        ValueError: `m < 1` · `center` nằm ngoài `[lo, hi]`.
    """
    if m < 1:
        raise ValueError(f"m phải ≥ 1, nhận {m}")
    if not lo <= center <= hi:
        raise ValueError(f"center={center} phải nằm trong cửa sổ [{lo}, {hi}]")
    if m == 1:
        return [center]
    if hi <= lo:
        return [center]
    step = (hi - lo) / (m - 1)
    frames = {int(round(lo + i * step)) for i in range(m)}
    # MỘT thứ tự duy nhất: gần `center` trước, `f` phá hoà cho tất định. Không có cờ
    # đổi thứ tự — thứ tự khác đã đo được là tệ hơn (35/0), và một nhánh không ai
    # dùng là một nhánh không ai kiểm.
    return sorted(frames, key=lambda f: (abs(f - center), f))


def guaranteed_span(m: int, delta: int) -> int:
    """Bề rộng khung hình được phủ CHẮC CHẮN bởi `m` điểm lưới bước `delta`."""
    return m * delta


def grid_hits(frames: Sequence[int], s: int, e: int) -> bool:
    """Có khung hình nào trong `frames` rơi vào đoạn đóng [s, e] không?"""
    return any(s <= f <= e for f in frames)


# ============================================================
# D5 — ĐẶT LƯỚI TỐI ƯU THEO POSTERIOR
# ============================================================


def optimal_placement(
    posterior: Sequence[float],
    m: int,
    window_length: int,
    *,
    frame_lo: int = 0,
) -> list[int]:
    """
    Chọn `m` khung hình để nộp sao cho TỔNG XÁC SUẤT TRÚNG là lớn nhất.

    Bài toán: cho posterior `p(t)` trên khoảnh khắc ngữ nghĩa `t`, mỗi khung
    hình nộp `f` "phủ" đoạn dài `L` đối xứng quanh `f`. Chọn `m` khung hình để
    cực đại khối lượng posterior của HỢP các đoạn phủ.

    Thuật toán: quy hoạch động trên các vị trí RỜI NHAU, cho nghiệm CHÍNH XÁC.

    Vì sao hạn chế về rời nhau là không mất tổng quát: hợp của `m` đoạn dài `L`
    là hợp rời của `t ≤ m` "dải" cực đại; dải thứ i dài ℓᵢ được phủ bởi cᵢ đoạn
    với ℓᵢ ≤ cᵢ·L và Σcᵢ = m. Lát mỗi dải bằng ⌈ℓᵢ/L⌉ ≤ cᵢ đoạn rời nhau thì
    phủ được trọn dải đó, tổng cộng dùng ≤ m đoạn rời nhau phủ một tập ⊇ hợp
    ban đầu. Vậy nghiệm tối ưu trong lớp rời nhau ≥ nghiệm tối ưu tổng quát. ∎

    Độ phức tạp: O(m · P) với P = số vị trí ứng viên.

    Args:
        posterior: p(t) trên các khung hình `frame_lo .. frame_lo + len(p) − 1`.
            Không cần chuẩn hoá; chỉ cần không âm.
        m: số khung hình được nộp.
        window_length: `L` giả định của cửa sổ đáp án. Nhỏ hơn = bi quan hơn.
        frame_lo: khung hình ứng với `posterior[0]`.

    Returns:
        Danh sách khung hình tăng dần, độ dài ≤ m (ngắn hơn nếu posterior hẹp).
    """
    if m < 1:
        raise ValueError(f"m phải ≥ 1, nhận {m}")
    if not posterior:
        return []
    if any(w < 0 for w in posterior):
        raise ValueError("posterior không được âm")

    lo_w, hi_w = half_widths(window_length)
    n = len(posterior)

    prefix = np.concatenate(([0.0], np.cumsum(np.asarray(posterior, dtype=np.float64))))

    # Vị trí ứng viên: mở rộng mỗi bên nửa cửa sổ, vì một khung hình ngay ngoài
    # miền posterior vẫn có thể phủ phần khối lượng ở rìa.
    cand_lo, cand_hi = -hi_w, n - 1 + lo_w
    n_cand = cand_hi - cand_lo + 1

    # Khối lượng phủ của từng vị trí ứng viên, tính một lần bằng tiền tố tổng.
    idx = np.arange(cand_lo, cand_hi + 1)
    left = np.clip(idx - lo_w, 0, n)
    right = np.clip(idx + hi_w + 1, 0, n)
    covered = prefix[right] - prefix[left]

    # dp[j][i] = khối lượng tốt nhất dùng ≤ j khung hình trong i ứng viên đầu.
    #   dp[j][i] = max(dp[j][i−1],  covered[i−1] + dp[j−1][max(0, i−L)])
    # Vế trái là MAX LUỸ TIẾN theo i, nên cả hàng j tính được bằng
    # `np.maximum.accumulate` thay vì vòng lặp Python.
    L = window_length
    prev_row = np.zeros(n_cand + 1, dtype=np.float64)
    rows: list[np.ndarray] = [prev_row]
    takes: list[np.ndarray] = [np.zeros(n_cand + 1, dtype=bool)]

    back = np.maximum(np.arange(n_cand + 1) - L, 0)
    for _ in range(1, m + 1):
        use = np.empty(n_cand + 1, dtype=np.float64)
        use[0] = -np.inf
        use[1:] = covered + prev_row[back[1:]]
        row = np.maximum.accumulate(use)
        np.maximum(row, 0.0, out=row)
        take = use >= row  # dùng ứng viên i khi nó chính là cực đại luỹ tiến
        take[0] = False
        rows.append(row)
        takes.append(take)
        prev_row = row

    # Truy vết.
    chosen: list[int] = []
    j, i = m, n_cand
    while j > 0 and i > 0:
        if takes[j][i]:
            chosen.append(cand_lo + i - 1 + frame_lo)
            i = max(0, i - L)
            j -= 1
        else:
            i -= 1

    chosen.sort()
    if len(chosen) < m:
        logger.debug(
            "optimal_placement: chỉ đặt được %d/%d khung hình (posterior hẹp hơn m·L).",
            len(chosen),
            m,
        )
    return chosen


def hit_probability(
    frames: Sequence[int],
    posterior: Sequence[float],
    window_length: int,
    *,
    frame_lo: int = 0,
) -> float:
    """
    Xác suất trúng khi nộp tập `frames`, dưới mô hình cửa sổ đối xứng.

    Dùng để chấm điểm một phương án đặt lưới, và để tính lợi ích biên `h(m)`
    trong bài toán phân bổ slot (D4).
    """
    if not posterior:
        return 0.0
    weights = np.asarray(posterior, dtype=np.float64)
    total = float(weights.sum())
    if total <= 0:
        return 0.0

    lo_w, hi_w = half_widths(window_length)
    n = len(weights)
    covered = np.zeros(n, dtype=bool)
    for f in frames:
        idx = f - frame_lo
        left = max(0, idx - lo_w)
        right = min(n - 1, idx + hi_w)
        if left <= right:
            covered[left : right + 1] = True

    return float(weights[covered].sum()) / total


# =============================================================================
# RẢI THÍCH ỨNG — BƯỚC CỐ ĐỊNH, KHÔNG PHẢI SỐ KHUNG CỐ ĐỊNH
# =============================================================================
#
# `spread_in_window(…, m)` rải `m` khung dù cửa sổ rộng hay hẹp. Nhưng cửa sổ đo được
# **p10 = 28 · p50 = 43 · p90 = 88 khung** — chênh hơn 3 lần. Bước cố định `m = 7` vì thế
# hoặc phủ thừa ở cửa sổ hẹp (phí ô ngân sách), hoặc hụt ở cửa sổ rộng.
#
# Hàm này lật ngược tham số: cố định **BƯỚC** rồi suy ra `m`.
#
# `step = 10` KHÔNG phải số fit ra — thể lệ nói cửa sổ đáp án rộng ~9–11 khung, nên bước
# 10 đặt hai khung rải liền nhau cách nhau **đúng một bề rộng cửa sổ**: phủ tối đa mà
# không chồng lấn phí.
#
# [ĐO] Final tại L=11, bootstrap bắt cặp, so với `m = 7` cố định:
#
#     bước    bộ GIỮ KÍN (n=110)              bộ tune (n=100)
#      8     +0,0146  [+0,0057,+0,0254] ✓     +0,0214  [+0,0074,+0,0360] ✓
#     10     +0,0280  [+0,0169,+0,0406] ✓     +0,0397  [+0,0260,+0,0546] ✓   ← chốt
#     12     +0,0107  [−0,0039,+0,0259] —     +0,0243  [+0,0078,+0,0410] ✓
#
# Đỉnh ở 10 trên **cả hai** bộ, và trùng bề rộng cửa sổ thể lệ nêu — cơ chế, không phải
# trùng hợp. `m` sinh ra: p10=4 · p50=5 · p90=10, trung bình 6,3 ⟹ ~16 mốc/bài nộp, xấp
# xỉ 14 mốc của bản cố định. Nó KHÔNG mua điểm bằng "nhiều mốc hơn" mà bằng **phân bổ
# đúng chỗ hơn**.

DEFAULT_SPREAD_STEP = 10
SPREAD_M_MIN = 3
SPREAD_M_MAX = 20


def adaptive_m(lo: int, hi: int, step: int = DEFAULT_SPREAD_STEP) -> int:
    """Số khung nên rải trong cửa sổ `[lo, hi]` để bước giữa chúng ≈ `step`.

    Args:
        lo: mép trái cửa sổ rải.
        hi: mép phải cửa sổ rải.
        step: bước mong muốn giữa hai khung liền nhau, tính bằng khung.

    Returns:
        `m` đã kẹp vào `[SPREAD_M_MIN, SPREAD_M_MAX]` — dưới 3 thì cửa sổ rộng bị bỏ
        trống, trên 20 thì một mốc ăn hết ngân sách 100 dòng.

    Raises:
        ValueError: `step < 1` · `hi < lo`.

    Examples:
        >>> adaptive_m(0, 43)          # cửa sổ trung vị
        5
        >>> adaptive_m(0, 8)           # cửa sổ hẹp — rải ít, khỏi phí ô
        3
        >>> adaptive_m(0, 120)         # cửa sổ rộng — rải dày hơn
        13
    """
    if step < 1:
        raise ValueError(f"step phải ≥ 1, nhận {step}")
    if hi < lo:
        raise ValueError(f"cửa sổ rỗng: [{lo}, {hi}]")
    width = hi - lo + 1
    return int(min(SPREAD_M_MAX, max(SPREAD_M_MIN, round(width / step) + 1)))
