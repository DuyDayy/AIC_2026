"""
V2 — Kiểm chứng Định lý 1 (phủ lưới) và bộ đặt lưới tối ưu.
===========================================================

Test cốt lõi là `test_theorem_1_exhaustive`: quét vét cạn mọi tổ hợp
(Δ, L, pha lệch) và khẳng định `hit ⟺ Δ ≤ L`. Nếu test này đỏ thì toàn bộ
lập luận bảo đảm độ phủ trong kế hoạch sụp đổ.

NGUỒN CỦA CÁC GIÁ TRỊ KỲ VỌNG:
  [ĐL1]     Định lý 1 (phủ lưới), phát biểu + chứng minh ở
            BAO_CAO_THIET_KE.md §2, dựa trên NGUYÊN LÝ CHUỒNG BỒ CÂU: trong
            `L ≥ Δ` số nguyên liên tiếp luôn có đủ một hệ thặng dư đầy đủ
            mod `Δ`. Đây là điều kiện CẦN VÀ ĐỦ, không phải kinh nghiệm —
            nên test kiểm được CẢ HAI CHIỀU (có phản ví dụ khi `Δ > L`).
  [PDF]     "Thong tin vong So tuyen AIC2026.pdf" mục 2.1.3 — `L < 10` cho
            TRAKE; ví dụ của BTC ở mục 2.1.1 là `[500,510]` (11 frame).
  [VÉT-CẠN] `test_theorem_1_exhaustive` và `test_matches_bruteforce_on_small_instances`
            tự thân là bằng chứng: quét toàn bộ không gian nhỏ / so DP với
            vét cạn tổ hợp, không dựa vào giá trị kỳ vọng viết tay.
  [WLOG]    `optimal_placement` hạn chế nghiệm về các đoạn RỜI NHAU. Chứng
            minh không mất tổng quát nằm trong docstring của hàm đó (lát mỗi
            "dải" cực đại bằng ⌈ℓ/L⌉ ≤ c đoạn rời nhau).

Chạy: python tests/test_coverage.py
"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.submission.coverage import (
    spread_in_window,
    grid_hits,
    guaranteed_span,
    half_widths,
    hit_probability,
    optimal_placement,
    uniform_grid,
)


class TestTheorem1(unittest.TestCase):
    """A ∩ [s, e] ≠ ∅ với mọi s  ⟺  Δ ≤ L."""

    def test_theorem_1_exhaustive(self):
        """
        Quét VÉT CẠN: với mỗi (Δ, L), thử mọi pha lệch của lưới và mọi vị trí
        cửa sổ trong một chu kỳ. Khẳng định hai chiều của Định lý 1.
        """
        span = 400
        for delta in range(1, 16):
            for L in range(1, 16):
                # Lưới vô hạn bước `delta`, pha `a`: {a + k·delta}.
                always_hits = True
                for a in range(delta):
                    grid = [a + k * delta for k in range(-2, span // delta + 3)]
                    for s in range(0, span):
                        if not grid_hits(grid, s, s + L - 1):
                            always_hits = False
                            break
                    if not always_hits:
                        break
                self.assertEqual(
                    always_hits,
                    delta <= L,
                    f"Định lý 1 sai tại Δ={delta}, L={L}: "
                    f"always_hits={always_hits} nhưng Δ≤L là {delta <= L}",
                )

    def test_counterexample_when_delta_exceeds_L(self):
        """Δ > L: tồn tại vị trí cửa sổ lọt hoàn toàn vào khe lưới."""
        delta, L = 10, 9
        grid = [k * delta for k in range(20)]
        # Cửa sổ [1, 9] nằm trọn giữa hai điểm lưới 0 và 10.
        self.assertFalse(grid_hits(grid, 1, L))

    def test_boundary_delta_equals_L_always_hits(self):
        """
        Δ = L là trường hợp biên chặt: vẫn luôn trúng.

        Lưu ý: Định lý 1 phát biểu cho lưới VÔ HẠN. Lưới phải trải rộng hơn
        miền quét của `s`, nếu không phần trượt chỉ là do cắt cụt lưới chứ
        không phải phản ví dụ của định lý.
        """
        span = 200
        for L in range(1, 20):
            grid = [k * L for k in range(span // L + 2)]
            for s in range(0, span):
                self.assertTrue(grid_hits(grid, s, s + L - 1), f"trượt tại L={L}, s={s}")


class TestUniformGrid(unittest.TestCase):
    def test_centered_on_center(self):
        grid = uniform_grid(center=1000, m=5, delta=10)
        self.assertEqual(grid, [980, 990, 1000, 1010, 1020])
        self.assertIn(1000, grid)

    def test_guaranteed_span_formula(self):
        self.assertEqual(guaranteed_span(20, 9), 180)
        self.assertEqual(guaranteed_span(100, 5), 500)

    def test_grid_covers_its_guaranteed_span(self):
        """
        Với Δ ≤ L, mọi cửa sổ có tâm nằm trong dải bảo đảm đều bị lưới trúng.
        Kiểm bằng cách quét mọi vị trí cửa sổ bên trong dải.
        """
        m, delta, L, center = 11, 5, 5, 1000
        grid = uniform_grid(center, m, delta)
        lo, hi = grid[0], grid[-1]
        for s in range(lo, hi - L + 2):
            self.assertTrue(grid_hits(grid, s, s + L - 1), f"trượt cửa sổ bắt đầu ở {s}")

    def test_clamped_to_video_bounds(self):
        grid = uniform_grid(center=5, m=5, delta=10, lo=0, hi=100)
        self.assertTrue(all(0 <= f <= 100 for f in grid))
        self.assertEqual(grid, sorted(set(grid)), "phải tăng dần và không trùng")

    def test_rejects_bad_params(self):
        with self.assertRaises(ValueError):
            uniform_grid(100, m=0, delta=5)
        with self.assertRaises(ValueError):
            uniform_grid(100, m=5, delta=0)


class TestHalfWidths(unittest.TestCase):
    def test_sums_to_L_minus_1(self):
        for L in range(1, 30):
            lo, hi = half_widths(L)
            self.assertEqual(lo + hi, L - 1, f"L={L}")

    def test_odd_length_is_symmetric(self):
        self.assertEqual(half_widths(11), (5, 5))

    def test_even_length_leans_right(self):
        self.assertEqual(half_widths(10), (4, 5))


class TestOptimalPlacement(unittest.TestCase):
    def test_single_frame_picks_densest_window(self):
        """m=1: phải chọn khung hình phủ khối lượng posterior lớn nhất."""
        posterior = [0.0] * 100
        for t in range(48, 53):  # khối tập trung quanh 50
            posterior[t] = 1.0
        placed = optimal_placement(posterior, m=1, window_length=5)
        self.assertEqual(len(placed), 1)
        self.assertAlmostEqual(hit_probability(placed, posterior, 5), 1.0)

    def test_beats_uniform_grid_on_multimodal_posterior(self):
        """
        Posterior hai đỉnh xa nhau: đặt tối ưu phải phủ cả hai đỉnh, còn lưới
        đều căn giữa sẽ rơi vào vùng trũng ở giữa.
        """
        posterior = [0.0] * 400
        for t in range(20, 25):
            posterior[t] = 1.0
        for t in range(370, 375):
            posterior[t] = 1.0

        optimal = optimal_placement(posterior, m=2, window_length=5)
        p_opt = hit_probability(optimal, posterior, 5)

        naive = uniform_grid(center=200, m=2, delta=5, lo=0, hi=399)
        p_naive = hit_probability(naive, posterior, 5)

        self.assertAlmostEqual(p_opt, 1.0)
        self.assertAlmostEqual(p_naive, 0.0)
        self.assertGreater(p_opt, p_naive)

    def test_monotone_in_m(self):
        """Thêm slot không bao giờ làm giảm xác suất trúng (h lõm, tăng)."""
        rng = random.Random(7)
        posterior = [rng.random() for _ in range(300)]
        prev = -1.0
        for m in range(1, 12):
            placed = optimal_placement(posterior, m=m, window_length=9)
            p = hit_probability(placed, posterior, 9)
            self.assertGreaterEqual(p + 1e-12, prev, f"giảm khi m={m}")
            prev = p

    def test_matches_bruteforce_on_small_instances(self):
        """
        So DP với vét cạn trên bài toán nhỏ — kiểm chứng nghiệm CHÍNH XÁC,
        gồm cả lập luận "hạn chế về vị trí rời nhau là không mất tổng quát".
        """
        from itertools import combinations

        rng = random.Random(11)
        for trial in range(30):
            n, L, m = 24, 5, 3
            posterior = [rng.choice([0.0, 0.5, 1.0, 2.0]) for _ in range(n)]
            dp_frames = optimal_placement(posterior, m=m, window_length=L)
            dp_p = hit_probability(dp_frames, posterior, L)

            lo_w, hi_w = half_widths(L)
            candidates = range(-hi_w, n + lo_w)
            best = max(
                hit_probability(combo, posterior, L)
                for combo in combinations(candidates, m)
            )
            self.assertAlmostEqual(
                dp_p, best, places=9, msg=f"trial {trial}: DP={dp_p} vs vét cạn={best}"
            )

    def test_empty_posterior(self):
        self.assertEqual(optimal_placement([], m=5, window_length=9), [])

    def test_negative_posterior_rejected(self):
        with self.assertRaises(ValueError):
            optimal_placement([1.0, -0.5], m=1, window_length=3)


class TestHitProbability(unittest.TestCase):
    def test_full_coverage_when_grid_tiles_the_range(self):
        """m·Δ = tổng số khung hình và lưới lát kín ⟹ xác suất trúng = 1."""
        posterior = [1.0] * 20
        frames = [2, 7, 12, 17]  # mỗi khung phủ [f−2, f+2] ⟹ lát kín [0, 19]
        self.assertAlmostEqual(hit_probability(frames, posterior, 5), 1.0)

    def test_off_center_grid_leaves_a_gap(self):
        """
        Lưới căn giữa sai chỗ để lại lỗ hổng — đây chính là lý do cần D5
        (`optimal_placement`) thay vì luôn dùng lưới đều.
        Lưới [5, 10, 15, 19] phủ [3, 19], bỏ sót khung 0–2 ⟹ 17/20 = 0.85.
        """
        posterior = [1.0] * 20
        frames = uniform_grid(center=10, m=4, delta=5, lo=0, hi=19)
        self.assertEqual(frames, [5, 10, 15, 19])
        self.assertAlmostEqual(hit_probability(frames, posterior, 5), 0.85)

        better = optimal_placement(posterior, m=4, window_length=5)
        self.assertAlmostEqual(hit_probability(better, posterior, 5), 1.0)

    def test_no_coverage(self):
        posterior = [0.0] * 50
        posterior[45] = 1.0
        self.assertAlmostEqual(hit_probability([5], posterior, 3), 0.0)

    def test_uniform_posterior_coverage_is_span_over_total(self):
        """Posterior đều: xác suất trúng = (số khung phủ được)/(tổng số khung)."""
        n, L, m = 200, 5, 4
        posterior = [1.0] * n
        frames = uniform_grid(center=100, m=m, delta=L, lo=0, hi=n - 1)
        self.assertAlmostEqual(hit_probability(frames, posterior, L), m * L / n)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestSpreadInWindow(unittest.TestCase):
    """
    Lưới THÍCH ỨNG theo khe cục bộ. Lý do tồn tại: thể lệ nói đáp án là *khung hình
    ngữ nghĩa*, **khác** keyframe kỹ thuật đã cấp, và cửa sổ *"thường dưới 10 frame"*.
    [ĐO] cửa sổ L=9 chỉ chứa sẵn keyframe của ta **23,5%** số lần ⟹ nộp thuần keyframe
    có trần cứng ở đó.
    """

    def test_m_equals_one_returns_the_keyframe_untouched(self):
        """Ngân sách một khung thì không được đoán mò — trả đúng keyframe."""
        self.assertEqual(spread_in_window(1000, 950, 1050, 1), [1000])

    def test_covers_half_the_gap_on_each_side(self):
        got = sorted(spread_in_window(1000, 950, 1050, 5))
        self.assertEqual(got[0], 950)
        self.assertEqual(got[-1], 1050)
        self.assertEqual(len(got), 5)

    def test_keyframe_comes_FIRST_not_in_the_middle(self):
        """
        Thứ tự tính điểm: `Final` cộng `R@k` ở k ∈ {1,5,20,50,100} nên câu đầu được
        tính vào cả năm mức. [ĐO] trên bài nộp THẬT của 100 truy vấn, đưa keyframe lên
        đầu: Final (cửa sổ chặt ±4) 0,4620 → 0,5400 — tốt hơn 35 câu, tệ hơn 0.
        """
        got = spread_in_window(1000, 950, 1050, 7)
        self.assertEqual(got[0], 1000, "keyframe phải ở hạng 1")
        self.assertEqual(len(set(got)), 7, "đổi thứ tự KHÔNG được đổi tập khung")

    def test_order_moves_outward_from_the_keyframe(self):
        got = spread_in_window(1000, 950, 1050, 5)
        d = [abs(f - 1000) for f in got]
        self.assertEqual(d, sorted(d))

    def test_adapts_to_asymmetric_gaps(self):
        """Khe trái hẹp, khe phải rộng ⟹ lưới phải lệch, không đối xứng."""
        got = sorted(spread_in_window(1000, 990, 1100, 5))
        self.assertEqual(got[0], 990)
        self.assertEqual(got[-1], 1100)

    def test_output_is_deduplicated(self):
        """
        KHÔNG kiểm sắp tăng dần: hợp đồng là **keyframe trước rồi toả**, xem
        `test_keyframe_comes_FIRST_not_in_the_middle`. Bất biến còn lại là khử trùng —
        cửa sổ hẹp và `m` lớn thì phép làm tròn sinh ra khung lặp, mà nộp trùng là
        phí slot (`R@k` lấy max).
        """
        got = spread_in_window(1000, 999, 1001, 9)
        self.assertEqual(len(got), len(set(got)))
        self.assertLessEqual(len(got), 3, "cửa sổ 3 khung không thể cho quá 3 khung")

    def test_degenerate_gap_falls_back_to_the_keyframe(self):
        self.assertEqual(spread_in_window(1000, 1000, 1000, 7), [1000])

    def test_rejects_center_outside_the_window(self):
        with self.assertRaises(ValueError):
            spread_in_window(1000, 1100, 1200, 5)
        with self.assertRaises(ValueError):
            spread_in_window(1000, 900, 950, 5)

    def test_rejects_zero_budget(self):
        with self.assertRaises(ValueError):
            spread_in_window(1000, 950, 1050, 0)

    def test_a_narrow_window_is_hit_far_more_often_than_by_the_keyframe_alone(self):
        """
        Mô phỏng đúng chế độ thi: mốc thật rơi ngẫu nhiên trong khe, cửa sổ L=9.
        Nộp một keyframe trúng ~L/khe; rải 5 khung phải trúng cao hơn HẲN.
        """
        import random
        rng = random.Random(0)
        gap, L, m = 96, 9, 5
        alone = spread = 0
        for _ in range(4000):
            t = rng.randint(1000 - gap // 2, 1000 + gap // 2)
            alone += abs(1000 - t) <= (L - 1) // 2
            spread += any(abs(f - t) <= (L - 1) // 2
                          for f in spread_in_window(1000, 1000 - gap // 2, 1000 + gap // 2, m))
        self.assertGreater(spread, alone * 3)


# ── rải THÍCH ỨNG: cố định BƯỚC, suy ra m ───────────────────────────────────

import pytest as _pytest

from src.submission.coverage import (DEFAULT_SPREAD_STEP, SPREAD_M_MAX,
                                     SPREAD_M_MIN, adaptive_m)


def test_cua_so_rong_hon_thi_rai_nhieu_hon():
    """Cả lý do tồn tại: cửa sổ đo được p10=28 · p90=88, chênh hơn 3 lần."""
    assert adaptive_m(0, 27) < adaptive_m(0, 42) < adaptive_m(0, 87)


def test_buoc_giua_hai_khung_xap_xi_step():
    """Bước phải bám `step`; đó là bất biến hàm này bảo đảm."""
    for w in (30, 50, 80, 120):
        m = adaptive_m(0, w - 1)
        if SPREAD_M_MIN < m < SPREAD_M_MAX:
            assert abs((w - 1) / (m - 1) - DEFAULT_SPREAD_STEP) <= 3


def test_kep_hai_dau():
    assert adaptive_m(0, 0) == SPREAD_M_MIN        # cửa sổ 1 khung
    assert adaptive_m(0, 100_000) == SPREAD_M_MAX  # cửa sổ vô lý


def test_step_lon_hon_thi_rai_thua_hon():
    w = 100
    assert adaptive_m(0, w, step=5) > adaptive_m(0, w, step=10) > adaptive_m(0, w, step=25)


def test_tat_dinh():
    assert adaptive_m(3, 47) == adaptive_m(3, 47)


def test_dau_vao_sai_thi_NO():
    with _pytest.raises(ValueError, match="step"):
        adaptive_m(0, 10, step=0)
    with _pytest.raises(ValueError, match="rỗng"):
        adaptive_m(10, 3)


def test_m_luon_dung_duoc_cho_spread_in_window():
    """`m` sinh ra phải luôn hợp lệ với `spread_in_window`, không bao giờ < 1."""
    rng = __import__("numpy").random.default_rng(0)
    for _ in range(200):
        lo = int(rng.integers(0, 5000)); w = int(rng.integers(1, 500))
        m = adaptive_m(lo, lo + w - 1)
        assert m >= 1
        out = spread_in_window(lo + w // 2, lo, lo + w - 1, m)
        assert len(out) == len(set(out)) and all(lo <= f <= lo + w - 1 for f in out)
