"""
Test cho `src/retrieval/dante.py` — tầng ④
===========================================

Ba bất biến, và cả ba đều **im lặng khi hỏng**: DP vẫn trả về một đường, điểm vẫn là
số thực, chỉ là đường sai.

1. **Thứ tự thời gian ngặt** `τ < t`. Hỏng ⟹ TRAKE trả các mốc lộn ngược hoặc trùng
   khung, mà theo luật chấm `id_{i,j}` phải nằm trong `[sⱼ, eⱼ]` của ĐÚNG mốc j.
2. **`N = 1` suy biến thành `argmax`**. Hỏng ⟹ KIS phải có nhánh code riêng, và hai
   nhánh sẽ trôi khỏi nhau.
3. **`O(N·T)` cho cùng kết quả với `O(N·T²)` viết thẳng.** Đây là chỗ dễ sai nhất:
   max luỹ tiến lệch một ô là ràng buộc thành `τ ≤ t` thay vì `τ < t`, và test bằng
   dữ liệu ngẫu nhiên đối chiếu bản chậm là cách duy nhất bắt được.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path as FsPath

import numpy as np

sys.path.insert(0, str(FsPath(__file__).resolve().parents[1]))

from src.retrieval.dante import DEFAULT_LAMBDA, dante, dante_over_videos


def slow(S, times_ms, lam):
    """`O(N·T²)` viết thẳng từ công thức — mốc đối chiếu, không tối ưu gì."""
    S = np.asarray(S, dtype=np.float64)
    t = (np.asarray(times_ms, dtype=np.float64) - times_ms[0]) / 1000.0
    n, T = S.shape
    dp = np.full((n, T), -np.inf)
    dp[0] = S[0]
    for i in range(1, n):
        for tt in range(T):
            best = -np.inf
            for tau in range(tt):
                v = dp[i - 1, tau] - lam * (t[tt] - t[tau])
                if v > best:
                    best = v
            dp[i, tt] = S[i, tt] + best if np.isfinite(best) else -np.inf
    return float(np.max(dp[n - 1]))


def ms(*xs):
    return np.array(xs, dtype=np.float64)


class TestStrictTimeOrder(unittest.TestCase):
    """Bất biến 1."""

    def test_moments_come_out_strictly_increasing(self):
        rng = np.random.default_rng(0)
        for _ in range(30):
            n, T = rng.integers(1, 5), rng.integers(5, 25)
            S = rng.normal(size=(n, T))
            p = dante(S, np.cumsum(rng.integers(200, 4000, T)).astype(float), 0.05)
            self.assertEqual(len(p.cols), n)
            self.assertEqual(list(p.cols), sorted(set(p.cols)),
                             "mốc phải tăng NGẶT, không trùng khung")

    def test_best_column_is_not_reused_even_when_hugely_attractive(self):
        """
        Cột 2 tốt cho CẢ HAI mốc. Nếu `τ ≤ t` thay vì `τ < t` thì cả hai cùng chọn
        cột 2 — và bài nộp TRAKE sẽ có hai mốc trùng khung.
        """
        S = np.array([[0, 0, 9, 0], [0, 0, 9, 0]], dtype=float)
        p = dante(S, ms(0, 1000, 2000, 3000), 0.0)
        self.assertEqual(len(set(p.cols)), 2)

    def test_fewer_frames_than_moments_has_no_valid_path(self):
        p = dante(np.zeros((4, 3)), ms(0, 1000, 2000), 0.1)
        self.assertEqual(p.cols, ())
        self.assertEqual(p.score, -np.inf)

    def test_exactly_enough_frames_forces_the_only_path(self):
        p = dante(np.zeros((3, 3)), ms(0, 1000, 2000), 0.1)
        self.assertEqual(p.cols, (0, 1, 2))


class TestKisDegenerates(unittest.TestCase):
    """Bất biến 2 — KIS không cần nhánh riêng."""

    def test_n_equals_one_is_plain_argmax(self):
        S = np.array([[0.1, 0.9, 0.3, 0.7]])
        p = dante(S, ms(0, 1000, 2000, 3000), DEFAULT_LAMBDA)
        self.assertEqual(p.cols, (1,))
        self.assertAlmostEqual(p.score, 0.9)

    def test_lambda_never_affects_the_single_moment_case(self):
        S = np.array([[0.1, 0.9, 0.3]])
        a = dante(S, ms(0, 60_000, 120_000), 0.0)
        b = dante(S, ms(0, 60_000, 120_000), 99.0)
        self.assertEqual(a, b, "không có khoảng cách nào để phạt khi N=1")


class TestMatchesBruteForce(unittest.TestCase):
    """Bất biến 3 — max luỹ tiến phải bằng bản quét thẳng."""

    def test_random_agreement(self):
        rng = np.random.default_rng(7)
        for _ in range(60):
            n, T = int(rng.integers(1, 5)), int(rng.integers(4, 30))
            if T < n:
                continue
            S = rng.normal(size=(n, T))
            t = np.cumsum(rng.integers(100, 5000, T)).astype(float)
            lam = float(rng.choice([0.0, 0.02, 0.1, 0.5, 3.0]))
            self.assertAlmostEqual(dante(S, t, lam).score, slow(S, t, lam), places=8)

    def test_path_score_equals_reported_score(self):
        """Backtrack phải trả đúng đường sinh ra điểm — không chỉ đúng điểm."""
        rng = np.random.default_rng(3)
        for _ in range(25):
            n, T = int(rng.integers(2, 5)), int(rng.integers(6, 20))
            S = rng.normal(size=(n, T))
            t = np.cumsum(rng.integers(100, 4000, T)).astype(float)
            lam = 0.08
            p = dante(S, t, lam)
            ts = (t - t[0]) / 1000.0
            man = sum(S[i, c] for i, c in enumerate(p.cols))
            man -= lam * sum(ts[b] - ts[a] for a, b in zip(p.cols, p.cols[1:]))
            self.assertAlmostEqual(p.score, man, places=8)


class TestLambdaPullsMomentsTogether(unittest.TestCase):
    def test_large_lambda_prefers_the_tight_cluster(self):
        """
        Hai lựa chọn: cặp điểm CAO nhưng cách 100 giây, hay cặp điểm thấp hơn mà kề
        nhau. λ lớn phải chọn cụm kề.
        """
        S = np.array([[9.0, 0.0, 5.0, 0.0],
                      [0.0, 9.0, 0.0, 5.0]])
        t = ms(0, 100_000, 200_000, 201_000)
        self.assertEqual(dante(S, t, 0.0).cols, (0, 1))
        self.assertEqual(dante(S, t, 0.5).cols, (2, 3))

    def test_lambda_zero_ignores_distance_entirely(self):
        S = np.array([[9.0, 0, 0, 0], [0, 0, 0, 9.0]])
        self.assertEqual(dante(S, ms(0, 1e5, 2e5, 3e5), 0.0).cols, (0, 3))

    def test_penalty_uses_time_not_keyframe_count(self):
        """
        Trục là mili-giây, không phải số thứ tự cột. Hai bố cục có CÙNG số cột nhưng
        khoảng thời gian khác nhau phải cho kết quả khác nhau — nếu không thì bước
        keyframe không đều (p10=19, p90=105 khung) bị coi như nhau.
        """
        S = np.array([[9.0, 0.0, 5.0], [0.0, 9.0, 5.0]])
        near = dante(S, ms(0, 500, 1000), 1.0).score
        far = dante(S, ms(0, 60_000, 120_000), 1.0).score
        self.assertGreater(near, far)


class TestGuards(unittest.TestCase):
    def test_rejects_unsorted_times(self):
        with self.assertRaises(ValueError):
            dante(np.zeros((2, 3)), ms(0, 5000, 1000), 0.1)

    def test_rejects_length_mismatch(self):
        with self.assertRaises(ValueError):
            dante(np.zeros((2, 3)), ms(0, 1000), 0.1)

    def test_rejects_negative_lambda(self):
        with self.assertRaises(ValueError):
            dante(np.zeros((2, 3)), ms(0, 1000, 2000), -0.1)

    def test_rejects_1d_score_matrix(self):
        with self.assertRaises(ValueError):
            dante(np.zeros(5), ms(0, 1, 2, 3, 4), 0.1)

    def test_equal_timestamps_are_allowed(self):
        """Hai keyframe cùng mốc ms là dữ liệu bẩn nhưng có thật — không được nổ."""
        p = dante(np.array([[1.0, 2.0], [3.0, 4.0]]), ms(1000, 1000), 0.1)
        self.assertEqual(p.cols, (0, 1))


class TestOverVideos(unittest.TestCase):
    RANGES = {"A": (0, 4), "B": (4, 9)}
    TIMES = ms(0, 1000, 2000, 3000, 0, 1000, 2000, 3000, 4000)

    def test_ranks_videos_by_best_path(self):
        S = np.zeros((2, 9))
        S[0, 5] = 5.0
        S[1, 6] = 5.0        # B có đường tốt
        S[0, 0] = 1.0
        S[1, 1] = 1.0        # A có đường yếu
        got = dante_over_videos(S, self.RANGES, self.TIMES, 0.0)
        self.assertEqual([v for v, _ in got], ["B", "A"])

    def test_cols_are_slice_local_not_global(self):
        S = np.zeros((2, 9))
        S[0, 5] = 5.0
        S[1, 6] = 5.0
        vid, p = dante_over_videos(S, self.RANGES, self.TIMES, 0.0)[0]
        self.assertEqual((vid, p.cols), ("B", (1, 2)),
                         "cols phải là chỉ số TRONG lát; khung tuyệt đối = lo + col")

    def test_video_shorter_than_n_is_dropped_not_crashing(self):
        """A có 4 khung, B có 5. N=5 ⟹ A rớt, B vừa đủ một đường duy nhất."""
        got = dante_over_videos(np.zeros((5, 9)), self.RANGES, self.TIMES, 0.1)
        self.assertEqual([v for v, _ in got], ["B"])
        self.assertEqual(got[0][1].cols, (0, 1, 2, 3, 4))

    def test_all_videos_too_short_gives_empty(self):
        got = dante_over_videos(np.zeros((6, 9)), self.RANGES, self.TIMES, 0.1)
        self.assertEqual(got, [])


    def test_slice_out_of_time_order_is_sorted_not_rejected(self):
        """
        [ĐO] 20/873 video có `pts_time` tụt lại khi `n` tăng. Chỉ mục sắp theo `n`, còn
        `τ < t` của DP nói về THỜI GIAN — nên lát phải được sắp lại, và `cols` trả về
        phải là chỉ số trong lát GỐC, không phải trong lát đã sắp.
        """
        rng = {"A": (0, 4)}
        t = ms(0, 3000, 1000, 2000)          # cột 1 và 2 đảo nhau
        S = np.zeros((2, 4))
        S[0, 2] = 5.0                        # mốc 0 ở giây 1
        S[1, 3] = 5.0                        # mốc 1 ở giây 2 — hợp lệ theo THỜI GIAN
        got = dante_over_videos(S, rng, t, 0.0)
        self.assertEqual(got[0][1].cols, (2, 3))

    def test_a_path_valid_by_column_but_invalid_by_time_is_refused(self):
        """
        Cái bẫy: cột 1 → cột 2 tăng theo CỘT nhưng giây 3 → giây 1 là LÙI theo thời
        gian. Nếu DP dùng thứ tự cột thì nó cắn câu này (điểm 8,0) thay vì đường thật
        (điểm 2,0). Đây chính là chế độ hỏng của 20 video có `n` lệch thời gian.
        """
        rng = {"A": (0, 4)}
        t = ms(3000, 0, 1000, 2000)          # cột 1 là khung SỚM NHẤT
        S = np.zeros((2, 4))
        S[0, 0] = 4.0                        # mốc 0 ở giây 3
        S[1, 1] = 4.0                        # mốc 1 ở giây 0 — sớm nhất, KHÔNG có
        #                                      tiền nhiệm nào theo thời gian
        S[0, 1] = 1.0                        # mốc 0 ở giây 0
        S[1, 2] = 1.0                        # mốc 1 ở giây 1  ← đường thật, 2,0
        got = dante_over_videos(S, rng, t, 0.0)
        self.assertEqual(got[0][1].cols, (1, 2))
        self.assertAlmostEqual(got[0][1].score, 2.0,
                               msg="theo thứ tự CỘT thì bẫy cho 8,0 — phải bị loại")

    def test_restricting_to_a_candidate_list(self):
        got = dante_over_videos(np.zeros((2, 9)), self.RANGES, self.TIMES, 0.1,
                                videos=["A"])
        self.assertEqual([v for v, _ in got], ["A"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
