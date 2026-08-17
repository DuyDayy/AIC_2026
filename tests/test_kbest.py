"""
Kiểm chứng k-best Viterbi (D3) và Định lý 3.
============================================

Test xương sống là `test_matches_bruteforce`: so k đường tốt nhất do List
Viterbi trả về với vét cạn toàn bộ tổ hợp trên bài toán nhỏ. Nếu khớp trên
hàng chục instance ngẫu nhiên thì cài đặt đúng.

`TestTheorem3` nối k-best với scorer thật để chứng minh bằng số rằng hệ theo
BỘ (tuple) thắng hệ theo TỪNG MỐC — lý do tồn tại của cả module này.

NGUỒN CỦA CÁC GIÁ TRỊ KỲ VỌNG:
  [ĐL3]     Định lý 3 (TRAKE không tách được theo mốc), BAO_CAO_THIET_KE.md §2:
            `max_i (1/N)Σⱼ I_ij ≤ (1/N)Σⱼ max_i I_ij`, thường NGẶT. Hệ quả:
            hedging phải làm trên BỘ N-tuple hoàn chỉnh, không phải từng mốc.
  [ĐL4]     Định lý 4 (ngân sách slot): tích Descartes `m^N ≤ B`. Với B=100,
            N=4 ⟹ m=3 (3⁴=81 ≤ 100). Đây là nguồn của con số 81 trong
            `test_cartesian_tuple_hedging_reaches_one`.
  [PDF]     `B = 100` câu trả lời/truy vấn (mục 2), và công thức R-Score TRAKE
            (mục 2.1.3) dùng để chấm trong `TestTheorem3`.
  [LIST-VIT] Tính đúng của List Viterbi song song: đường tốt thứ k vào trạng
            thái `(j,i)` phải có phần đầu nằm trong K đường tốt nhất của trạng
            thái tiền nhiệm (nếu không, thay phần đầu cho đường tốt hơn ⟹ mâu
            thuẫn). Lập luận đầy đủ ở docstring `k_best_alignments`.
  [VÉT-CẠN] `test_matches_bruteforce` so với vét cạn tổ hợp trên 40 instance
            ngẫu nhiên (N, M, k, min_gap đều random) — tự thân là bằng chứng.
  [NHỊP-ĐỘ] Phạt decay theo khoảng cách thời gian `exp(-λΔt)`, áp NGAY TRONG
            quy hoạch động theo hệ thức kiểu DANTE
            `DP[i,t] = S[i,t] + max_τ(DP[i−1,τ] − λ(t−τ))`. Kiểm chứng bằng vét
            cạn có phạt (`brute_force(..., pacing_penalty=λ)`), tính thẳng từ
            định nghĩa chứ không mượn lại hệ thức truy hồi.
  [HẬU-KỲ]  `TestPacingInsideDpVsPostHoc` dựng một ca mà bộ tối ưu-có-phạt nằm
            NGOÀI top-k thô. Hậu kỳ không thể tìm ra nó vì nó không có trong đầu
            vào — đây là bằng chứng rằng đưa phạt vào DP không phải tối ưu hoá
            phong cách mà là sửa một chế độ hỏng thật.

Chạy: python tests/test_kbest.py
"""

import os
import random
import sys
import unittest
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scoring.rscore import Interval, TrakeAnswer, TrakeGroundTruth, final_score
from src.submission.kbest import AlignedTuple, apply_pacing_penalty, best_alignment, k_best_alignments


def brute_force(candidates, scores, k, min_gap=1, pacing_penalty=0.0):
    """
    Vét cạn mọi bộ tăng dần thoả min_gap, trả k điểm tốt nhất.

    Phạt nhịp độ tính TRỰC TIẾP từ định nghĩa (`λ × tổng khoảng cách`), không
    mượn lại hệ thức truy hồi của cài đặt — nếu không, test sẽ chỉ xác nhận
    rằng cài đặt bằng chính nó.
    """
    n, m = len(scores), len(candidates)
    out = []
    for combo in combinations(range(m), n):
        if all(
            candidates[combo[j + 1]] - candidates[combo[j]] >= min_gap for j in range(n - 1)
        ):
            total = sum(scores[j][combo[j]] for j in range(n))
            span = sum(
                candidates[combo[j + 1]] - candidates[combo[j]] for j in range(n - 1)
            )
            out.append(total - pacing_penalty * span)
    out.sort(reverse=True)
    return out[:k]


class TestKBestCorrectness(unittest.TestCase):
    def test_matches_bruteforce(self):
        """List Viterbi phải khớp vét cạn trên 40 instance ngẫu nhiên."""
        rng = random.Random(42)
        for trial in range(40):
            n = rng.randint(1, 4)
            m = rng.randint(n, 9)
            candidates = sorted(rng.sample(range(0, 60), m))
            scores = [[rng.uniform(-3, 3) for _ in range(m)] for _ in range(n)]
            k = rng.randint(1, 12)
            min_gap = rng.choice([1, 1, 3, 7])

            got = [t.score for t in k_best_alignments(candidates, scores, k, min_gap=min_gap)]
            want = brute_force(candidates, scores, k, min_gap=min_gap)

            self.assertEqual(len(got), len(want), f"trial {trial}: số lượng lệch")
            for a, b in zip(got, want):
                self.assertAlmostEqual(a, b, places=9, msg=f"trial {trial}")

    def test_returned_frames_reconstruct_their_score(self):
        """Truy vết phải đúng: điểm của bộ trả về = tổng điểm các ô đã chọn."""
        rng = random.Random(3)
        candidates = sorted(rng.sample(range(200), 12))
        scores = [[rng.uniform(0, 1) for _ in candidates] for _ in range(3)]
        index = {f: i for i, f in enumerate(candidates)}

        for tup in k_best_alignments(candidates, scores, k=10):
            recomputed = sum(scores[j][index[f]] for j, f in enumerate(tup.frames))
            self.assertAlmostEqual(tup.score, recomputed, places=9)

    def test_results_sorted_descending(self):
        rng = random.Random(5)
        candidates = list(range(0, 40, 2))
        scores = [[rng.uniform(0, 1) for _ in candidates] for _ in range(3)]
        got = k_best_alignments(candidates, scores, k=15)
        self.assertEqual([t.score for t in got], sorted((t.score for t in got), reverse=True))

    def test_all_tuples_distinct(self):
        rng = random.Random(9)
        candidates = list(range(20))
        scores = [[rng.uniform(0, 1) for _ in candidates] for _ in range(3)]
        got = k_best_alignments(candidates, scores, k=50)
        frames = [t.frames for t in got]
        self.assertEqual(len(frames), len(set(frames)), "có bộ trùng lặp")


class TestConstraints(unittest.TestCase):
    def test_temporal_order_strictly_increasing(self):
        rng = random.Random(1)
        candidates = list(range(30))
        scores = [[rng.uniform(0, 1) for _ in candidates] for _ in range(4)]
        for tup in k_best_alignments(candidates, scores, k=20):
            self.assertEqual(list(tup.frames), sorted(tup.frames))
            self.assertEqual(len(set(tup.frames)), len(tup.frames))

    def test_min_gap_respected(self):
        rng = random.Random(2)
        candidates = list(range(0, 100))
        scores = [[rng.uniform(0, 1) for _ in candidates] for _ in range(3)]
        gap = 15
        for tup in k_best_alignments(candidates, scores, k=20, min_gap=gap):
            for a, b in zip(tup.frames, tup.frames[1:]):
                self.assertGreaterEqual(b - a, gap)

    def test_infeasible_returns_empty(self):
        """M < N: không thể xếp N mốc tăng dần trên M khung hình."""
        candidates = [10, 20]
        scores = [[1.0, 1.0]] * 5
        self.assertEqual(k_best_alignments(candidates, scores, k=5), [])

    def test_min_gap_too_large_returns_empty(self):
        candidates = [0, 1, 2]
        scores = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        self.assertEqual(k_best_alignments(candidates, scores, k=3, min_gap=100), [])

    def test_single_moment_is_top_k_frames(self):
        """N=1 suy biến thành: lấy k khung hình điểm cao nhất."""
        candidates = [5, 10, 15, 20]
        scores = [[0.1, 0.9, 0.3, 0.7]]
        got = k_best_alignments(candidates, scores, k=3)
        self.assertEqual([t.frames for t in got], [(10,), (20,), (15,)])


class TestValidation(unittest.TestCase):
    def test_non_increasing_candidates_rejected(self):
        with self.assertRaises(ValueError):
            k_best_alignments([10, 5], [[1.0, 1.0]], k=1)

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            k_best_alignments([1, 2, 3], [[1.0, 1.0]], k=1)

    def test_bad_k_rejected(self):
        with self.assertRaises(ValueError):
            k_best_alignments([1, 2], [[1.0, 1.0]], k=0)

    def test_empty_inputs(self):
        self.assertEqual(k_best_alignments([], [], k=5), [])
        self.assertIsNone(best_alignment([], []))


class TestTheorem3(unittest.TestCase):
    """
    Định lý 3 bằng số: hệ theo BỘ thắng hệ theo TỪNG MỐC.

    Kịch bản: N=4 mốc. Ta biết mỗi mốc nằm ở một trong 3 vị trí khả dĩ, và
    đúng vị trí thật nằm trong đó. Hai chiến lược cùng ngân sách:
      (a) Hệ theo mốc: nộp 4 câu, câu j chỉ chỉnh mốc j cho đúng.
      (b) Hệ theo bộ : nộp tích Descartes 3⁴ = 81 bộ (Định lý 4).
    """

    def setUp(self):
        self.gt = TrakeGroundTruth(
            video_id="V",
            windows=(
                Interval(100, 108),
                Interval(200, 208),
                Interval(300, 308),
                Interval(400, 408),
            ),
        )
        # Với mỗi mốc, 3 ứng viên; ứng viên thứ 2 là đúng.
        self.per_moment = [
            [50, 104, 150],
            [180, 204, 250],
            [280, 304, 350],
            [380, 404, 450],
        ]

    def test_per_moment_hedging_caps_at_one_over_N(self):
        """Mỗi câu chỉ đúng 1 mốc ⟹ R-Score mỗi câu = 1/4, Final = 0.25."""
        answers = []
        for j in range(4):
            frames = [self.per_moment[i][2 if i != j else 1] for i in range(4)]
            answers.append(TrakeAnswer("V", tuple(frames)))
        report = final_score(answers, self.gt)
        self.assertAlmostEqual(report.best, 0.25)
        self.assertAlmostEqual(report.final, 0.25)

    def test_cartesian_tuple_hedging_reaches_one(self):
        """Tích Descartes 3⁴ = 81 ≤ 100 bộ ⟹ có 1 bộ đúng cả 4 mốc."""
        from itertools import product

        answers = [TrakeAnswer("V", combo) for combo in product(*self.per_moment)]
        self.assertEqual(len(answers), 81)
        self.assertLessEqual(len(answers), 100, "phải nằm trong ngân sách B=100")

        report = final_score(answers, self.gt)
        self.assertAlmostEqual(report.best, 1.0)
        self.assertGreater(report.final, 0.25, "phải thắng hệ theo mốc")

    def test_kbest_puts_the_perfect_tuple_first_when_scores_are_informative(self):
        """
        Nối với k-best: nếu điểm phản ánh đúng độ tin, bộ hoàn hảo phải ra
        hạng 1 ⟹ Final = 1.0 (Định lý 2: sắp giảm dần thì Final = best).
        """
        candidates = sorted({f for row in self.per_moment for f in row})
        index = {f: i for i, f in enumerate(candidates)}
        # Ứng viên đúng có điểm cao nhất trong mỗi mốc.
        scores = [[-9.0] * len(candidates) for _ in range(4)]
        for j, opts in enumerate(self.per_moment):
            for rank, f in enumerate(opts):
                scores[j][index[f]] = 1.0 if rank == 1 else 0.2

        tuples = k_best_alignments(candidates, scores, k=100, min_gap=1)
        answers = [TrakeAnswer("V", t.frames) for t in tuples]

        report = final_score(answers, self.gt)
        self.assertAlmostEqual(report.per_k[1], 1.0)
        self.assertAlmostEqual(report.final, 1.0)
        self.assertAlmostEqual(report.ranking_loss, 0.0)


class TestPacingPenalty(unittest.TestCase):
    """
    apply_pacing_penalty — tái xếp hạng theo nhịp độ (ý tưởng TRAKE beam
    search exp(-0.01·Δt) mượn từ dự án tham chiếu).
    """

    def test_zero_decay_preserves_scores_but_sorts(self):
        tuples = [AlignedTuple((0, 100), 0.5), AlignedTuple((0, 10), 0.9)]
        result = apply_pacing_penalty(tuples, decay_rate=0.0)
        self.assertEqual([t.score for t in result], [0.9, 0.5])
        self.assertEqual([t.frames for t in result], [(0, 10), (0, 100)])

    def test_penalizes_wide_gaps_more_than_narrow(self):
        """Hai bộ cùng điểm gốc: bộ có mốc gần nhau phải thắng sau khi phạt."""
        tight = AlignedTuple((100, 110, 120), score=1.0)  # tổng gap = 20
        wide = AlignedTuple((0, 500, 1000), score=1.0)  # tổng gap = 1000
        result = apply_pacing_penalty([wide, tight], decay_rate=0.01)
        self.assertEqual(result[0].frames, tight.frames)
        self.assertLess(result[1].score, result[0].score)

    def test_can_flip_ranking_when_raw_score_gap_is_small(self):
        """
        Bộ điểm-gốc cao hơn NHƯNG rải rác phải tụt xuống dưới bộ điểm thấp hơn
        chút nhưng nhịp độ hợp lý, nếu decay đủ lớn — đúng ý tưởng TRAKE beam.
        """
        scattered = AlignedTuple((0, 1000), score=1.0)  # gap=1000
        close = AlignedTuple((0, 10), score=0.95)  # gap=10
        result = apply_pacing_penalty([scattered, close], decay_rate=0.01)
        self.assertEqual(result[0].frames, close.frames, "nhịp độ hợp lý phải thắng")

    def test_recomputed_score_formula(self):
        t = AlignedTuple((10, 25, 50), score=2.0)  # gaps: 15+25=40
        [result] = apply_pacing_penalty([t], decay_rate=0.1)
        self.assertAlmostEqual(result.score, 2.0 - 0.1 * 40)

    def test_single_frame_tuple_unaffected(self):
        """N=1 (KIS/QA suy biến): không có cặp liên tiếp nào để phạt."""
        t = AlignedTuple((42,), score=0.7)
        [result] = apply_pacing_penalty([t], decay_rate=0.5)
        self.assertAlmostEqual(result.score, 0.7)

    def test_rejects_negative_decay(self):
        with self.assertRaises(ValueError):
            apply_pacing_penalty([AlignedTuple((0, 1), 1.0)], decay_rate=-0.1)

    def test_empty_input(self):
        self.assertEqual(apply_pacing_penalty([], decay_rate=0.1), [])

    def test_composes_with_kbest_output_end_to_end(self):
        """Chạy k_best_alignments thật rồi tái xếp hạng — không cần dữ liệu giả lập tay."""
        candidates = list(range(0, 200, 5))
        # Cố tình làm 2 bộ đồng điểm khớp-mốc nhưng nhịp độ khác xa nhau.
        scores = [[0.0] * len(candidates) for _ in range(2)]
        i0, i1 = candidates.index(0), candidates.index(15)  # gần nhau
        j0, j1 = candidates.index(0), candidates.index(195)  # xa nhau
        for i in (i0, j0):
            scores[0][i] = 1.0
        for i in (i1, j1):
            scores[1][i] = 1.0

        raw = k_best_alignments(candidates, scores, k=20, min_gap=1)
        repaced = apply_pacing_penalty(raw, decay_rate=0.05)

        self.assertEqual(repaced[0].frames, (0, 15), "sau khi phạt nhịp độ, cặp gần phải lên hạng 1")


class TestPacingInsideDp(unittest.TestCase):
    """Phạt nhịp độ nằm TRONG DP — kiểm chứng bằng vét cạn có phạt."""

    def test_matches_bruteforce_with_penalty(self):
        """[VÉT-CẠN] 40 instance ngẫu nhiên, λ ngẫu nhiên."""
        rng = random.Random(1234)
        for trial in range(40):
            n = rng.randint(1, 4)
            m = rng.randint(n, 9)
            candidates = sorted(rng.sample(range(0, 60), m))
            scores = [[rng.uniform(-3, 3) for _ in range(m)] for _ in range(n)]
            k = rng.randint(1, 12)
            min_gap = rng.choice([1, 1, 3, 7])
            lam = rng.choice([0.0, 0.001, 0.02, 0.15])

            got = [
                t.score
                for t in k_best_alignments(
                    candidates, scores, k, min_gap=min_gap, pacing_penalty=lam
                )
            ]
            want = brute_force(candidates, scores, k, min_gap=min_gap, pacing_penalty=lam)

            self.assertEqual(len(got), len(want), f"trial {trial}: số lượng lệch")
            for a, b in zip(got, want):
                self.assertAlmostEqual(a, b, places=9, msg=f"trial {trial}, λ={lam}")

    def test_zero_penalty_is_identical_to_unpenalised(self):
        """λ=0 phải TRÙNG KHỚP đường cũ — nếu không, mọi test cũ mất hiệu lực."""
        rng = random.Random(7)
        candidates = sorted(rng.sample(range(300), 15))
        scores = [[rng.uniform(0, 1) for _ in candidates] for _ in range(3)]

        plain = k_best_alignments(candidates, scores, k=20)
        zero = k_best_alignments(candidates, scores, k=20, pacing_penalty=0.0)

        self.assertEqual([t.frames for t in plain], [t.frames for t in zero])
        for a, b in zip(plain, zero):
            self.assertAlmostEqual(a.score, b.score, places=12)

    def test_score_equals_raw_minus_penalty(self):
        """`score` trả về là giá trị mục tiêu ĐÃ trừ phạt, không phải điểm thô."""
        rng = random.Random(11)
        candidates = sorted(rng.sample(range(400), 12))
        scores = [[rng.uniform(0, 1) for _ in candidates] for _ in range(3)]
        index = {f: i for i, f in enumerate(candidates)}
        lam = 0.01

        for tup in k_best_alignments(candidates, scores, k=8, pacing_penalty=lam):
            raw = sum(scores[j][index[f]] for j, f in enumerate(tup.frames))
            span = tup.frames[-1] - tup.frames[0]
            self.assertAlmostEqual(tup.score, raw - lam * span, places=9)

    def test_larger_penalty_never_widens_the_best_tuple(self):
        """Tăng λ ⟹ bộ tốt nhất không thể trải rộng hơn. Tính đơn điệu."""
        rng = random.Random(21)
        candidates = list(range(0, 500, 5))
        scores = [[rng.uniform(0, 1) for _ in candidates] for _ in range(3)]

        spans = []
        for lam in (0.0, 0.005, 0.05, 0.5):
            best = best_alignment(candidates, scores, pacing_penalty=lam)
            spans.append(best.frames[-1] - best.frames[0])
        self.assertEqual(spans, sorted(spans, reverse=True), f"spans={spans}")

    def test_rejects_negative_penalty(self):
        with self.assertRaises(ValueError):
            k_best_alignments([0, 1, 2], [[1.0, 1.0, 1.0]], k=1, pacing_penalty=-0.1)

    def test_respects_min_gap_with_penalty(self):
        """Phạt không được phép nới lỏng ràng buộc cứng."""
        candidates = list(range(0, 100, 5))
        scores = [[1.0] * len(candidates) for _ in range(3)]
        for tup in k_best_alignments(candidates, scores, k=10, min_gap=20, pacing_penalty=0.01):
            gaps = [b - a for a, b in zip(tup.frames, tup.frames[1:])]
            self.assertTrue(all(g >= 20 for g in gaps), tup.frames)

    def test_single_moment_penalty_is_noop(self):
        """1 mốc ⟹ không có khoảng cách nào để phạt."""
        candidates = [0, 50, 900]
        scores = [[0.1, 0.9, 0.5]]
        self.assertEqual(best_alignment(candidates, scores, pacing_penalty=0.5).frames, (50,))


class TestPacingInsideDpVsPostHoc(unittest.TestCase):
    """
    [HẬU-KỲ] Vì sao hậu kỳ KHÔNG thay được phạt trong DP.

    Dựng ma trận điểm sao cho bộ tối ưu-có-phạt có tổng điểm khớp-mốc thô THẤP
    hơn `k` bộ khác. Nó bị loại ngay ở vòng sinh k-best, nên
    `apply_pacing_penalty` — vốn chỉ xếp lại đầu vào — không bao giờ thấy nó.
    """

    def _setup(self):
        # 30 khung cách đều 10 khung hình.
        candidates = list(range(0, 300, 10))
        idx = {c: i for i, c in enumerate(candidates)}
        n = 2
        scores = [[0.0] * len(candidates) for _ in range(n)]

        # 90 cặp "rải rác" điểm cao: mốc 0 ở khung 0–80, mốc 1 ở khung 200–290.
        # Điểm thô 1.80 mỗi cặp — cao hơn cặp liền mạch bên dưới.
        for t in range(9):
            scores[0][idx[10 * t]] = 0.90
        for t in range(10):
            scores[1][idx[200 + 10 * t]] = 0.90

        # MỘT cặp "liền mạch" điểm thô THẤP hơn: 100 → 110, cách nhau 10 khung.
        # Có phạt λ=0.01: cặp này được 1.60 − 0.10 = 1.50, mọi cặp rải rác tốt
        # nhất chỉ được 1.80 − 0.01×120 = 0.60. Thắng NGẶT, không hoà.
        scores[0][idx[100]] = 0.80
        scores[1][idx[110]] = 0.80
        return candidates, scores

    def test_post_hoc_cannot_find_the_well_paced_tuple(self):
        candidates, scores = self._setup()
        k = 20

        raw = k_best_alignments(candidates, scores, k=k)
        self.assertNotIn(
            (100, 110), [t.frames for t in raw],
            "tiền đề của test hỏng: bộ nhịp độ tốt lại nằm sẵn trong top-k thô",
        )

        repaced = apply_pacing_penalty(raw, decay_rate=0.01)
        self.assertNotEqual(
            repaced[0].frames, (100, 110),
            "hậu kỳ không thể trả về bộ không có trong đầu vào của nó",
        )

    def test_penalty_inside_dp_finds_it(self):
        candidates, scores = self._setup()
        inside = k_best_alignments(candidates, scores, k=20, pacing_penalty=0.01)
        self.assertEqual(
            inside[0].frames, (100, 110),
            "phạt trong DP phải giữ được đường nhịp độ tốt ngay từ vòng sinh",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
