"""
Kiểm chứng scorer bằng chính các ví dụ trong PDF của BTC.
=========================================================

Mỗi test dưới đây tái hiện một ví dụ có sẵn đáp số trong
"Thong tin vong So tuyen AIC2026.pdf" mục 2. Nếu test đỏ thì scorer sai,
và mọi con số tối ưu hoá dựa trên nó đều vô nghĩa.

Chạy: python -m pytest tests/test_rscore.py -v
"""

import os
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scoring.rscore import (
    Interval,
    KISAnswer,
    KISGroundTruth,
    QAAnswer,
    QAGroundTruth,
    TrakeAnswer,
    TrakeGroundTruth,
    stable_seed,
    final_score,
    r_at_k,
    r_score,
    slot_weight,
)


class TestKISExamplesFromPDF(unittest.TestCase):
    """PDF mục 2.1.1 — GT: L01_V001, khung hình 500 đến 510."""

    def setUp(self):
        self.gt = KISGroundTruth(video_id="L01_V001", window=Interval(500, 510))

    def test_correct_video_and_frame_in_window(self):
        """L01_V001, 505 → Đúng! R-Score = 1."""
        self.assertEqual(r_score(KISAnswer("L01_V001", 505), self.gt), 1.0)

    def test_correct_video_frame_outside_window(self):
        """L01_V001, 600 → Sai! R-Score = 0."""
        self.assertEqual(r_score(KISAnswer("L01_V001", 600), self.gt), 0.0)

    def test_wrong_video(self):
        """L02_V003, 505 → Sai video. R-Score = 0."""
        self.assertEqual(r_score(KISAnswer("L02_V003", 505), self.gt), 0.0)

    def test_window_is_closed_on_both_ends(self):
        """PDF dùng ký hiệu [s, e] — cả hai biên đều tính là đúng."""
        self.assertEqual(r_score(KISAnswer("L01_V001", 500), self.gt), 1.0)
        self.assertEqual(r_score(KISAnswer("L01_V001", 510), self.gt), 1.0)
        self.assertEqual(r_score(KISAnswer("L01_V001", 499), self.gt), 0.0)
        self.assertEqual(r_score(KISAnswer("L01_V001", 511), self.gt), 0.0)

    def test_window_length_matches_pdf_example(self):
        """[500, 510] → L = 11 frame. L là tham số của Định lý 1 (phủ lưới)."""
        self.assertEqual(self.gt.window.length, 11)


class TestQAExamplesFromPDF(unittest.TestCase):
    """PDF mục 2.1.2 — GT: L05_V005, [800, 900], answer "màu xanh"."""

    def setUp(self):
        self.gt = QAGroundTruth(
            video_id="L05_V005",
            window=Interval(800, 900),
            accepted_answers=frozenset({"màu xanh"}),
        )

    def test_all_three_conditions_met(self):
        """L05_V005, 888, màu xanh → Hoàn hảo! R-Score = 1."""
        self.assertEqual(r_score(QAAnswer("L05_V005", 888, "màu xanh"), self.gt), 1.0)

    def test_wrong_answer(self):
        """L05_V005, 888, màu trắng → Sai answer. R-Score = 0."""
        self.assertEqual(r_score(QAAnswer("L05_V005", 888, "màu trắng"), self.gt), 0.0)

    def test_wrong_video(self):
        """L06_V007, 888, màu xanh → Sai video. R-Score = 0."""
        self.assertEqual(r_score(QAAnswer("L06_V007", 888, "màu xanh"), self.gt), 0.0)

    def test_frame_outside_window_kills_correct_answer(self):
        """Đúng answer nhưng sai frame vẫn 0 — cả ba điều kiện là phép AND."""
        self.assertEqual(r_score(QAAnswer("L05_V005", 950, "màu xanh"), self.gt), 0.0)

    def test_answer_normalization(self):
        """Khác hoa/thường và dấu cách thừa vẫn phải khớp."""
        self.assertEqual(r_score(QAAnswer("L05_V005", 888, "  Màu   Xanh "), self.gt), 1.0)

    def test_multiple_accepted_answers(self):
        """PDF: đáp án "5" hoặc "Năm" đều được chấp nhận."""
        gt = QAGroundTruth(
            video_id="video_xyz",
            window=Interval(3400, 3500),
            accepted_answers=frozenset({"5", "Năm"}),
        )
        self.assertEqual(r_score(QAAnswer("video_xyz", 3450, "5"), gt), 1.0)
        self.assertEqual(r_score(QAAnswer("video_xyz", 3450, "năm"), gt), 1.0)
        self.assertEqual(r_score(QAAnswer("video_xyz", 3450, "6"), gt), 0.0)


class TestTrakeExampleFromPDF(unittest.TestCase):
    """
    PDF mục 2.1.3 — GT: L10_V010, 4 khoảnh khắc:
        [95,105], [145,155], [195,205], [245,255]
    Đội thi nộp: L10_V010, 101, 156, 203, 251 → khớp 3/4 → R-Score = 0.75
    """

    def setUp(self):
        self.gt = TrakeGroundTruth(
            video_id="L10_V010",
            windows=(
                Interval(95, 105),
                Interval(145, 155),
                Interval(195, 205),
                Interval(245, 255),
            ),
        )

    def test_pdf_worked_example_equals_three_quarters(self):
        answer = TrakeAnswer("L10_V010", (101, 156, 203, 251))
        self.assertAlmostEqual(r_score(answer, self.gt), 0.75)

    def test_each_moment_individually(self):
        """101 ∈ [95,105] ✓ | 156 ∉ [145,155] ✗ | 203 ✓ | 251 ✓"""
        self.assertTrue(self.gt.windows[0].contains(101))
        self.assertFalse(self.gt.windows[1].contains(156))
        self.assertTrue(self.gt.windows[2].contains(203))
        self.assertTrue(self.gt.windows[3].contains(251))

    def test_wrong_video_is_zero_even_if_every_frame_correct(self):
        """Điều kiện tiên quyết: sai video ⟹ 0 điểm ngay lập tức."""
        perfect_frames = TrakeAnswer("L99_V999", (100, 150, 200, 250))
        self.assertEqual(r_score(perfect_frames, self.gt), 0.0)

    def test_all_correct(self):
        self.assertEqual(r_score(TrakeAnswer("L10_V010", (100, 150, 200, 250)), self.gt), 1.0)

    def test_denominator_is_N_not_submitted_count(self):
        """
        Nộp thiếu mốc: mẫu số vẫn là N. PDF nói "N là tổng số khoảnh khắc trong
        truy vấn" — không phải số frame ta nộp. Nộp 2 frame đúng trên N=4 → 0.5,
        chứ không phải 1.0.
        """
        self.assertEqual(r_score(TrakeAnswer("L10_V010", (100, 150)), self.gt), 0.5)

    def test_moment_windows_are_under_10_frames(self):
        """
        PDF: "đoạn ứng với khoảnh khắc ngữ nghĩa này thường rất ngắn, thông
        thường là dưới 10 frame". Ví dụ của BTC là 11 frame — sát ngưỡng.
        Đây là `L` trong Định lý 1: quyết định bước lưới Δ tối đa.
        """
        for w in self.gt.windows:
            self.assertEqual(w.length, 11)


class TestRAtK(unittest.TestCase):
    def test_max_over_prefix(self):
        scores = [0.1, 0.9, 0.3]
        self.assertAlmostEqual(r_at_k(scores, 1), 0.1)
        self.assertAlmostEqual(r_at_k(scores, 2), 0.9)
        self.assertAlmostEqual(r_at_k(scores, 3), 0.9)

    def test_k_larger_than_list(self):
        """Nộp thiếu: lấy max phần có, không lỗi."""
        self.assertAlmostEqual(r_at_k([0.4], 100), 0.4)

    def test_empty(self):
        self.assertEqual(r_at_k([], 5), 0.0)

    def test_k_must_be_positive(self):
        with self.assertRaises(ValueError):
            r_at_k([0.5], 0)


class TestFinalScoreExampleFromPDF(unittest.TestCase):
    """
    PDF mục 2.2 — đội thi nộp 100 câu trả lời:
        câu 1  → R-Score = 0.5
        câu 3  → R-Score = 0.8 (cao nhất trong 100 câu)
        câu 15 → R-Score = 0.6
        còn lại thấp hơn
    Top1 = 0.5; Top5 = Top20 = Top50 = Top100 = 0.8
    Final = (0.5 + 0.8 + 0.8 + 0.8 + 0.8) / 5 = 0.74
    """

    def setUp(self):
        # N = 10 mốc để R-Score nhận đúng các giá trị 0.5 / 0.8 / 0.6.
        self.gt = TrakeGroundTruth(
            video_id="V",
            windows=tuple(Interval(100 * j, 100 * j + 9) for j in range(10)),
        )

    def _answer_with_hits(self, n_hits: int) -> TrakeAnswer:
        """Tạo câu trả lời trúng đúng `n_hits` mốc đầu, trượt phần còn lại."""
        frames = [100 * j if j < n_hits else -1 for j in range(10)]
        return TrakeAnswer("V", tuple(frames))

    def test_reproduces_pdf_final_score_of_0_74(self):
        answers = [self._answer_with_hits(1)] * 100
        answers[0] = self._answer_with_hits(5)  # rank 1  → 0.5
        answers[2] = self._answer_with_hits(8)  # rank 3  → 0.8
        answers[14] = self._answer_with_hits(6)  # rank 15 → 0.6

        report = final_score(answers, self.gt)

        self.assertAlmostEqual(report.per_k[1], 0.5)
        self.assertAlmostEqual(report.per_k[5], 0.8)
        self.assertAlmostEqual(report.per_k[20], 0.8)
        self.assertAlmostEqual(report.per_k[50], 0.8)
        self.assertAlmostEqual(report.per_k[100], 0.8)
        self.assertAlmostEqual(report.final, 0.74)

    def test_ranking_loss_diagnoses_sorting_problem(self):
        """
        Cùng một TẬP câu trả lời, chỉ đổi thứ tự: đưa câu tốt nhất lên rank 1
        thì Final = best (Định lý 2) và ranking_loss = 0.
        """
        answers = [self._answer_with_hits(1)] * 100
        answers[0] = self._answer_with_hits(5)
        answers[2] = self._answer_with_hits(8)
        answers[14] = self._answer_with_hits(6)
        unsorted = final_score(answers, self.gt)

        # Định lý 2: sắp giảm dần theo R-Score thật.
        resorted = sorted(answers, key=lambda a: r_score(a, self.gt), reverse=True)
        sorted_report = final_score(resorted, self.gt)

        self.assertAlmostEqual(unsorted.ranking_loss, 0.8 - 0.74)
        self.assertAlmostEqual(sorted_report.final, 0.8)
        self.assertAlmostEqual(sorted_report.ranking_loss, 0.0)
        self.assertGreater(sorted_report.final, unsorted.final)

    def test_sorting_descending_is_optimal_for_every_k(self):
        """
        Định lý 2 (dạng mạnh): sắp giảm dần đạt cận trên ĐỒNG THỜI ở mọi k.
        Kiểm bằng cách so với 200 hoán vị ngẫu nhiên.
        """
        import random

        rng = random.Random(0)
        answers = [self._answer_with_hits(h) for h in [5, 8, 6, 2, 9, 1, 3]]
        best = final_score(
            sorted(answers, key=lambda a: r_score(a, self.gt), reverse=True), self.gt
        )
        for _ in range(200):
            shuffled = answers[:]
            rng.shuffle(shuffled)
            report = final_score(shuffled, self.gt)
            for k in report.per_k:
                self.assertLessEqual(report.per_k[k], best.per_k[k] + 1e-12)
            self.assertLessEqual(report.final, best.final + 1e-12)


class TestSlotWeight(unittest.TestCase):
    """Định lý 2 — trọng số thực của từng vị trí nộp."""

    def test_weights_match_plan_table(self):
        self.assertAlmostEqual(slot_weight(1), 1.0)
        self.assertAlmostEqual(slot_weight(2), 0.8)
        self.assertAlmostEqual(slot_weight(5), 0.8)
        self.assertAlmostEqual(slot_weight(6), 0.6)
        self.assertAlmostEqual(slot_weight(20), 0.6)
        self.assertAlmostEqual(slot_weight(21), 0.4)
        self.assertAlmostEqual(slot_weight(50), 0.4)
        self.assertAlmostEqual(slot_weight(51), 0.2)
        self.assertAlmostEqual(slot_weight(100), 0.2)

    def test_no_slot_is_worthless(self):
        """Mọi slot tới 100 đều có trọng số > 0 ⟹ phải điền đủ 100."""
        for rank in range(1, 101):
            self.assertGreater(slot_weight(rank), 0.0)

    def test_weight_is_non_increasing(self):
        weights = [slot_weight(r) for r in range(1, 101)]
        self.assertEqual(weights, sorted(weights, reverse=True))

    def test_slot_1_worth_five_times_slot_100(self):
        self.assertAlmostEqual(slot_weight(1) / slot_weight(100), 5.0)


class TestEdgeCases(unittest.TestCase):
    def test_empty_submission_scores_zero(self):
        gt = KISGroundTruth("V", Interval(0, 10))
        report = final_score([], gt)
        self.assertEqual(report.final, 0.0)
        self.assertIsNone(report.best_rank)

    def test_mismatched_types_raise(self):
        gt = KISGroundTruth("V", Interval(0, 10))
        with self.assertRaises(TypeError):
            r_score(TrakeAnswer("V", (5,)), gt)

    def test_empty_interval_rejected(self):
        with self.assertRaises(ValueError):
            Interval(10, 9)

    def test_over_budget_submission_is_truncated(self):
        """Nộp 150 câu: chỉ 100 câu đầu được tính."""
        gt = KISGroundTruth("V", Interval(0, 10))
        answers = [KISAnswer("V", 999)] * 100 + [KISAnswer("V", 5)] * 50
        self.assertEqual(final_score(answers, gt).final, 0.0)


class StableSeed(unittest.TestCase):
    """
    `stable_seed` phải cho CÙNG một số ở mọi tiến trình. Nó thay `hash()` — thứ có muối
    ngẫu nhiên mỗi tiến trình và từng làm Final lệch 0,0047 giữa hai lần chấm cùng dữ
    liệu, đủ để lật argmax một phép quét trọng số.
    """

    def test_khoa_gia_tri(self):
        """Khoá cứng giá trị: đổi thuật toán băm là mọi số đã báo không còn tái lập."""
        self.assertEqual(stable_seed("q_0001"), 2435255333)
        self.assertEqual(stable_seed(""), 0)

    def test_dinh_truoc_qua_tien_trinh_khac(self):
        """Chạy ở TIẾN TRÌNH KHÁC phải ra cùng số — đây là điều `hash()` không đảm bảo."""
        out = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, %r)\n"
             "from src.scoring.rscore import stable_seed; print(stable_seed('q_0001'))"
             % os.path.dirname(os.path.dirname(os.path.abspath(__file__)))],
            capture_output=True, text=True, check=True)
        self.assertEqual(int(out.stdout.strip()), stable_seed("q_0001"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
