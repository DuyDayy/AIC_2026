"""
Test cho `src/retrieval/score_matrix.py` — tầng ③ hợp nhất
==========================================================

Bất biến trung tâm, và là **toàn bộ lý do** tầng này tồn tại:

    E[z_m | khung CÓ modality m] = 0   =   giá trị khung KHÔNG có m nhận được

Nhờ đó **có dữ liệu không còn là lợi thế**. RRF thiếu đúng tính chất này và đo được nó
mất **−0,2927 FINAL** trên 688 truy vấn — nhóm video có OCR bị nâng hạng có hệ thống,
bất kể liên quan.

Nếu bất biến này hỏng, không có gì báo: điểm vẫn là số thực, thứ hạng vẫn sắp được, chỉ
là nguồn phủ rộng hơn thắng nguồn liên quan hơn.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.score_matrix import (
    NORMALIZERS,
    fuse,
    rank_normalize,
    score_matrix,
    z_normalize,
)
from src.retrieval.sources import SourceScores


def src(name, scores, covered=None):
    s = np.asarray(scores, dtype=np.float32)
    c = np.ones_like(s, dtype=bool) if covered is None else np.asarray(covered, dtype=bool)
    return SourceScores(name, s, c)


class TestCoverageBiasIsRemoved(unittest.TestCase):
    """Bất biến trung tâm — xem docstring module."""

    def test_expectation_over_covered_is_zero(self):
        for f in NORMALIZERS.values():
            s = np.array([1.0, 5.0, 9.0, 100.0, 0.0], dtype=np.float32)
            c = np.array([1, 1, 1, 1, 0], dtype=bool)
            self.assertAlmostEqual(float(f(s, c)[c].mean()), 0.0, places=5, msg=f.__name__)

    def test_uncovered_gets_exactly_zero(self):
        for f in NORMALIZERS.values():
            out = f(np.array([1.0, 2.0, 3.0], np.float32), np.array([1, 0, 1], bool))
            self.assertEqual(out[1], 0.0, f.__name__)

    def test_having_data_confers_no_advantage(self):
        """
        Ca mô phỏng đúng lỗi RRF: một nửa kho có OCR, nửa kia không. Nếu có dữ liệu là
        lợi thế thì trung bình nhóm phủ sẽ dương.
        """
        rng = np.random.default_rng(0)
        s = np.zeros(1000, dtype=np.float32)
        c = np.zeros(1000, dtype=bool)
        c[:500] = True
        s[:500] = rng.normal(3.0, 1.0, 500)     # điểm dương hết, như BM25
        for mode in NORMALIZERS:
            out = fuse([src("ocr", s, c)], {"ocr": 1.0}, mode)
            self.assertAlmostEqual(float(out[:500].mean()), float(out[500:].mean()),
                                   places=4, msg=mode)


class TestZNormalize(unittest.TestCase):
    def test_unit_variance_over_covered(self):
        out = z_normalize(np.array([1.0, 2.0, 3.0, 4.0], np.float32), np.ones(4, bool))
        self.assertAlmostEqual(float(out.std()), 1.0, places=5)

    def test_constant_scores_give_all_zero(self):
        """σ = 0 ⟹ nguồn không phân biệt được gì; 0 là cách nói đúng, không phải NaN."""
        out = z_normalize(np.full(5, 7.0, np.float32), np.ones(5, bool))
        self.assertTrue(np.all(out == 0))
        self.assertFalse(np.isnan(out).any())

    def test_no_covered_gives_all_zero(self):
        out = z_normalize(np.array([1.0, 2.0], np.float32), np.zeros(2, bool))
        self.assertTrue(np.all(out == 0))

    def test_order_is_preserved(self):
        s = np.array([5.0, 1.0, 3.0], np.float32)
        out = z_normalize(s, np.ones(3, bool))
        self.assertEqual(list(np.argsort(-out)), list(np.argsort(-s)))


class TestRankNormalize(unittest.TestCase):
    def test_bounded_regardless_of_outliers(self):
        """
        Điểm BM25 lệch nặng: vài khung khớp rất cao, còn lại 0. `z` để giá trị ngoại lai
        chi phối; `rank` thì không — đó là toàn bộ lý do có phương án thứ hai.
        """
        s = np.array([0, 0, 0, 0, 1e6], np.float32)
        c = np.ones(5, bool)
        self.assertLessEqual(float(np.abs(rank_normalize(s, c)).max()), 0.5)
        self.assertGreater(float(np.abs(z_normalize(s, c)).max()), 1.5)

    def test_order_is_preserved(self):
        s = np.array([5.0, 1.0, 3.0], np.float32)
        out = rank_normalize(s, np.ones(3, bool))
        self.assertEqual(list(np.argsort(-out)), list(np.argsort(-s)))

    def test_ties_get_the_SAME_value_not_spread_by_row_order(self):
        """
        Bất biến trung tâm của hàm này, và nó ĐÃ HỎNG một lần.

        Bản cũ gán mỗi khung một phân vị riêng theo thứ tự hàng, nên điểm bằng nhau
        vẫn nhận giá trị khác nhau. [ĐO] một truy vấn OCR thật: 151.615/169.409 khung
        điểm đúng bằng 0, và **66.910 khung không khớp gì nhận điểm DƯƠNG** chỉ vì
        vị trí hàng — tức vì TÊN VIDEO, do chỉ mục sắp theo `(video_id, n)`.
        """
        out = rank_normalize(np.array([0.0, 0.0, 0.0, 0.0, 5.0], np.float32),
                             np.ones(5, bool))
        self.assertEqual(len(set(out[:4].tolist())), 1,
                         "bốn điểm bằng nhau phải nhận CÙNG một giá trị")
        self.assertGreater(out[4], out[0])

    def test_a_huge_tie_block_stays_centred(self):
        """Khối hoà lớn (khuôn của BM25) phải nằm quanh 0, không lệch lên."""
        s = np.zeros(1000, dtype=np.float32)
        s[-5:] = [1.0, 2.0, 3.0, 4.0, 5.0]
        out = rank_normalize(s, np.ones(1000, bool))
        tie = out[:995]
        self.assertEqual(len(set(tie.tolist())), 1)
        self.assertLess(float(tie[0]), 0.0, "khối hoà thua 5 khung khớp thật")
        self.assertGreater(float(out[-1]), float(tie[0]))

    def test_ties_do_not_crash(self):
        out = rank_normalize(np.array([2.0, 2.0, 2.0], np.float32), np.ones(3, bool))
        self.assertFalse(np.isnan(out).any())

    def test_single_covered_frame(self):
        out = rank_normalize(np.array([9.0, 0.0], np.float32), np.array([1, 0], bool))
        self.assertFalse(np.isnan(out).any())


class TestFuse(unittest.TestCase):
    def test_weights_express_relative_trust_only(self):
        """
        Nhân mọi trọng số với cùng một số KHÔNG được đổi thứ hạng — phép chuẩn hoá đã
        đưa các nguồn về cùng đơn vị, nên chỉ TỈ LỆ có nghĩa.
        """
        a, b = src("visual", [1.0, 2.0, 3.0]), src("ocr", [3.0, 1.0, 2.0])
        one = fuse([a, b], {"visual": 1.0, "ocr": 0.5})
        two = fuse([a, b], {"visual": 4.0, "ocr": 2.0})
        self.assertEqual(list(np.argsort(-one)), list(np.argsort(-two)))

    def test_zero_weight_drops_a_source_entirely(self):
        a, b = src("visual", [1.0, 2.0, 3.0]), src("ocr", [9.0, 0.0, 0.0])
        only = fuse([a], {"visual": 1.0})
        both = fuse([a, b], {"visual": 1.0, "ocr": 0.0})
        np.testing.assert_allclose(only, both)

    def test_missing_weight_is_an_error_not_a_default(self):
        """
        Thêm nguồn mới rồi quên nối trọng số phải NỔ. Mặc định 0 làm nguồn mới im lặng
        biến mất, và không ai biết vì điểm vẫn hợp lệ.
        """
        with self.assertRaises(ValueError) as c:
            fuse([src("visual", [1.0]), src("moi", [1.0])], {"visual": 1.0})
        self.assertIn("moi", str(c.exception))

    def test_rejects_ragged_sources(self):
        with self.assertRaises(ValueError):
            fuse([src("visual", [1.0, 2.0]), src("ocr", [1.0])],
                 {"visual": 1.0, "ocr": 1.0})

    def test_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            fuse([src("visual", [1.0])], {"visual": 1.0}, mode="softmax")

    def test_empty_sources(self):
        self.assertEqual(fuse([], {}).shape, (0,))

    def test_partial_source_can_still_change_the_winner(self):
        """
        Điểm của cả tầng: nguồn phủ MỘT PHẦN vẫn được phép đổi kết quả — chỉ là không
        được đổi nhờ việc *có mặt*, mà nhờ *khớp hơn trung bình*.
        """
        vis = src("visual", [0.50, 0.52, 0.48])
        ocr = src("ocr", [9.0, 0.0, 0.0], [1, 1, 0])
        w = fuse([vis, ocr], {"visual": 1.0, "ocr": 1.0})
        self.assertEqual(int(np.argmax(w)), 0)


class TestScoreMatrix(unittest.TestCase):
    def test_single_probe_gives_one_row(self):
        S = score_matrix([[src("visual", [1.0, 2.0])]], {"visual": 1.0})
        self.assertEqual(S.shape, (1, 2))

    def test_n_probes_give_n_rows_in_order(self):
        S = score_matrix([[src("visual", [1.0, 0.0])], [src("visual", [0.0, 1.0])]],
                         {"visual": 1.0})
        self.assertEqual(S.shape, (2, 2))
        self.assertEqual(int(np.argmax(S[0])), 0)
        self.assertEqual(int(np.argmax(S[1])), 1)

    def test_kis_degenerates_to_max_over_the_single_row(self):
        """`N = 1` ⟹ `max_t S[0,t]` chính là kết quả KIS, không cần nhánh code riêng."""
        S = score_matrix([[src("visual", [0.1, 0.9, 0.3])]], {"visual": 1.0})
        self.assertEqual(int(np.argmax(S[0])), 1)

    def test_empty(self):
        self.assertEqual(score_matrix([], {}).shape, (0, 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
