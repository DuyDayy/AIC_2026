"""
Test cho `src/retrieval/rerank.py` — tầng ⑤
============================================

Bất biến trung tâm: **khung không có mảnh cắt phải là `covered=False`, không phải điểm
0**. Gộp hai thứ đó chính là lỗi đã giết RRF (−0,2927 FINAL): mọi khung CÓ mảnh cắt được
lợi bất kể liên quan.

Bất biến thứ hai: **hai mảng song song `refs` và `crop_vecs` lệch nhau phải NỔ**, vì
lệch một phần tử là điểm gắn nhầm khung — im lặng hoàn toàn.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.rerank import (
    MAX_PER_FRAME,
    MIN_AREA,
    PAD,
    CropRef,
    collect_crops,
    crop_scores,
    vlm_scores,
)

IDS = [("A", 1), ("A", 2), ("B", 1)]


def ref(row, area=0.2, ent="Clothing", box=(0.1, 0.1, 0.5, 0.5)):
    return CropRef(row, "A", 1, ent, box, area)


def write_objects(root: Path, vid: str, n: int, items):
    d = root / vid
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{n:03d}.json").write_text(json.dumps({
        "detection_class_entities": [e for e, _ in items],
        "detection_boxes": [list(b) for _, b in items],
    }), encoding="utf-8")


class TestCoverageIsNotZero(unittest.TestCase):
    """Bất biến trung tâm."""

    def test_frame_without_crop_is_uncovered(self):
        r = crop_scores(3, [ref(0)], np.array([[1.0, 0.0]]), np.array([1.0, 0.0]))
        self.assertEqual(list(r.covered), [True, False, False])
        self.assertEqual(r.scores[1], 0.0)

    def test_a_bad_crop_match_still_counts_as_covered(self):
        """
        Điểm âm (mảnh cắt ngược nghĩa) vẫn là 'CÓ dữ liệu'. Nếu nhầm sang uncovered thì
        khung có mảnh cắt tệ lại được đối xử như khung không có gì — tức được tha.
        """
        r = crop_scores(2, [ref(0)], np.array([[-1.0, 0.0]]), np.array([1.0, 0.0]))
        self.assertTrue(r.covered[0])
        self.assertLess(r.scores[0], 0.0)

    def test_coverage_fraction_is_reported(self):
        r = crop_scores(4, [ref(0), ref(2)], np.eye(2, 2), np.array([1.0, 0.0]))
        self.assertAlmostEqual(r.coverage, 0.5)

    def test_no_crops_at_all(self):
        r = crop_scores(3, [], np.zeros((0, 2)), np.array([1.0, 0.0]))
        self.assertEqual(r.coverage, 0.0)
        self.assertTrue(np.all(r.scores == 0))


class TestMaxOverCrops(unittest.TestCase):
    def test_frame_takes_the_best_of_its_crops(self):
        """Một vật khớp là đủ để khung đáng lên — trung bình sẽ phạt khung đông vật."""
        v = np.array([[0.2, 0.0], [0.9, 0.0]], dtype=np.float32)
        r = crop_scores(2, [ref(0), ref(0)], v, np.array([1.0, 0.0]))
        self.assertAlmostEqual(float(r.scores[0]), 0.9, places=5)

    def test_order_of_crops_does_not_matter(self):
        v1 = np.array([[0.2, 0.0], [0.9, 0.0]], dtype=np.float32)
        v2 = v1[::-1].copy()
        a = crop_scores(2, [ref(0), ref(0)], v1, np.array([1.0, 0.0]))
        b = crop_scores(2, [ref(0), ref(0)], v2, np.array([1.0, 0.0]))
        np.testing.assert_allclose(a.scores, b.scores)

    def test_negative_only_crops_do_not_stay_at_zero(self):
        """
        Nếu khởi tạo bằng 0 rồi chỉ ghi khi lớn hơn, khung toàn mảnh âm sẽ giữ 0 và
        trông tốt hơn khung có mảnh +0,1. Phải ghi lần đầu vô điều kiện.
        """
        v = np.array([[-0.5, 0.0], [-0.3, 0.0]], dtype=np.float32)
        r = crop_scores(2, [ref(0), ref(0)], v, np.array([1.0, 0.0]))
        self.assertAlmostEqual(float(r.scores[0]), -0.3, places=5)


class TestParallelArrayGuards(unittest.TestCase):
    """Bất biến thứ hai."""

    def test_vector_count_mismatch_raises(self):
        with self.assertRaises(ValueError):
            crop_scores(3, [ref(0), ref(1)], np.zeros((1, 2)), np.zeros(2))

    def test_dim_mismatch_raises(self):
        with self.assertRaises(ValueError):
            crop_scores(3, [ref(0)], np.zeros((1, 4)), np.zeros(2))

    def test_row_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            crop_scores(2, [ref(5)], np.zeros((1, 2)), np.zeros(2))


class TestCollectCrops(unittest.TestCase):
    def test_picks_largest_boxes_first(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_objects(root, "A", 1, [
                ("Car", (0.0, 0.0, 0.2, 0.2)),          # 0,04
                ("Clothing", (0.0, 0.0, 0.9, 0.9)),     # 0,81  ← lớn nhất
                ("Person", (0.0, 0.0, 0.5, 0.5)),       # 0,25
            ])
            got = collect_crops([0], IDS, root)
            self.assertEqual([c.entity for c in got], ["Clothing", "Person"])

    def test_drops_boxes_below_min_area(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_objects(root, "A", 1, [("Car", (0.0, 0.0, 0.05, 0.05))])   # 0,0025
            self.assertEqual(collect_crops([0], IDS, root), [])

    def test_class_filter_when_given(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_objects(root, "A", 1, [("Car", (0, 0, 0.9, 0.9)),
                                         ("Clothing", (0, 0, 0.8, 0.8))])
            got = collect_crops([0], IDS, root, classes={"Clothing"})
            self.assertEqual([c.entity for c in got], ["Clothing"])

    def test_default_is_every_class(self):
        """Đoán sai lớp thì mảnh ĐÚNG không bao giờ được xét — mặc định phải rộng."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_objects(root, "A", 1, [("Xyz", (0, 0, 0.9, 0.9))])
            self.assertEqual(len(collect_crops([0], IDS, root)), 1)

    def test_missing_object_file_is_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(collect_crops([0, 1], IDS, Path(d)), [])

    def test_row_is_carried_through(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_objects(root, "B", 1, [("Car", (0, 0, 0.9, 0.9))])
            got = collect_crops([2], IDS, root)
            self.assertEqual((got[0].row, got[0].video_id, got[0].n), (2, "B", 1))

    def test_respects_max_per_frame(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_objects(root, "A", 1, [("A", (0, 0, .9, .9)), ("B", (0, 0, .8, .8)),
                                         ("C", (0, 0, .7, .7)), ("D", (0, 0, .6, .6))])
            self.assertEqual(len(collect_crops([0], IDS, root)), MAX_PER_FRAME)
            self.assertEqual(len(collect_crops([0], IDS, root, max_per_frame=3)), 3)


class TestPixelBox(unittest.TestCase):
    def test_pad_widens_the_box(self):
        c = ref(0, box=(0.25, 0.25, 0.75, 0.75))
        l, t, r, b = c.pixel_box(1000, 1000)
        self.assertLess(l, 250)
        self.assertGreater(r, 750)

    def test_clamped_at_the_frame_edge(self):
        c = ref(0, box=(0.0, 0.0, 1.0, 1.0))
        l, t, r, b = c.pixel_box(640, 480)
        self.assertEqual((l, t), (0, 0))
        self.assertEqual((r, b), (640, 480))

    def test_pad_is_the_documented_value(self):
        self.assertAlmostEqual(PAD, 0.06)
        self.assertAlmostEqual(MIN_AREA, 0.01)


class TestFusesThroughZNormalization(unittest.TestCase):
    """
    Tầng ⑤ phải đi qua CÙNG `fuse` của ③ — nếu nó tự cộng thì bất biến `E[z|covered]=0`
    không được áp dụng và độ phủ một phần lại thành lợi thế.
    """

    def _fuse(self, sims, rows, n=3):
        from src.retrieval.score_matrix import fuse
        from src.retrieval.sources import SourceScores
        vis = SourceScores("visual", np.full(n, 0.5, np.float32), np.ones(n, bool))
        cr = crop_scores(n, [ref(r) for r in rows],
                         np.array([[s, 0.0] for s in sims], np.float32),
                         np.array([1.0, 0.0]))
        return fuse([vis, cr], {"visual": 1.0, "crop": 1.0})

    def test_a_bad_crop_ranks_BELOW_a_frame_with_no_crop(self):
        """
        Khung 0 có mảnh cắt khớp kém, khung 2 không có mảnh nào. Sau chuẩn hoá z, khung
        2 nhận đúng 0 = kỳ vọng của nhóm phủ, nên khung 0 phải thua nó. Nếu ⑤ tự cộng
        điểm thô thay vì đi qua `fuse`, khung 0 sẽ được +(−1) mà khung 2 được 0 và thứ
        tự vẫn đúng — nhưng khung có mảnh cắt TỐT sẽ thắng bất kể liên quan. Test này
        khoá chiều còn lại.
        """
        out = self._fuse([-1.0, 1.0], [0, 1])
        self.assertLess(out[0], out[2], "mảnh cắt tệ phải THUA khung không có mảnh")

    def test_a_good_crop_beats_frames_without_crops(self):
        out = self._fuse([1.0, -1.0], [0, 1])
        self.assertEqual(int(np.argmax(out)), 0)

    def test_single_covered_frame_contributes_nothing(self):
        """
        Một khung phủ duy nhất ⟹ σ = 0 ⟹ `z_normalize` trả 0 hết. Đó là ĐÚNG: không có
        gì để so thì nguồn không được phép nói gì. Khoá lại để không ai 'sửa' thành
        cộng thẳng điểm thô.
        """
        out = self._fuse([5.0], [0])
        self.assertTrue(np.allclose(out, out[0]), "mọi khung phải bằng nhau")


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestVlmScores(unittest.TestCase):
    """
    Bậc 2. Cùng bất biến với bậc 1: **được chấm** không được thành lợi thế, vì chỉ
    trăm ứng viên đầu mới được VLM nhìn.
    """

    def test_unscored_frames_are_uncovered_not_zero(self):
        r = vlm_scores(5, [0, 2], [0.9, 0.1])
        self.assertEqual(list(r.covered), [True, False, True, False, False])
        self.assertAlmostEqual(r.coverage, 0.4)

    def test_a_low_probability_still_counts_as_covered(self):
        """P=0 nghĩa là 'VLM đã xem và nói KHÔNG', khác hẳn 'chưa xem'."""
        r = vlm_scores(3, [0], [0.0])
        self.assertTrue(r.covered[0])
        self.assertEqual(r.scores[0], 0.0)

    def test_probabilities_are_kept_as_given(self):
        r = vlm_scores(3, [1], [0.73])
        self.assertAlmostEqual(float(r.scores[1]), 0.73, places=6)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            vlm_scores(5, [0, 1], [0.5])

    def test_row_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            vlm_scores(3, [7], [0.5])

    def test_empty_is_allowed(self):
        r = vlm_scores(4, [], [])
        self.assertEqual(r.coverage, 0.0)

    def test_a_scored_but_rejected_frame_ranks_below_an_unscored_one(self):
        """
        Sau chuẩn hoá z, khung VLM nói KHÔNG phải thua khung VLM chưa xem — nếu không
        thì việc lọt vào top-K để được chấm tự nó thành lợi thế.
        """
        from src.retrieval.score_matrix import fuse
        from src.retrieval.sources import SourceScores
        base = SourceScores("visual", np.full(4, 0.5, np.float32), np.ones(4, bool))
        v = vlm_scores(4, [0, 1], [0.0, 1.0])
        out = fuse([base, v], {"visual": 1.0, "vlm": 1.0})
        self.assertLess(out[0], out[3])
        self.assertGreater(out[1], out[3])
