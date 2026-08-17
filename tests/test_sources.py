"""
Test cho `src/retrieval/sources.py` — tầng ② bốn nguồn điểm
===========================================================

Bất biến trung tâm: **điểm 0 và "không có dữ liệu" là hai chuyện khác nhau**, và
`covered` là chỗ duy nhất phân biệt được.

Gộp chúng làm một chính là lỗi đã giết RRF: nguồn phủ một phần nhận điểm dương *bất kể
liên quan*, làm một nhóm video được nâng hạng có hệ thống và mất **−0,2927 FINAL** trên
688 truy vấn. Tầng ③ chuẩn hoá z **chỉ trên tập có dữ liệu**, nên nếu `covered` sai thì
③ sai theo mà không có gì báo.

Thuần tuý: không tải model, không đọc file thật (trừ hai test cuối cố ý chạm dữ liệu).
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.sources import (
    ASR_WINDOW_MS,
    AsrSource,
    SourceScores,
    TextSource,
    VisualSource,
    load_asr_segments,
    load_frame_idx,
    load_ocr_text,
)

IDS = [("A", 1), ("A", 2), ("A", 3), ("B", 1)]


class _FakeIndex:
    def __init__(self, emb):
        self.emb = np.asarray(emb, dtype=np.float32)

    @property
    def dim(self):
        return self.emb.shape[1]


class TestSourceScoresInvariant(unittest.TestCase):
    def test_mismatched_arrays_raise(self):
        """Hai mảng song song lệch nhau = lỗi im lặng, phải nổ tại chỗ dựng."""
        with self.assertRaises(ValueError):
            SourceScores("x", np.zeros(3, np.float32), np.zeros(2, bool))

    def test_coverage_is_the_fraction_with_data(self):
        s = SourceScores("x", np.zeros(4, np.float32), np.array([1, 0, 1, 1], bool))
        self.assertAlmostEqual(s.coverage, 0.75)

    def test_empty_source_has_zero_coverage_not_nan(self):
        s = SourceScores("x", np.zeros(0, np.float32), np.zeros(0, bool))
        self.assertEqual(s.coverage, 0.0)


class TestVisualSource(unittest.TestCase):
    def setUp(self):
        e = np.array([[1, 0], [0, 1], [1, 1], [-1, 0]], dtype=np.float32)
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        self.src = VisualSource(_FakeIndex(e))

    def test_covers_everything(self):
        """Mọi khung đều có vector theo định nghĩa — không có ca 'thiếu dữ liệu'."""
        self.assertEqual(self.src.score(np.array([1.0, 0.0])).coverage, 1.0)

    def test_cosine_ranks_the_aligned_row_first(self):
        s = self.src.score(np.array([1.0, 0.0])).scores
        self.assertEqual(int(np.argmax(s)), 0)

    def test_opposite_direction_is_negative(self):
        """Cosine âm là thông tin thật (ngược nghĩa), không được kẹp về 0."""
        self.assertLess(self.src.score(np.array([1.0, 0.0])).scores[3], 0)

    def test_rejects_wrong_dim(self):
        with self.assertRaises(ValueError):
            self.src.score(np.ones(5))


class TestTextSource(unittest.TestCase):
    TEXT = {("A", 1): "con meo trang nam tren ghe",
            ("A", 3): "con cho den chay ngoai san",
            ("B", 1): "con meo den"}

    def setUp(self):
        self.src = TextSource("ocr", IDS, self.TEXT)

    def test_covered_marks_only_frames_with_text(self):
        c = self.src.score("meo").covered
        self.assertEqual(list(c), [True, False, True, True])

    def test_uncovered_frame_scores_zero(self):
        """Khung không có chữ phải là 0 VÀ `covered=False` — ③ phân biệt bằng cờ."""
        r = self.src.score("meo")
        self.assertEqual(r.scores[1], 0.0)
        self.assertFalse(r.covered[1])

    def test_matching_frames_score_above_zero(self):
        s = self.src.score("meo").scores
        self.assertGreater(s[0], 0)
        self.assertGreater(s[3], 0)
        self.assertEqual(s[2], 0.0, "khung có chữ nhưng KHÔNG khớp cũng là 0")

    def test_zero_from_no_match_and_zero_from_no_data_look_different(self):
        """Chính là bất biến trung tâm của module."""
        r = self.src.score("meo")
        self.assertEqual((r.scores[1], r.covered[1]), (0.0, False))   # không có dữ liệu
        self.assertEqual((r.scores[2], r.covered[2]), (0.0, True))    # có, nhưng không khớp

    def test_empty_query_scores_nothing_but_keeps_coverage(self):
        r = self.src.score("   ")
        self.assertTrue(np.all(r.scores == 0))
        self.assertEqual(list(r.covered), [True, False, True, True])

    def test_source_with_no_text_at_all(self):
        r = TextSource("ocr", IDS, {}).score("meo")
        self.assertEqual(r.coverage, 0.0)
        self.assertTrue(np.all(r.scores == 0))

    def test_scores_align_with_ids_order(self):
        """Ánh xạ ngược `doc_id → hàng` sai thì điểm gắn nhầm khung, im lặng hoàn toàn."""
        r = TextSource("ocr", IDS, {("B", 1): "duy nhat cho khung nay"}).score("duy nhat")
        self.assertGreater(r.scores[3], 0)
        self.assertTrue(np.all(r.scores[:3] == 0))


class TestAsrWindow(unittest.TestCase):
    SEG = {"A": [(0, 3000, "chao mung quy vi"), (18000, 22000, "ket thuc ban tin")]}
    MS = {("A", 1): 1000.0, ("A", 2): 10000.0, ("A", 3): 20000.0, ("B", 1): 0.0}

    def test_frame_inside_a_segment_gets_its_text(self):
        r = AsrSource(IDS, self.MS, self.SEG).score("chao mung")
        self.assertGreater(r.scores[0], 0)

    def test_silent_gap_has_no_data(self):
        """
        Khung ở giây 10 nằm giữa hai đoạn, ngoài cửa sổ ±5s ⟹ KHÔNG có dữ liệu, chứ
        không phải "có dữ liệu và không khớp".
        """
        r = AsrSource(IDS, self.MS, self.SEG).score("chao mung")
        self.assertFalse(r.covered[1])

    def test_video_without_transcript_is_uncovered(self):
        r = AsrSource(IDS, self.MS, self.SEG).score("chao mung")
        self.assertFalse(r.covered[3])

    def test_window_width_changes_coverage(self):
        """
        Cửa sổ là đánh đổi: rộng thì mọi khung mang gần cùng một đoạn text và ASR mất
        khả năng phân biệt TRONG video; hẹp thì khung im lặng trống. Test khoá việc nó
        thật sự là tham số, không phải hằng số ẩn.
        """
        narrow = AsrSource(IDS, self.MS, self.SEG, window_ms=100).score("chao").coverage
        wide = AsrSource(IDS, self.MS, self.SEG, window_ms=30_000).score("chao").coverage
        self.assertLess(narrow, wide)

    def test_default_window_is_the_documented_value(self):
        self.assertEqual(ASR_WINDOW_MS, 5_000)

    def test_overlapping_segments_are_joined(self):
        seg = {"A": [(0, 2000, "phan mot"), (1500, 3500, "phan hai")]}
        r = AsrSource([("A", 1)], {("A", 1): 1800.0}, seg, window_ms=0).score("phan hai")
        self.assertGreater(r.scores[0], 0)


class TestFrameIdxIsNotN(unittest.TestCase):
    """
    `n` là số thứ tự keyframe; `frame_idx` là số khung thật. Luật thi chấm theo
    `frame_idx`.

    [ĐO] **0/173.426** khung có `n == frame_idx`; lệch trung vị **5.267**, lớn nhất
    **68.464**. Nộp nhầm `n` làm sai MỌI câu — và `writer.py` không bắt được, vì nó chỉ
    kiểm `0 ≤ frame_id < số khung video` mà `n` luôn thoả.
    """

    def test_mapping_is_never_the_identity(self):
        m = load_frame_idx()
        if not m:
            self.skipTest("chưa có metadata")
        same = [k for k, v in m.items() if k[1] == v]
        self.assertEqual(same, [], f"{len(same)} khung có n == frame_idx — nhầm sẽ không lộ")

    def test_offset_is_large_enough_to_be_fatal(self):
        """Nếu lệch nhỏ thì nhầm chỉ hơi sai; đo cho thấy nó lệch hàng nghìn khung."""
        m = load_frame_idx()
        if not m:
            self.skipTest("chưa có metadata")
        import statistics
        self.assertGreater(statistics.median(v - k[1] for k, v in m.items()), 1000)

    def test_agrees_with_objects_full_records(self):
        """`objects-full` lưu cả hai trường — hai nguồn phải khớp, nếu không là lệch dữ liệu."""
        m = load_frame_idx()
        root = Path("data/objects-full")
        if not m or not root.is_dir():
            self.skipTest("chưa có dữ liệu")
        import json as _j
        n = 0
        for vd in sorted(p for p in root.iterdir() if p.is_dir())[:5]:
            for f in sorted(vd.glob("*.json"))[:20]:
                d = _j.loads(f.read_text(encoding="utf-8"))
                self.assertEqual(m[(vd.name, d["frame_id"])], d["frame_idx"], f.name)
                n += 1
        self.assertGreater(n, 50)

    def test_gt_eval_file_carries_both(self):
        p = Path("data/eval/textual_kis_gt.json")
        if not p.exists():
            self.skipTest("chưa có bộ eval")
        import json as _j
        for q in _j.loads(p.read_text(encoding="utf-8"))["queries"]:
            self.assertIn("frame_idx", q, q["id"])
            self.assertNotEqual(q["frame_idx"], q["frame_id"], q["id"])


class TestLoadersOnRealData(unittest.TestCase):
    """Hai test cố ý chạm dữ liệu thật — bắt lỗi khuôn file mà dữ liệu giả không bắt."""

    def test_ocr_loader_reads_the_real_file(self):
        p = Path("data/OCR/ocr.jsonl")
        if not p.exists():
            self.skipTest("chưa có data/OCR/ocr.jsonl")
        with open(p, encoding="utf-8") as fh:
            head = [json.loads(next(fh)) for _ in range(50)]
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "ocr.jsonl"
            f.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in head),
                         encoding="utf-8")
            m = load_ocr_text(f)
        self.assertTrue(m, "không đọc được dòng nào — khuôn file đã đổi?")
        k, v = next(iter(m.items()))
        self.assertIsInstance(k[0], str)
        self.assertIsInstance(k[1], int)
        self.assertEqual(v, v.lower(), "text_ascii_folded phải là chữ thường đã bỏ dấu")

    def test_asr_loader_reads_the_real_tree(self):
        if not list(Path("data/ASR").glob("*/results/*.json")):
            self.skipTest("chưa có data/ASR")
        seg = load_asr_segments("data/ASR")
        self.assertGreater(len(seg), 800, "phải có ~873 video")
        v = next(iter(seg.values()))
        a, b, t = v[0]
        self.assertLessEqual(a, b)
        self.assertTrue(t.strip())


if __name__ == "__main__":
    unittest.main(verbosity=2)
