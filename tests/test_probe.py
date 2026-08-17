"""
Test cho `src/retrieval/probe.py` — tầng ① probe hoá
====================================================

Tầng ① làm **đúng hai việc**: tách mốc, và rút chuỗi trong ngoặc kép. Đa biến thể và
rút gọn truy vấn đều đã bị phép đo bác bỏ — số đo giữ trong docstring của module.

Bất biến quan trọng nhất: **tách mốc phải BẢO THỦ.** `N = len(probes)` đi thẳng vào
DP ở tầng ④, nên tách thừa một mốc làm DP đi tìm một khoảnh khắc không tồn tại — và
nó vẫn chạy, vẫn trả điểm, không có gì báo.

Các ca dưới đây dùng **nguyên văn đề thi thật**, không phải câu tự nghĩ, vì chính đề
thật mới chứa cái bẫy: `lần lượt` và `đầu tiên` trông như từ nối trình tự nhưng đo
được chúng **chưa bao giờ** đứng ở ranh giới (0/2 và 0/5 trên 30 đề).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.probe import (
    Probe,
    build_probes,
    extract_quoted,
    sentences,
    split_events,
)

# Nguyên văn từ 30 đề thi thật.
Q12 = ("Trong một lễ hội trình diễn, các người mẫu lần lượt bước trên sàn. "
       "Mở màn là những bộ cổ phục Nhật Bình với đủ màu sắc.")
Q10 = ("Cảnh láp ráp trong một xưởng. E1: Một cánh tay rô bot đang lắp một cái khung "
       "cho một chiếc xe. E2: Một công nhân đang lắp gì đó rồi quay một cái nút.")
Q11 = ("Cảnh quay một người phụ nữ lớn tuổi đang đọc sách, tay đeo một xâu hạt. "
       "Sau đó là cảnh quay người phụ nữ này đang mở một quyển sách.")


class TestSplitEventsIsConservative(unittest.TestCase):
    """Tách thừa nguy hiểm hơn tách thiếu — xem docstring module."""

    def test_single_scene_stays_one_probe(self):
        """
        [ĐO] `lần lượt` xuất hiện 2 lần trên 30 đề, **cả hai đều giữa câu** và cả hai
        đều mô tả MỘT cảnh liền mạch. Tách ở đó là chẻ một cảnh thành hai mốc rời.
        """
        self.assertEqual(len(split_events(Q12)), 1)

    def test_enumeration_is_not_a_sequence(self):
        """*"2 mức giá … lần lượt là 9…"* — liệt kê, không phải chuỗi sự kiện."""
        q = "Hình ảnh những chiếc bánh có 2 mức giá được viết lần lượt là 9 và 12 nghìn."
        self.assertEqual(len(split_events(q)), 1)

    def test_dau_tien_is_not_a_boundary(self):
        """`đầu tiên` chỉ mốc THỨ NHẤT, không phải ranh giới — tách ra đoạn rỗng."""
        q = "Cảnh quay đầu tiên là một con thuyền trên sông rộng."
        self.assertEqual(len(split_events(q)), 1)

    def test_explicit_markers_win(self):
        """`E1:`/`E2:` là dấu hiệu đáng tin nhất vì người ra đề tự đánh số."""
        segs = split_events(Q10)
        self.assertEqual(len(segs), 2)
        self.assertTrue(segs[0].startswith("Một cánh tay rô bot"))
        self.assertTrue(segs[1].startswith("Một công nhân"))

    def test_sentence_initial_connective_splits(self):
        segs = split_events(Q11)
        self.assertEqual(len(segs), 2)
        self.assertTrue(segs[1].startswith("Sau đó"))

    def test_mid_sentence_connective_does_not_split(self):
        q = "Người đàn ông bước vào rồi sau đó ngồi xuống ghế."
        self.assertEqual(len(split_events(q)), 1)

    def test_empty_query(self):
        self.assertEqual(split_events(""), [])
        self.assertEqual(split_events("   "), [])

    def test_no_empty_segment_is_ever_produced(self):
        for q in (Q10, Q11, Q12, "Sau đó là cảnh biển.", "E1: a. E2: b."):
            self.assertTrue(all(s.strip() for s in split_events(q)), q)

    def test_single_marker_does_not_split(self):
        """Một `E1:` lẻ không phải chuỗi — cần ít nhất hai mốc mới là TRAKE."""
        self.assertEqual(len(split_events("Cảnh xưởng. E1: cánh tay robot lắp khung.")), 1)


class TestQuoted(unittest.TestCase):
    def test_straight_and_curly_quotes(self):
        self.assertEqual(extract_quoted('có chữ "Nà Ní" trên bảng'), ("Nà Ní",))
        self.assertEqual(extract_quoted("dòng chữ “Happy New Year” trên kính"),
                         ("Happy New Year",))

    def test_multiple_and_deduplicated(self):
        q = 'chữ "A B" rồi chữ "C D" rồi lại "A B"'
        self.assertEqual(extract_quoted(q), ("A B", "C D"))

    def test_none_when_no_quotes(self):
        self.assertEqual(extract_quoted("không có ngoặc kép nào"), ())

    def test_ignores_single_character_quotes(self):
        """Ngoặc quanh một ký tự thường là dấu nháy trong văn bản, không phải trích dẫn."""
        self.assertEqual(extract_quoted("chữ \"A\" nhỏ"), ())


class TestBuildProbes(unittest.TestCase):
    def test_kis_gives_exactly_one_probe(self):
        p = build_probes(Q12)
        self.assertEqual(len(p), 1)
        self.assertEqual(p[0].index, 0)

    def test_trake_gives_n_probes_indexed_in_order(self):
        p = build_probes(Q10)
        self.assertEqual([x.index for x in p], [0, 1])

    def test_every_probe_carries_its_own_text(self):
        for q in (Q10, Q11, Q12):
            for pr in build_probes(q):
                self.assertTrue(pr.text.strip())

    def test_quoted_attaches_to_the_probe_containing_it(self):
        q = 'Cảnh A bình thường. Sau đó cảnh có chữ "Nà Ní" trên bảng tên.'
        p = build_probes(q)
        self.assertEqual(len(p), 2)
        self.assertEqual(p[0].quoted, ())
        self.assertEqual(p[1].quoted, ("Nà Ní",))

    def test_probe_is_immutable(self):
        with self.assertRaises(Exception):
            build_probes(Q12)[0].text = "khác"   # type: ignore[misc]

    def test_empty_query_gives_no_probes(self):
        self.assertEqual(build_probes("  "), [])


class TestOnRealQueryFile(unittest.TestCase):
    """Chạy trên chính bộ eval — bắt hồi quy mà câu tự nghĩ không bắt được."""

    def test_every_gt_query_gives_at_least_one_probe(self):
        p = Path("data/eval/textual_kis_gt.json")
        if not p.exists():
            self.skipTest("chưa có bộ eval")
        import json
        for q in json.loads(p.read_text(encoding="utf-8"))["queries"]:
            probes = build_probes(q["query"])
            self.assertGreaterEqual(len(probes), 1, q["id"])
            self.assertTrue(all(pr.text.strip() for pr in probes), q["id"])

    def test_textual_kis_queries_are_single_probe(self):
        """
        Bộ eval là Textual KIS — mỗi câu tả MỘT khung. Nếu tách ra nhiều mốc thì tầng
        ④ sẽ đi tìm nhiều khoảnh khắc cho một câu chỉ có một, tức tách sai.
        """
        p = Path("data/eval/textual_kis_gt.json")
        if not p.exists():
            self.skipTest("chưa có bộ eval")
        import json
        multi = [q["id"] for q in json.loads(p.read_text(encoding="utf-8"))["queries"]
                 if len(build_probes(q["query"])) > 1]
        self.assertEqual(multi, [], f"tách thừa mốc ở {multi}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
