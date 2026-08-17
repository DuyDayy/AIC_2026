"""
Test cho `src/ingestion/vector_index.py` — chỉ mục phẳng
========================================================

Hai bất biến, mỗi cái ứng với một lỗi KHÔNG BAO GIỜ TỰ BÁO:

**1. `emb[i]` phải ứng với `ids[i]`.** Lệch một hàng thì mọi truy vấn trả về khung SAI
mà không có exception nào — điểm vẫn là số thực hợp lệ, thứ hạng vẫn sắp được. Đúng
lớp lỗi vừa xảy ra ở chỉ mục detection (`detection_boxes` dài 100, `regions` dài 16).

**2. Mỗi video phải là một LÁT LIÊN TỤC.** DANTE chạy DP theo từng video trên
`emb[lo:hi]`. Lát không liên tục ⟹ DP nhận khung của video khác, và nó vẫn chạy, vẫn
trả điểm.

Thuần tuý: không tải model, không cần GPU.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.jina_encoder import l2_normalize
from src.ingestion.vector_index import (
    EMB_FILE,
    pending_videos,
    save_video_shard,
    video_is_encoded,
    IDS_FILE,
    RANGES_FILE,
    build_flat_index,
    check_alignment,
    load_flat_index,
    save_flat_index,
)


def unit(*vals) -> np.ndarray:
    return l2_normalize(np.array([vals], dtype=np.float32))[0]


FIDX = {("L21_V001", 1): 8, ("L21_V001", 3): 129,
        ("L21_V002", 2): 61, ("L21_V002", 5): 425}


def sample_rows():
    """Cố tình đưa vào SAI thứ tự — `build_flat_index` phải tự sắp."""
    return [
        ("L21_V002", 5, unit(0.0, 1.0, 0.0)),
        ("L21_V001", 3, unit(0.0, 0.0, 1.0)),
        ("L21_V001", 1, unit(1.0, 0.0, 0.0)),
        ("L21_V002", 2, unit(1.0, 1.0, 0.0)),
    ]


class TestBuild(unittest.TestCase):
    def test_sorts_by_video_then_frame(self):
        idx = build_flat_index(sample_rows(), FIDX)
        self.assertEqual(idx.ids,
                         [("L21_V001", 1), ("L21_V001", 3), ("L21_V002", 2), ("L21_V002", 5)])

    def test_ranges_are_contiguous_slices(self):
        idx = build_flat_index(sample_rows(), FIDX)
        self.assertEqual(idx.ranges, {"L21_V001": (0, 2), "L21_V002": (2, 4)})

    def test_ranges_derived_from_ids_not_trusted_from_caller(self):
        """`ranges` phải là hàm của `ids`, không phải nguồn sự thật thứ hai."""
        idx = build_flat_index(sample_rows(), FIDX)
        for vid, (lo, hi) in idx.ranges.items():
            self.assertTrue(all(v == vid for v, _ in idx.ids[lo:hi]))
            self.assertEqual(hi - lo, sum(1 for v, _ in idx.ids if v == vid))

    def test_rows_follow_their_ids_through_the_sort(self):
        """Bất biến 1: sắp lại hàng thì vector phải đi theo id của nó."""
        idx = build_flat_index(sample_rows(), FIDX)
        np.testing.assert_allclose(idx.emb[idx.ids.index(("L21_V001", 1))],
                                   unit(1.0, 0.0, 0.0), atol=1e-6)
        np.testing.assert_allclose(idx.emb[idx.ids.index(("L21_V002", 5))],
                                   unit(0.0, 1.0, 0.0), atol=1e-6)

    def test_rejects_duplicate_key(self):
        rows = [("L21_V001", 1, unit(1.0, 0.0)), ("L21_V001", 1, unit(0.0, 1.0))]
        with self.assertRaises(ValueError) as c:
            build_flat_index(rows, FIDX)
        self.assertIn("trùng khoá", str(c.exception))

    def test_rejects_mixed_dims(self):
        with self.assertRaises(ValueError):
            build_flat_index([("A", 1, unit(1.0, 0.0)), ("A", 2, unit(1.0, 0.0, 0.0))], {("A",1):1,("A",2):2})

    def test_rejects_unnormalized_vectors(self):
        rows = [("A", 1, np.array([3.0, 4.0], dtype=np.float32))]
        with self.assertRaises(ValueError) as c:
            build_flat_index(rows, FIDX)
        self.assertIn("chuẩn hoá", str(c.exception))

    def test_empty_input(self):
        idx = build_flat_index([], {})
        self.assertEqual(idx.n_frames, 0)
        self.assertEqual(idx.ids, [])


class TestSearch(unittest.TestCase):
    def setUp(self):
        self.idx = build_flat_index(sample_rows(), FIDX)

    def test_exact_match_ranks_first(self):
        hits = self.idx.search(unit(1.0, 0.0, 0.0), top_k=2)
        self.assertEqual(self.idx.ids[hits[0][0]], ("L21_V001", 1))
        self.assertAlmostEqual(hits[0][1], 1.0, places=5)

    def test_scores_descending(self):
        s = [sc for _, sc in self.idx.search(unit(1.0, 1.0, 1.0), top_k=4)]
        self.assertEqual(s, sorted(s, reverse=True))

    def test_dot_product_equals_cosine_because_normalized(self):
        q = unit(1.0, 1.0, 0.0)
        row = self.idx.emb[0]
        cos = float(row @ q) / (np.linalg.norm(row) * np.linalg.norm(q))
        self.assertAlmostEqual(float(row @ q), cos, places=5)

    def test_top_k_larger_than_corpus_is_clamped(self):
        self.assertEqual(len(self.idx.search(unit(1.0, 0.0, 0.0), top_k=99)), 4)

    def test_rejects_wrong_dim(self):
        with self.assertRaises(ValueError):
            self.idx.search(np.ones(7, dtype=np.float32))

    def test_rejects_non_positive_top_k(self):
        with self.assertRaises(ValueError):
            self.idx.search(unit(1.0, 0.0, 0.0), top_k=0)

    def test_search_on_empty_index(self):
        self.assertEqual(build_flat_index([], {}).search(np.zeros(0), top_k=5), [])


class TestVideoSlice(unittest.TestCase):
    """Bất biến 2 — đầu vào của DANTE."""

    def setUp(self):
        self.idx = build_flat_index(sample_rows(), FIDX)

    def test_slice_has_only_that_video(self):
        s = self.idx.video_slice("L21_V001")
        self.assertEqual(s.shape[0], 2)
        np.testing.assert_allclose(s[0], unit(1.0, 0.0, 0.0), atol=1e-6)

    def test_slice_is_a_view_not_a_copy(self):
        self.assertIsNotNone(self.idx.video_slice("L21_V002").base)

    def test_frames_are_ascending(self):
        f = self.idx.frames_of("L21_V002")
        self.assertEqual(f, [2, 5])
        self.assertEqual(f, sorted(f))

    def test_missing_video_raises_instead_of_empty(self):
        """
        Trả mảng rỗng sẽ làm DP chạy tiếp và chấm video đó như "không khớp", giấu mất
        chuyện chỉ mục dựng thiếu.
        """
        with self.assertRaises(KeyError):
            self.idx.video_slice("L99_V999")

    def test_slices_cover_everything_exactly_once(self):
        total = sum(hi - lo for lo, hi in self.idx.ranges.values())
        self.assertEqual(total, self.idx.n_frames)


class TestSaveLoadRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name) / "embed"
        self.idx = build_flat_index(sample_rows(), FIDX)
        save_flat_index(self.d, self.idx)

    def tearDown(self):
        self.tmp.cleanup()

    def test_files_written(self):
        for f in (EMB_FILE, IDS_FILE, RANGES_FILE):
            self.assertTrue((self.d / f).exists(), f"thiếu {f}")

    def test_round_trip_preserves_ids_and_ranges(self):
        back = load_flat_index(self.d)
        self.assertEqual(back.ids, self.idx.ids)
        self.assertEqual(back.ranges, self.idx.ranges)

    def test_stored_as_fp16_loaded_as_fp32(self):
        """fp16 để tiết kiệm đĩa; fp32 trong RAM vì BLAS fp32 nhanh hơn nhiều."""
        self.assertEqual(np.load(self.d / EMB_FILE).dtype, np.float16)
        self.assertEqual(load_flat_index(self.d).emb.dtype, np.float32)

    def test_fp16_round_trip_keeps_search_result(self):
        back = load_flat_index(self.d)
        q = unit(1.0, 0.0, 0.0)
        self.assertEqual(back.ids[back.search(q, 1)[0][0]],
                         self.idx.ids[self.idx.search(q, 1)[0][0]])

    def test_detects_row_count_mismatch(self):
        """Bất biến 1: bớt một hàng embedding phải NỔ, không được im lặng."""
        emb = np.load(self.d / EMB_FILE)
        np.save(self.d / EMB_FILE, emb[:-1])
        with self.assertRaises(ValueError) as c:
            load_flat_index(self.d)
        self.assertIn("lệch", str(c.exception))

    def test_detects_tampered_ranges(self):
        r = json.loads((self.d / RANGES_FILE).read_text(encoding="utf-8"))
        r["L21_V001"] = [0, 3]
        (self.d / RANGES_FILE).write_text(json.dumps(r), encoding="utf-8")
        with self.assertRaises(ValueError) as c:
            load_flat_index(self.d)
        self.assertIn("không khớp", str(c.exception))

    def test_detects_unsorted_ids(self):
        raw = np.load(self.d / IDS_FILE, allow_pickle=True)
        np.save(self.d / IDS_FILE, raw[::-1], allow_pickle=True)
        with self.assertRaises(ValueError) as c:
            load_flat_index(self.d)
        self.assertIn("chưa sắp", str(c.exception))


class TestFrameIdxTravelsWithTheIndex(unittest.TestCase):
    """
    `n` là số thứ tự keyframe; `frame_idx` là số khung thật, và luật thi chấm cái sau.
    [ĐO] 0/173.426 khung có `n == frame_idx`, lệch trung vị 5.267.

    Chỉ mục mang `frame_idx` để không đường ra nào phải TỰ NHỚ tra bảng — quên tra là
    lỗi im lặng, còn dữ liệu đi kèm thì không thể quên.
    """

    def test_answer_returns_frame_idx_not_n(self):
        idx = build_flat_index(sample_rows(), FIDX)
        r = idx.ids.index(("L21_V001", 3))
        self.assertEqual(idx.answer(r), ("L21_V001", 129))
        self.assertNotEqual(idx.answer(r)[1], 3)

    def test_build_refuses_without_frame_idx(self):
        """Không cho dựng chỉ mục thiếu số khung thật — đó là cả điểm của thay đổi này."""
        with self.assertRaises(ValueError) as c:
            build_flat_index(sample_rows())
        self.assertIn("frame_idx", str(c.exception))

    def test_build_refuses_when_a_frame_is_missing(self):
        partial = {k: v for k, v in list(FIDX.items())[:2]}
        with self.assertRaises(ValueError):
            build_flat_index(sample_rows(), partial)

    def test_frame_idx_follows_rows_through_the_sort(self):
        """Sắp lại hàng thì `frame_idx` phải đi theo đúng id của nó."""
        idx = build_flat_index(sample_rows(), FIDX)
        for i, k in enumerate(idx.ids):
            self.assertEqual(int(idx.frame_idx[i]), FIDX[k])

    def test_round_trip_preserves_frame_idx(self):
        import tempfile as _t
        with _t.TemporaryDirectory() as d:
            idx = build_flat_index(sample_rows(), FIDX)
            save_flat_index(d, idx)
            self.assertEqual(list(load_flat_index(d).frame_idx), list(idx.frame_idx))

    def test_old_index_without_frame_idx_is_refused(self):
        """Chỉ mục cũ chỉ có `n` phải bị TỪ CHỐI, không được đọc rồi chạy tiếp."""
        import tempfile as _t
        from src.ingestion.vector_index import FIDX_FILE
        with _t.TemporaryDirectory() as d:
            save_flat_index(d, build_flat_index(sample_rows(), FIDX))
            (Path(d) / FIDX_FILE).unlink()
            with self.assertRaises(ValueError) as c:
                load_flat_index(d)
            self.assertIn("frame_idx", str(c.exception))


class TestAnswerPath(unittest.TestCase):
    """
    Đường ra bài nộp: bên gọi truyền **chỉ số hàng**, không bao giờ gõ số khung.

    Lưới `valid_frames` của writer chỉ bắt 97,4% ca nhầm — 2,6% khung có `n` trùng đúng
    một `frame_idx` hợp lệ. Nên lớp phòng thủ thứ nhất là không tạo cơ hội nhầm.
    """

    def setUp(self):
        self.idx = build_flat_index(sample_rows(), FIDX)

    def test_single_row_gives_kis_shape(self):
        r = self.idx.ids.index(("L21_V001", 3))
        self.assertEqual(self.idx.answer_path([r]), ("L21_V001", (129,)))

    def test_multi_row_gives_trake_shape(self):
        a = self.idx.ids.index(("L21_V001", 1))
        b = self.idx.ids.index(("L21_V001", 3))
        self.assertEqual(self.idx.answer_path([a, b]), ("L21_V001", (8, 129)))

    def test_rejects_rows_from_different_videos(self):
        a = self.idx.ids.index(("L21_V001", 1))
        b = self.idx.ids.index(("L21_V002", 2))
        with self.assertRaises(ValueError):
            self.idx.answer_path([a, b])

    def test_rejects_descending_moments(self):
        a = self.idx.ids.index(("L21_V001", 1))
        b = self.idx.ids.index(("L21_V001", 3))
        with self.assertRaises(ValueError):
            self.idx.answer_path([b, a])

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            self.idx.answer_path([])


class TestSaveVideoShard(unittest.TestCase):
    """
    Đường GHI, chạy trên đĩa THẬT. Hàm này tồn tại vì bug dưới đây từng sống trong
    script Modal và sống sót qua cả benchmark lẫn test — nhánh benchmark `return`
    trước khi tới đoạn ghi, còn test thì tự tạo file bằng tên đúng chuẩn.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name) / "shard"
        rng = np.random.default_rng(3)
        self.mat = l2_normalize(rng.normal(size=(3, 1024)))
        self.ns = [1, 2, 5]

    def tearDown(self):
        self.tmp.cleanup()

    def test_writes_exactly_two_files_with_exact_names(self):
        """
        `np.save` TỰ THÊM `.npy` khi tên chưa có đuôi đó. Tên tạm `x.npy.tmp` vì vậy
        thành `x.npy.tmp.npy` và `replace()` nổ FileNotFoundError — bug thật, đã tái
        hiện. Test này khoá đúng tên file được tạo ra.
        """
        save_video_shard(self.d, "L21_V001", self.ns, self.mat)
        self.assertEqual(sorted(p.name for p in self.d.iterdir()),
                         ["L21_V001.json", "L21_V001.npy"])

    def test_no_temp_file_left_behind(self):
        save_video_shard(self.d, "L21_V001", self.ns, self.mat)
        self.assertEqual(list(self.d.glob("*.tmp*")), [])

    def test_checkpoint_sees_it_as_done(self):
        save_video_shard(self.d, "L21_V001", self.ns, self.mat)
        ok, why = video_is_encoded(self.d, "L21_V001", self.ns, 1024)
        self.assertTrue(ok, why)

    def test_written_vectors_load_back_unchanged_within_fp16(self):
        save_video_shard(self.d, "L21_V001", self.ns, self.mat)
        back = np.load(self.d / "L21_V001.npy").astype(np.float32)
        np.testing.assert_allclose(back, self.mat, atol=1e-3)

    def test_rejects_row_count_mismatch_at_write_time(self):
        """Chặn ngay chỗ ghi; để lệch xuống hạ nguồn là lỗi im lặng."""
        with self.assertRaises(ValueError):
            save_video_shard(self.d, "L21_V001", [1, 2], self.mat)

    def test_rewrite_is_idempotent(self):
        save_video_shard(self.d, "L21_V001", self.ns, self.mat)
        save_video_shard(self.d, "L21_V001", self.ns, self.mat)
        self.assertEqual(len(list(self.d.iterdir())), 2)


class TestMatryoshkaAtLoadTime(unittest.TestCase):
    """
    Chỉ mục lưu ĐỦ chiều; cắt xảy ra lúc ĐỌC. Giá trị của Matryoshka là quyền chọn số
    chiều sau — cắt lúc mã hoá là vứt quyền đó để tiết kiệm 178 MB, và lấy lại thì
    phải mã hoá lại toàn bộ 173.426 khung ($4,79 + 26 phút).
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name) / "embed"
        rng = np.random.default_rng(7)
        rows = [(f"L21_V{v:03d}", n, l2_normalize(rng.normal(size=(1, 64)))[0])
                for v in (1, 2) for n in (1, 2, 3)]
        fx = {(f"L21_V{v:03d}", n): 100 * v + 7 * n for v in (1, 2) for n in (1, 2, 3)}
        save_flat_index(self.d, build_flat_index(rows, fx))

    def tearDown(self):
        self.tmp.cleanup()

    def test_loads_full_dim_by_default(self):
        self.assertEqual(load_flat_index(self.d).dim, 64)

    def test_truncates_at_load(self):
        self.assertEqual(load_flat_index(self.d, dim=16).dim, 16)

    def test_truncated_rows_stay_unit_length(self):
        """Cắt rồi PHẢI chuẩn hoá lại, nếu không tích vô hướng thôi là cosine."""
        emb = load_flat_index(self.d, dim=16).emb
        np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-3)

    def test_disk_still_holds_full_dim_after_a_truncated_load(self):
        """Đọc cắt KHÔNG được làm hỏng file — quyền chọn phải còn nguyên cho lần sau."""
        load_flat_index(self.d, dim=16)
        self.assertEqual(load_flat_index(self.d).dim, 64)

    def test_ids_and_ranges_unaffected_by_truncation(self):
        full, cut = load_flat_index(self.d), load_flat_index(self.d, dim=16)
        self.assertEqual(full.ids, cut.ids)
        self.assertEqual(full.ranges, cut.ranges)

    def test_rejects_widening_at_load(self):
        with self.assertRaises(ValueError):
            load_flat_index(self.d, dim=128)


class TestCheckpointOnRealFiles(unittest.TestCase):
    """
    Checkpoint cho lượt Modal. **File thật, hỏng thật** — không mock, không patch:
    mỗi ca dưới đây ghi ra đĩa đúng kiểu hỏng mà container spot gây ra rồi bắt
    `video_is_encoded` phát hiện.

    "Xong" phải là ĐỌC ĐƯỢC VÀ ĐÚNG. Checkpoint chỉ đếm file sẽ coi file cắt dở là
    xong và bỏ qua vĩnh viễn — để lại lỗ trong chỉ mục mà không gì báo.
    """

    NS = [1, 4, 9]
    DIM = 8

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.write_good("L21_V001")

    def tearDown(self):
        self.tmp.cleanup()

    def write_good(self, vid, ns=None, dim=None):
        ns = self.NS if ns is None else ns
        dim = self.DIM if dim is None else dim
        np.save(self.d / f"{vid}.npy", np.zeros((len(ns), dim), dtype=np.float16))
        (self.d / f"{vid}.json").write_text(json.dumps(ns), encoding="utf-8")

    def test_complete_video_is_skipped(self):
        ok, why = video_is_encoded(self.d, "L21_V001", self.NS, self.DIM)
        self.assertTrue(ok, why)

    def test_missing_json_is_not_done(self):
        """Chết giữa `np.save` và `write_text` — `.npy` đủ, `.json` chưa có."""
        (self.d / "L21_V001.json").unlink()
        ok, why = video_is_encoded(self.d, "L21_V001", self.NS, self.DIM)
        self.assertFalse(ok)
        self.assertIn("thiếu", why)

    def test_truncated_npy_is_not_done(self):
        """Ghi dở THẬT: cắt cụt file .npy giữa chừng."""
        p = self.d / "L21_V001.npy"
        p.write_bytes(p.read_bytes()[:40])
        ok, why = video_is_encoded(self.d, "L21_V001", self.NS, self.DIM)
        self.assertFalse(ok)
        self.assertIn("đọc được", why)

    def test_corrupt_json_is_not_done(self):
        (self.d / "L21_V001.json").write_text("[1, 4,", encoding="utf-8")
        ok, why = video_is_encoded(self.d, "L21_V001", self.NS, self.DIM)
        self.assertFalse(ok)
        self.assertIn("cắt dở", why)

    def test_frame_list_must_match_not_just_length(self):
        """Cùng SỐ LƯỢNG nhưng khác danh sách khung ⟹ đã mã hoá nhầm tập khung."""
        self.write_good("L21_V002", ns=[1, 4, 99])
        ok, why = video_is_encoded(self.d, "L21_V002", self.NS, self.DIM)
        self.assertFalse(ok, "khớp số lượng mà lệch danh sách vẫn phải bị bắt")

    def test_row_count_mismatch_is_not_done(self):
        np.save(self.d / "L21_V001.npy", np.zeros((2, self.DIM), dtype=np.float16))
        ok, why = video_is_encoded(self.d, "L21_V001", self.NS, self.DIM)
        self.assertFalse(ok)
        self.assertIn("≠", why)

    def test_wrong_dim_is_not_done(self):
        """Đổi `dim` giữa chừng mà không xoá file cũ — phải làm lại, không dùng lại."""
        ok, why = video_is_encoded(self.d, "L21_V001", self.NS, dim=512)
        self.assertFalse(ok)
        self.assertIn("số chiều", why)

    def test_reads_header_only_not_whole_array(self):
        """
        Với 873 video, nạp cả mảng chỉ để xem `shape` là đọc thừa hàng trăm MB.
        Mảng lớn phải kiểm được nhanh — mmap không đọc phần dữ liệu.
        """
        big = np.zeros((5000, 512), dtype=np.float16)
        np.save(self.d / "BIG.npy", big)
        (self.d / "BIG.json").write_text(json.dumps(list(range(5000))), encoding="utf-8")
        t = time.perf_counter()
        ok, _ = video_is_encoded(self.d, "BIG", list(range(5000)), 512)
        self.assertTrue(ok)
        self.assertLess(time.perf_counter() - t, 0.05)

    def test_pending_lists_only_unfinished(self):
        self.write_good("L21_V003")
        (self.d / "L21_V003.json").unlink()
        todo, why = pending_videos(self.d, {"L21_V001": self.NS, "L21_V003": self.NS}, self.DIM)
        self.assertEqual(todo, ["L21_V003"])
        self.assertIn("thiếu", why["L21_V003"])


class TestCheckAlignment(unittest.TestCase):
    def test_empty_when_everything_present(self):
        idx = build_flat_index(sample_rows(), FIDX)
        self.assertEqual(check_alignment(idx, idx.ids), [])

    def test_reports_missing_frames(self):
        idx = build_flat_index(sample_rows(), FIDX)
        missing = check_alignment(idx, [("L21_V001", 1), ("L21_V001", 99)])
        self.assertEqual(missing, [("L21_V001", 99)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
