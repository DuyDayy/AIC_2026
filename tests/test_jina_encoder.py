"""
Test cho `src/ingestion/jina_encoder.py` — phần lõi THUẦN TUÝ, không tải model
=============================================================================

Chỗ dễ sai nhất của tầng mã hoá không phải lời gọi model, mà là **thứ tự hai phép
toán Matryoshka**. Cắt trước rồi chuẩn hoá cho ‖v‖ = 1; chuẩn hoá trước rồi cắt cho
‖v‖ < 1 — và vector không đơn vị làm hỏng mọi thứ ở hạ nguồn cùng lúc:

    * tích vô hướng không còn là cosine
    * độ dài vector (vô nghĩa về ngữ nghĩa) lẻn vào điểm số

Không lỗi nào báo. Nên nhóm test đầu tiên khoá đúng thứ tự đó.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.jina_encoder import (
    DEFAULT_DIM,
    FULL_DIM,
    JinaEncoder,
    is_normalized,
    l2_normalize,
    truncate_and_normalize,
)


class TestL2Normalize(unittest.TestCase):
    def test_rows_become_unit_length(self):
        m = np.array([[3.0, 4.0], [1.0, 0.0], [-5.0, 12.0]])
        out = l2_normalize(m)
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)

    def test_direction_preserved(self):
        m = np.array([[3.0, 4.0]])
        np.testing.assert_allclose(l2_normalize(m)[0], [0.6, 0.8], atol=1e-6)

    def test_does_not_mutate_input(self):
        m = np.array([[3.0, 4.0]])
        before = m.copy()
        l2_normalize(m)
        np.testing.assert_array_equal(m, before)

    def test_zero_row_stays_zero_instead_of_nan(self):
        """
        Hàng toàn 0 phải ra vector 0, KHÔNG phải NaN. Vector 0 cho cosine 0 với mọi
        truy vấn — đúng nghĩa "không khớp gì". NaN thì lan ra cả bảng điểm.
        """
        out = l2_normalize(np.zeros((1, 4)))
        self.assertFalse(np.isnan(out).any())
        np.testing.assert_array_equal(out[0], np.zeros(4))

    def test_rejects_non_2d(self):
        with self.assertRaises(ValueError):
            l2_normalize(np.array([1.0, 2.0]))

    def test_empty_matrix_is_allowed(self):
        self.assertEqual(l2_normalize(np.zeros((0, 8))).shape, (0, 8))


class TestTruncateThenNormalize(unittest.TestCase):
    """Bất biến trung tâm của module — xem docstring."""

    def test_truncated_rows_are_unit_length(self):
        rng = np.random.default_rng(0)
        m = rng.normal(size=(16, FULL_DIM))
        out = truncate_and_normalize(m, DEFAULT_DIM)
        self.assertEqual(out.shape, (16, DEFAULT_DIM))
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)

    def test_wrong_order_would_not_be_unit_length(self):
        """
        Chứng minh thứ tự sai KHÔNG tương đương: chuẩn hoá 1024 rồi cắt 512 cho
        ‖v‖ < 1 vì `‖v[:512]‖² = Σ_{i<512} v_i² ≤ 1`, bằng nhau chỉ khi đuôi toàn 0.
        """
        rng = np.random.default_rng(1)
        m = rng.normal(size=(8, FULL_DIM))
        wrong = l2_normalize(m)[:, :DEFAULT_DIM]
        norms = np.linalg.norm(wrong, axis=1)
        self.assertTrue(np.all(norms < 0.99), f"thứ tự sai lại ra chuẩn: {norms[:3]}")
        self.assertTrue(is_normalized(truncate_and_normalize(m, DEFAULT_DIM)))

    def test_truncation_keeps_leading_dims_direction(self):
        """Cắt là BỎ ĐUÔI, không xáo trộn: tỉ lệ giữa các chiều đầu phải giữ nguyên."""
        m = np.array([[3.0, 4.0, 100.0, 200.0]])
        out = truncate_and_normalize(m, 2)
        np.testing.assert_allclose(out[0], [0.6, 0.8], atol=1e-6)

    def test_full_dim_is_just_normalize(self):
        rng = np.random.default_rng(2)
        m = rng.normal(size=(4, 32))
        np.testing.assert_allclose(truncate_and_normalize(m, 32), l2_normalize(m), atol=1e-6)

    def test_rejects_widening(self):
        """Cắt không nới rộng được; đệm 0 sẽ tạo vector trông hợp lệ mà sai nghĩa."""
        with self.assertRaises(ValueError) as ctx:
            truncate_and_normalize(np.zeros((2, 64)), 128)
        self.assertIn("không nới rộng", str(ctx.exception))

    def test_rejects_non_positive_dim(self):
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                truncate_and_normalize(np.zeros((2, 64)), bad)

    def test_supports_documented_matryoshka_sizes(self):
        rng = np.random.default_rng(3)
        m = rng.normal(size=(4, FULL_DIM))
        for dim in (1024, 512, 256, 64):
            out = truncate_and_normalize(m, dim)
            self.assertEqual(out.shape[1], dim)
            self.assertTrue(is_normalized(out), f"{dim} chiều không chuẩn hoá")


class TestIsNormalized(unittest.TestCase):
    def test_true_for_unit_rows(self):
        self.assertTrue(is_normalized(np.array([[1.0, 0.0], [0.0, -1.0]])))

    def test_false_for_unscaled(self):
        self.assertFalse(is_normalized(np.array([[3.0, 4.0]])))

    def test_zero_row_counts_as_valid(self):
        self.assertTrue(is_normalized(np.array([[1.0, 0.0], [0.0, 0.0]])))

    def test_survives_fp16_round_trip(self):
        """
        Chỉ mục lưu fp16 trên đĩa. Nếu ngưỡng quá chặt thì mọi lần đọc lại đều báo
        động giả — phép kiểm phải chịu được đúng vòng fp32 → fp16 → fp32 mà
        `save_flat_index` thực hiện.
        """
        rng = np.random.default_rng(4)
        m = l2_normalize(rng.normal(size=(64, DEFAULT_DIM)))
        self.assertTrue(is_normalized(m.astype(np.float16).astype(np.float32)))

    def test_empty_is_vacuously_true(self):
        self.assertTrue(is_normalized(np.zeros((0, 8))))


class TestEncoderIsLazy(unittest.TestCase):
    """
    Dựng đối tượng KHÔNG được tải 0.9B tham số. Nạp sớm làm mọi lỗi cấu hình chỉ lộ
    ra SAU khi đã trả tiền khởi động — đúng lỗi đã mắc ở đường OWLv2.
    """

    def test_construction_loads_nothing(self):
        enc = JinaEncoder()
        self.assertIsNone(enc._model)
        self.assertEqual(enc.dim, DEFAULT_DIM)

    def test_rejects_dim_above_model_output(self):
        with self.assertRaises(ValueError):
            JinaEncoder(dim=FULL_DIM + 1)

    def test_empty_batch_returns_correct_shape_without_model(self):
        """
        Lô rỗng là chuyện thường khi phân mảnh. Phải trả `(0, dim)` chứ không `(0, 0)`,
        nếu không `np.concatenate` với lô khác sẽ nổ vì lệch chiều — và phải làm được
        điều đó mà KHÔNG chạm tới model.
        """
        enc = JinaEncoder(dim=256)
        self.assertEqual(enc.encode_images([]).shape, (0, 256))
        self.assertEqual(enc.encode_texts([]).shape, (0, 256))
        self.assertIsNone(enc._model)


if __name__ == "__main__":
    unittest.main(verbosity=2)


# ============================================================
# 🔴 GHIM REVISION — lệch bản mô hình là lỗi CÂM tuyệt đối
# ============================================================
#
# `from_pretrained` không ghim sẽ kéo bản MỚI NHẤT trên HuggingFace. Vector KHUNG mã
# hoá bằng `e10d47f5…` + mã `39e6a55a…`; truy vấn mã hoá bằng bản khác thì hai bên
# nằm ở hai không gian hơi lệch — cosine vẫn ra số, thứ hạng vẫn sắp được, và không
# có chỉ số nào báo động.
#
# Đã QUAN SÁT thật: một lần gọi không ghim đã tự tải bản mã remote mới về.
# `scripts/run.py` — đường chạy thi GỐC — thiếu ghim này cho tới khi quét lại toàn bộ.

import ast as _ast
import pathlib as _pl


def test_encoder_ghim_dung_cap_sha_cua_vector_khung():
    from src.ingestion.jina_encoder import JinaEncoder

    e = JinaEncoder()
    assert e.revision == "e10d47f5691d0454a0fb5d13f46f2199b74cb436"
    assert e.code_revision == "39e6a55ae971b59bea6e44675d237c99762e7ee2"


def test_moi_loi_goi_from_pretrained_cua_jina_deu_co_revision():
    """Quét MÃ NGUỒN: mọi chỗ nạp jina-clip phải truyền `revision`.

    Kiểm bằng `ast` nên không phải nạp mô hình, và không bỏ sót chỗ nào vì quên đọc.
    """
    goc = _pl.Path(__file__).resolve().parents[1]
    for p in (goc / "src/ingestion/jina_encoder.py", goc / "scripts/run.py",
              goc / "scripts/rf_prepare.py", goc / "scripts/rf_server.py",
              goc / "scripts/modal_query.py"):
        if not p.is_file():
            continue
        src = p.read_text(encoding="utf-8")
        if "jina-clip" not in src and "MODEL_NAME" not in src:
            continue
        for n in _ast.walk(_ast.parse(src)):
            if not (isinstance(n, _ast.Call)
                    and isinstance(n.func, _ast.Attribute)
                    and n.func.attr == "from_pretrained"):
                continue
            # chỉ xét lời gọi có `trust_remote_code` — đó là dấu của jina-clip;
            # Qwen-VL nạp bằng lớp khác và không dùng cặp sha này.
            kw = {k.arg for k in n.keywords}
            if "trust_remote_code" not in kw:
                continue
            assert "revision" in kw, (
                f"{p.name}:{n.lineno} nạp jina-clip KHÔNG ghim revision — "
                f"truy vấn và khung có thể đến từ hai bản mô hình khác nhau")
