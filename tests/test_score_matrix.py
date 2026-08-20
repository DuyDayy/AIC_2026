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

import numpy as _np
import pytest as _pytest

from src.retrieval.score_matrix import (
    NORMALIZERS,
    ZERO_MEAN_NORMALIZERS,
    fuse,
    hierarchical_rrf,
    rank_normalize,
    rrf_normalize,
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
        for name in ZERO_MEAN_NORMALIZERS:
            f = NORMALIZERS[name]
            s = np.array([1.0, 5.0, 9.0, 100.0, 0.0], dtype=np.float32)
            c = np.array([1, 1, 1, 1, 0], dtype=bool)
            self.assertAlmostEqual(float(f(s, c)[c].mean()), 0.0, places=5, msg=f.__name__)

    def test_uncovered_gets_exactly_zero(self):
        # Cái này thì MỌI normalizer phải giữ, kể cả rrf/minmax.
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
        for mode in ZERO_MEAN_NORMALIZERS:
            out = fuse([src("ocr", s, c)], {"ocr": 1.0}, mode)
            self.assertAlmostEqual(float(out[:500].mean()), float(out[500:].mean()),
                                   places=4, msg=mode)

    def test_rrf_va_minmax_CO_thien_lech_phu_va_dieu_do_duoc_ghi_nhan(self):
        """Ghim mặt trái, để không ai tưởng mọi normalizer đều an toàn với nguồn thưa.

        `rrf` và `minmax` cố ý nằm ngoài `ZERO_MEAN_NORMALIZERS`. Test này ĐỎ nếu ai đó
        thêm chúng vào đó — vì lúc ấy bất biến của module thành lời nói suông.
        """
        rng = np.random.default_rng(0)
        s = np.zeros(1000, dtype=np.float32)
        c = np.zeros(1000, dtype=bool)
        c[:500] = True
        s[:500] = rng.normal(3.0, 1.0, 500)
        for mode in ("rrf", "minmax"):
            self.assertNotIn(mode, ZERO_MEAN_NORMALIZERS, mode)
            out = fuse([src("ocr", s, c)], {"ocr": 1.0}, mode)
            self.assertGreater(float(out[:500].mean()), float(out[500:].mean()), mode)


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


# ---------------------------------------------------------------------------
# hierarchical_rrf — gộp expansion trong modality trước, rồi mới giữa modality
#
# Bất biến chịu lực là "Σ beta = 1 trong từng modality". Mất nó thì modality có
# nhiều expansion hơn tự có nhiều quyền vote hơn, và [ĐO] cấu hình phẳng tụt từ
# MRR 0,5281 xuống 0,4427 trên bench_kis đúng vì lý do đó.
# ---------------------------------------------------------------------------


def _ss(name, vals):
    a = _np.asarray(vals, dtype=_np.float32)
    return SourceScores(name, a, a > 0)


def test_hierarchical_bang_rrf_thuong_khi_moi_modality_mot_run():
    """Một run mỗi modality thì phân cấp phải suy biến về RRF đều thông thường."""
    v, o = _ss("v", [3.0, 2.0, 1.0]), _ss("o", [1.0, 3.0, 2.0])
    got = hierarchical_rrf({"v": [v], "o": [o]})
    want = (rrf_normalize(v.scores, v.covered) + rrf_normalize(o.scores, o.covered)) / 2
    _np.testing.assert_allclose(got, want, rtol=1e-6)


def test_them_expansion_KHONG_tang_quyen_vote_cua_modality():
    """Đây là toàn bộ lý do có tầng beta.

    Nhân đôi một run của Visual (hai bản y hệt) KHÔNG được làm Visual nặng hơn. Ở RRF
    phẳng thì có; ở phân cấp thì không.
    """
    v, o = _ss("v", [3.0, 2.0, 1.0]), _ss("o", [1.0, 3.0, 2.0])
    one = hierarchical_rrf({"v": [v], "o": [o]})
    dup = hierarchical_rrf({"v": [v, v], "o": [o]})
    _np.testing.assert_allclose(one, dup, rtol=1e-6)
    # còn cộng phẳng thì lệch — đối chứng cho thấy phép kiểm trên không tầm thường
    flat = (2 * rrf_normalize(v.scores, v.covered) + rrf_normalize(o.scores, o.covered)) / 3
    assert not _np.allclose(one, flat)


def test_beta_duoc_chuan_hoa_du_ben_goi_khong_chuan_hoa():
    v, o = _ss("v", [3.0, 2.0, 1.0]), _ss("o", [1.0, 3.0, 2.0])
    a = hierarchical_rrf({"v": [v, v], "o": [o]}, beta={"v": [3.0, 1.0]})
    b = hierarchical_rrf({"v": [v, v], "o": [o]}, beta={"v": [0.75, 0.25]})
    _np.testing.assert_allclose(a, b, rtol=1e-6)


def test_alpha_lech_thi_ket_qua_lech_theo():
    v, o = _ss("v", [3.0, 2.0, 1.0]), _ss("o", [1.0, 3.0, 2.0])
    eq = hierarchical_rrf({"v": [v], "o": [o]})
    hv = hierarchical_rrf({"v": [v], "o": [o]}, alpha={"v": 0.9, "o": 0.1})
    assert int(_np.argmax(eq)) != int(_np.argmax(hv)) or hv[0] > eq[0]


def test_dau_vao_sai_thi_NO_chu_khong_mac_dinh_0():
    v = _ss("v", [3.0, 2.0, 1.0])
    with _pytest.raises(ValueError, match="ít nhất một modality"):
        hierarchical_rrf({})
    with _pytest.raises(ValueError, match="không có run nào"):
        hierarchical_rrf({"v": []})
    with _pytest.raises(ValueError, match="thiếu alpha"):
        hierarchical_rrf({"v": [v], "o": [v]}, alpha={"v": 1.0})
    with _pytest.raises(ValueError, match="beta"):
        hierarchical_rrf({"v": [v, v]}, beta={"v": [1.0]})
    with _pytest.raises(ValueError, match="lệch số khung"):
        hierarchical_rrf({"v": [v, _ss("v2", [1.0, 2.0])]})


# ============================================================
# CHỐT CHẶN ĐƯỜNG CHẠY THI — đọc `scripts/run.py` ở mức cú pháp
# ============================================================
#
# Hai điều dưới đây là QUYẾT ĐỊNH KIẾN TRÚC, không phải tham số:
#   · ③ và ⑤ hợp bằng HẠNG, không có chuẩn hoá z ở bất kỳ tầng nào
#   · β chia ĐỀU — mỗi bản mở rộng đóng góp ngang bản gốc
#
# Cả hai từng bị đặt ngược trong chính kho này (⑤ hợp bằng `fuse(..., "z")`; β từng là
# `(0,5 · 0,25 · 0,25)` ưu ái bản gốc, lấy từ hạt giống playbook chứ không từ phép đo).
# Không có chốt thì lần sau chúng quay lại trong im lặng — điểm vẫn ra số, thứ hạng vẫn
# sắp được. Đọc bằng `ast` để KHÔNG phải import `run.py` (nó nạp torch).

import ast as _ast
import pathlib as _pl

_RUN_PY = _pl.Path(__file__).resolve().parents[1] / "scripts" / "run.py"
_RUN_AST = _ast.parse(_RUN_PY.read_text(encoding="utf-8"))


def test_duong_chay_thi_KHONG_con_chuan_hoa_z_o_bat_ky_tang_nao():
    """Mọi lời gọi `fuse`/`fuse_matrix` đều đã thay bằng `hierarchical_rrf`."""
    goi = [
        n.func.id
        for n in _ast.walk(_RUN_AST)
        if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Name)
    ]
    assert "fuse" not in goi and "fuse_matrix" not in goi, (
        "run.py gọi lại `fuse` — ③/⑤ phải hợp bằng `hierarchical_rrf`"
    )
    assert "hierarchical_rrf" in goi
    # và không còn chuỗi "z" nào nằm ở vị trí tham số chuẩn hoá
    chuoi_z = [
        n
        for n in _ast.walk(_RUN_AST)
        if isinstance(n, _ast.Constant) and n.value == "z"
    ]
    assert not chuoi_z, f'còn hằng chuỗi "z" ở dòng {[n.lineno for n in chuoi_z]}'


def test_beta_chia_DEU_va_tong_bang_1():
    """`_beta_for(n)` = (1/n,)·n với mọi n — không ưu ái bản gốc."""
    ns = {}
    for node in _RUN_AST.body:
        if isinstance(node, _ast.FunctionDef) and node.name == "_beta_for":
            exec(compile(_ast.Module([node], []), "<_beta_for>", "exec"), ns)
    assert "_beta_for" in ns, "run.py không còn `_beta_for`"
    beta_for = ns["_beta_for"]
    for n in range(1, 8):
        b = beta_for(n)
        assert len(b) == n
        assert len(set(b)) == 1, f"β lệch ở n={n}: {b}"
        assert abs(sum(b) - 1.0) < 1e-9
    # ba nguồn (gốc + 2 bản mở rộng) ⟹ 1/3, KHÔNG phải 50-50
    assert beta_for(3) == (1 / 3, 1 / 3, 1 / 3)


def test_pool_cap_khong_duoc_cat_ro_that():
    """`POOL_CAP` phải ≥ 7 run × `POOL_PER_SOURCE`, nếu không nó cắt mọi truy vấn."""
    hang = {
        t.id: node.value.value
        for node in _RUN_AST.body
        if isinstance(node, _ast.Assign) and isinstance(node.value, _ast.Constant)
        for t in node.targets
        if isinstance(t, _ast.Name)
    }
    assert hang["POOL_CAP"] >= 7 * hang["POOL_PER_SOURCE"], (
        f"POOL_CAP={hang['POOL_CAP']} cắt rổ thật (7 run × {hang['POOL_PER_SOURCE']})"
    )
