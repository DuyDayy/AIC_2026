"""
Kiểm chứng BM25 (`src/retrieval/bm25.py`).

Trọng tâm: các tính chất khiến BM25 dùng được làm THƯỚC ĐO — tất định, IDF không
âm, bão hoà tần suất, chuẩn hoá độ dài. Một chỉ mục không tất định làm mọi phép
A/B phía sau vô nghĩa, nên tính tất định được test riêng.
"""

from __future__ import annotations

import math
import unittest

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.bm25 import Bm25Index, looks_unfolded, tokenize


class TestTokenize(unittest.TestCase):
    def test_giu_chu_va_so(self):
        self.assertEqual(tokenize("ban tin 60 giay hom nay"),
                         ["ban", "tin", "60", "giay", "hom", "nay"])

    def test_bo_dau_cau_va_ha_chu(self):
        self.assertEqual(tokenize("HTV-News, 06:30:11!"),
                         ["htv", "news", "06", "30", "11"])

    def test_rong(self):
        self.assertEqual(tokenize(""), [])
        self.assertEqual(tokenize("   ,,,   "), [])


class TestIdf(unittest.TestCase):
    def test_idf_khong_bao_gio_am(self):
        """
        Biến thể `ln(1 + …)` phải chặn dưới ở 0. Dạng gốc ra ÂM khi từ xuất hiện
        ở hơn nửa số tài liệu — khi ấy một từ phổ biến sẽ KÉO TỤT điểm tài liệu
        chứa nó, đảo ngược xếp hạng một cách âm thầm.
        """
        # "tin" có ở 4/5 tài liệu — đúng vùng mà dạng gốc ra âm.
        docs = [(f"d{i}", "tin tuc" if i < 4 else "khac") for i in range(5)]
        idx = Bm25Index.build(docs)
        self.assertGreaterEqual(idx.idf("tin"), 0.0)

    def test_tu_hiem_co_idf_cao_hon_tu_pho_bien(self):
        docs = [("d0", "tin tuc labubu")] + [(f"d{i}", "tin tuc") for i in range(1, 20)]
        idx = Bm25Index.build(docs)
        self.assertGreater(idx.idf("labubu"), idx.idf("tin"))

    def test_tu_khong_co_trong_kho(self):
        idx = Bm25Index.build([("d0", "tin tuc")])
        self.assertEqual(idx.idf("khongtontai"), 0.0)

    def test_cong_thuc_idf_dung_nguyen_van(self):
        docs = [(f"d{i}", "alpha" if i < 3 else "beta") for i in range(10)]
        idx = Bm25Index.build(docs)
        n, df = 10, 3
        self.assertAlmostEqual(
            idx.idf("alpha"), math.log(1.0 + (n - df + 0.5) / (df + 0.5)), places=12
        )


class TestRanking(unittest.TestCase):
    def test_tai_lieu_khop_dung_len_dau(self):
        idx = Bm25Index.build([
            ("a", "chay rung tai california"),
            ("b", "gia ca thi truong nong san"),
            ("c", "dam chay rung lan rong"),
        ])
        hits = idx.search("chay rung")
        self.assertEqual(hits[0].doc_id, "a")
        self.assertIn("c", [h.doc_id for h in hits])
        self.assertNotIn("b", [h.doc_id for h in hits])

    def test_chi_tra_diem_duong(self):
        idx = Bm25Index.build([("a", "alpha"), ("b", "beta")])
        hits = idx.search("alpha")
        self.assertEqual([h.doc_id for h in hits], ["a"])

    def test_truy_van_khong_khop_gi(self):
        idx = Bm25Index.build([("a", "alpha")])
        self.assertEqual(idx.search("zeta"), [])

    def test_top_k_cat_dung(self):
        idx = Bm25Index.build([(f"d{i}", "tin tuc") for i in range(10)])
        self.assertEqual(len(idx.search("tin", top_k=3)), 3)

    def test_bao_hoa_tan_suat(self):
        """
        `k₁` làm tần suất bão hoà: lặp từ 100 lần KHÔNG được cho điểm gấp ~100
        lần so với 1 lần. Không có tính chất này thì spam từ khoá thắng mọi thứ.
        """
        idx = Bm25Index.build([
            ("mot", "alpha beta"),
            ("nhieu", " ".join(["alpha"] * 100) + " beta"),
        ])
        s = {h.doc_id: h.score for h in idx.search("alpha")}
        self.assertGreater(s["nhieu"], s["mot"])
        self.assertLess(s["nhieu"], 10 * s["mot"])

    def test_chuan_hoa_do_dai(self):
        """Cùng một lần xuất hiện, tài liệu NGẮN được điểm cao hơn tài liệu dài."""
        idx = Bm25Index.build([
            ("ngan", "labubu"),
            ("dai", "labubu " + " ".join(f"w{i}" for i in range(200))),
        ])
        s = {h.doc_id: h.score for h in idx.search("labubu")}
        self.assertGreater(s["ngan"], s["dai"])


class TestDeterminism(unittest.TestCase):
    """Không tất định ⟹ mọi phép A/B phía sau chỉ đang đo nhiễu."""

    def test_pha_hoa_theo_doc_id(self):
        idx = Bm25Index.build([("z", "alpha"), ("a", "alpha"), ("m", "alpha")])
        hits = idx.search("alpha")
        self.assertEqual([h.doc_id for h in hits], ["a", "m", "z"])

    def test_hai_lan_tim_cho_ket_qua_giong_het(self):
        idx = Bm25Index.build([(f"d{i}", f"tin tuc so {i}") for i in range(50)])
        a = [(h.doc_id, h.score) for h in idx.search("tin so 7")]
        b = [(h.doc_id, h.score) for h in idx.search("tin so 7")]
        self.assertEqual(a, b)


class TestUnfoldedGuard(unittest.TestCase):
    """
    Rào chắn cho lỗi ĐÃ XẢY RA THẬT: truy vấn còn dấu, chỉ mục đã bỏ dấu.

    Regex `[a-z0-9]+` loại sạch ký tự có dấu nên "đốt" → "t". Không ngoại lệ nào
    được ném, chỉ điểm tụt — trung vị hạng của video đúng rơi xuống 107.5/865.
    """

    def test_phat_hien_van_ban_con_dau(self):
        self.assertTrue(looks_unfolded("đốt cháy rừng"))
        self.assertTrue(looks_unfolded("Hồ Chí Minh"))
        self.assertTrue(looks_unfolded("ĐỐT"))          # chữ hoa cũng phải bắt

    def test_khong_bao_dong_gia_tren_van_ban_da_bo_dau(self):
        self.assertFalse(looks_unfolded("dot chay rung"))
        self.assertFalse(looks_unfolded("ho chi minh"))
        self.assertFalse(looks_unfolded("HTV News 06:30:11"))
        self.assertFalse(looks_unfolded(""))

    def test_search_canh_bao_khi_truy_van_con_dau(self):
        idx = Bm25Index.build([("a", "dot chay rung")])
        with self.assertLogs("src.retrieval.bm25", level="WARNING") as cm:
            idx.search("đốt cháy rừng")
        self.assertIn("CÒN DẤU", "\n".join(cm.output))

    def test_search_khong_canh_bao_khi_da_bo_dau(self):
        idx = Bm25Index.build([("a", "dot chay rung")])
        with self.assertNoLogs("src.retrieval.bm25", level="WARNING"):
            idx.search("dot chay")

    def test_khong_tu_sua_ngam(self):
        """
        Cảnh báo, KHÔNG tự bỏ dấu: tự sửa sẽ tạo cài đặt bỏ dấu thứ hai. Truy vấn
        còn dấu vẫn phải cho kết quả rác — đó là bằng chứng nó không âm thầm
        được "cứu".
        """
        idx = Bm25Index.build([("a", "dot chay rung")])
        with self.assertLogs("src.retrieval.bm25", level="WARNING"):
            co_dau = idx.search("đốt cháy rừng")
        khong_dau = idx.search("dot chay rung")
        self.assertNotEqual(
            [(h.doc_id, round(h.score, 9)) for h in co_dau],
            [(h.doc_id, round(h.score, 9)) for h in khong_dau],
        )


class TestInvariants(unittest.TestCase):
    def test_doc_id_trung_bi_tu_choi(self):
        idx = Bm25Index()
        idx.add("a", ["x"])
        with self.assertRaises(ValueError):
            idx.add("a", ["y"])

    def test_khong_them_sau_khi_freeze(self):
        idx = Bm25Index.build([("a", "x")])
        with self.assertRaises(RuntimeError):
            idx.add("b", ["y"])

    def test_tham_so_khong_hop_le(self):
        with self.assertRaises(ValueError):
            Bm25Index(k1=-1.0)
        with self.assertRaises(ValueError):
            Bm25Index(b=1.5)

    def test_kho_rong(self):
        idx = Bm25Index.build([])
        self.assertEqual(len(idx), 0)
        self.assertEqual(idx.search("bat ky"), [])

    def test_freeze_hai_lan_khong_sao(self):
        idx = Bm25Index.build([("a", "x")])
        idx.freeze()
        self.assertEqual(len(idx), 1)


if __name__ == "__main__":
    unittest.main()
