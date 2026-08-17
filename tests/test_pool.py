"""Rổ ứng viên hợp từ top riêng của từng nguồn."""

import numpy as np
import pytest

from src.retrieval.pool import PoolResult, pool_mask, union_pool
from src.retrieval.sources import SourceScores


def src(name, scores, covered=None):
    s = np.asarray(scores, dtype=np.float32)
    c = np.ones(s.shape, dtype=bool) if covered is None else np.asarray(covered, bool)
    return SourceScores(name, s, c)


# ── điều module này SINH RA ĐỂ LÀM: nguồn yếu vẫn có suất ────────────────────

def test_nguon_yeu_van_de_cu_duoc_du_diem_thap():
    """Cả lý do tồn tại của module: một khung chỉ có bằng chứng OCR vẫn vào rổ."""
    vis = src("visual", [9.0, 8.0, 7.0, 0.0])
    ocr = src("ocr", [0.0, 0.0, 0.0, 5.0])
    p = union_pool([vis, ocr], per_source=1, weights={"visual": 1.0, "ocr": 0.13})
    assert set(p.rows.tolist()) == {0, 3}
    assert p.provenance[3] == ("ocr",)
    # khung 3 thua xa theo điểm hợp, vẫn vào rổ nhờ suất riêng của OCR
    assert 3 in p.provenance


def test_lai_lich_ghi_moi_nguon_da_de_cu():
    a = src("visual", [5.0, 1.0])
    b = src("ocr", [4.0, 1.0])
    p = union_pool([a, b], per_source=1, weights={"visual": 1.0, "ocr": 1.0})
    assert p.provenance[0] == ("visual", "ocr")
    assert p.per_source_n == {"visual": 1, "ocr": 1}


def test_khung_khong_phu_khong_bao_gio_duoc_de_cu():
    """Nguồn thưa đề cử khung điểm 0 chính là lỗi đã giết RRF."""
    thua = src("ocr", [0.0, 0.0, 3.0], covered=[False, False, True])
    p = union_pool([thua], per_source=3, weights={"ocr": 1.0})
    assert p.rows.tolist() == [2]
    assert p.per_source_n["ocr"] == 1


def test_nguon_phu_rong_khong_bi_gioi_han_boi_per_source():
    day = src("visual", [1.0, 2.0, 3.0, 4.0])
    p = union_pool([day], per_source=2, weights={"visual": 1.0})
    assert sorted(p.rows.tolist()) == [2, 3]


# ── tính tất định và thứ tự ──────────────────────────────────────────────────

def test_rows_sap_theo_base_giam_dan():
    a = src("visual", [1.0, 5.0, 3.0])
    p = union_pool([a], per_source=3, base=np.array([1.0, 5.0, 3.0], np.float32))
    assert p.rows.tolist() == [1, 2, 0]


def test_base_tu_tinh_bang_weights_khi_khong_truyen():
    a = src("visual", [1.0, 0.0])
    b = src("ocr", [0.0, 1.0])
    p1 = union_pool([a, b], per_source=2, weights={"visual": 1.0, "ocr": 0.0})
    p2 = union_pool([a, b], per_source=2, weights={"visual": 1.0, "ocr": 0.0})
    assert p1.rows.tolist() == p2.rows.tolist()      # tất định giữa hai lần gọi


def test_cap_cat_sau_khi_moi_nguon_da_co_suat():
    vis = src("visual", [9.0, 8.0, 7.0, 6.0, 0.0])
    ocr = src("ocr", [0.0, 0.0, 0.0, 0.0, 5.0])
    p = union_pool([vis, ocr], per_source=4, weights={"visual": 1.0, "ocr": 0.13}, cap=2)
    assert len(p) == 2
    assert set(p.provenance) == set(p.rows.tolist())   # lai lịch cắt theo


# ── bỏ sót phải NỔ, không im lặng ────────────────────────────────────────────

def test_per_source_be_hon_1_la_loi():
    with pytest.raises(ValueError, match="per_source"):
        union_pool([src("a", [1.0])], per_source=0, weights={"a": 1.0})


def test_khong_co_nguon_la_loi():
    with pytest.raises(ValueError, match="không có nguồn"):
        union_pool([], per_source=1, weights={})


def test_nguon_lech_so_khung_la_loi():
    with pytest.raises(ValueError, match="lệch số khung"):
        union_pool([src("a", [1.0, 2.0]), src("b", [1.0])], per_source=1,
                   weights={"a": 1.0, "b": 1.0})


def test_thieu_ca_base_lan_weights_la_loi():
    with pytest.raises(ValueError, match="base"):
        union_pool([src("a", [1.0])], per_source=1)


def test_base_lech_so_khung_la_loi():
    with pytest.raises(ValueError, match="khung"):
        union_pool([src("a", [1.0, 2.0])], per_source=1,
                   base=np.array([1.0], np.float32))


# ── mặt nạ ───────────────────────────────────────────────────────────────────

def test_pool_mask_dung_vi_tri():
    m = pool_mask(np.array([0, 3]), 5)
    assert m.tolist() == [True, False, False, True, False]


def test_repr_doc_duoc():
    p = union_pool([src("visual", [1.0, 2.0])], per_source=1, weights={"visual": 1.0})
    assert "PoolResult" in repr(p) and "visual" in repr(p)
    assert isinstance(p, PoolResult)


# ── tính chất: rổ hợp LUÔN chứa top của từng nguồn ───────────────────────────

def test_ro_hop_luon_chua_top1_cua_moi_nguon():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = int(rng.integers(5, 40))
        ss = [src(f"s{i}", rng.normal(size=n).astype(np.float32)) for i in range(4)]
        p = union_pool(ss, per_source=3, weights={f"s{i}": 1.0 for i in range(4)})
        got = set(p.rows.tolist())
        for s in ss:
            assert int(np.argmax(s.scores)) in got


# ── hạn ngạch mỗi video: chặn nguồn PHẲNG làm ngập rổ ────────────────────────

RANGES = {"A": (0, 4), "B": (4, 8)}


def test_nguon_phang_khong_con_chiem_het_ro():
    """OCR khớp logo kênh ⟹ điểm phẳng cả video A. Không hạn ngạch thì nó lấy sạch."""
    phang = src("ocr", [5.0, 5.0, 5.0, 5.0, 1.0, 0.9, 0.8, 0.7])
    khong_han = union_pool([phang], per_source=4, weights={"ocr": 1.0})
    assert sorted(khong_han.rows.tolist()) == [0, 1, 2, 3]      # ngập video A

    co_han = union_pool([phang], per_source=4, weights={"ocr": 1.0},
                        ranges=RANGES, per_video=2)
    rows = sorted(co_han.rows.tolist())
    assert len(rows) == 4
    assert sum(r < 4 for r in rows) == 2      # video A đúng 2 suất
    assert sum(r >= 4 for r in rows) == 2     # 2 suất còn lại sang video B


def test_han_ngach_giu_dung_khung_diem_cao_nhat_moi_video():
    s = src("visual", [1.0, 9.0, 8.0, 2.0, 7.0, 3.0, 6.0, 4.0])
    p = union_pool([s], per_source=4, weights={"visual": 1.0},
                   ranges=RANGES, per_video=1)
    rows = sorted(p.rows.tolist())
    assert rows == [1, 4]        # cao nhất của A là hàng 1, của B là hàng 4


def test_han_ngach_khong_doi_ket_qua_khi_diem_da_trai_deu():
    s = src("visual", [9.0, 1.0, 0.5, 0.2, 8.0, 0.9, 0.4, 0.1])
    a = union_pool([s], per_source=2, weights={"visual": 1.0})
    b = union_pool([s], per_source=2, weights={"visual": 1.0},
                   ranges=RANGES, per_video=2)
    assert sorted(a.rows.tolist()) == sorted(b.rows.tolist()) == [0, 4]


def test_per_video_thieu_ranges_la_loi():
    with pytest.raises(ValueError, match="ranges"):
        union_pool([src("a", [1.0, 2.0])], per_source=1, weights={"a": 1.0},
                   per_video=1)


def test_per_video_be_hon_1_la_loi():
    with pytest.raises(ValueError, match="per_video"):
        union_pool([src("a", [1.0, 2.0])], per_source=1, weights={"a": 1.0},
                   ranges={"A": (0, 2)}, per_video=0)


def test_han_ngach_van_ton_trong_covered():
    s = src("ocr", [9.0, 9.0, 9.0, 9.0, 0.0, 0.0, 0.0, 0.0],
            covered=[True, True, False, False, False, False, False, False])
    p = union_pool([s], per_source=4, weights={"ocr": 1.0},
                   ranges=RANGES, per_video=3)
    assert sorted(p.rows.tolist()) == [0, 1]     # chỉ 2 khung được phủ


# ── chốt `score > 0`: covered KHÔNG có nghĩa là KHỚP ─────────────────────────

def test_khung_diem_0_khong_duoc_de_cu_du_da_covered():
    """`covered` = nguồn CÓ dữ liệu; nó KHÔNG nói truy vấn khớp.

    Khung có chữ OCR nhưng không khớp từ nào vẫn `covered=True` với BM25 = 0. Cho nó
    vào rổ là để NHIỄU chiếm suất của nguồn khác.
    """
    s = src("ocr", [3.0, 0.0, 0.0, 0.0], covered=[True, True, True, True])
    p = union_pool([s], per_source=4, weights={"ocr": 1.0})
    assert p.rows.tolist() == [0]
    assert p.per_source_n["ocr"] == 1


def test_nguon_khong_khop_gi_thi_de_cu_KHONG_khung():
    """Truy vấn không khớp từ vựng nào ⟹ nguồn đó im lặng, không đẩy rác vào rổ."""
    ocr = src("ocr", [0.0, 0.0, 0.0], covered=[True, True, True])
    vis = src("visual", [5.0, 4.0, 3.0])
    p = union_pool([vis, ocr], per_source=2, weights={"visual": 1.0, "ocr": 0.13})
    assert p.per_source_n["ocr"] == 0
    assert sorted(p.rows.tolist()) == [0, 1]
    assert all("ocr" not in v for v in p.provenance.values())


def test_diem_am_cung_bi_loai():
    """z-score âm nghĩa là dưới trung bình — cũng không phải bằng chứng khớp."""
    s = src("asr", [2.0, -1.0, -3.0], covered=[True, True, True])
    p = union_pool([s], per_source=3, weights={"asr": 1.0})
    assert p.rows.tolist() == [0]
