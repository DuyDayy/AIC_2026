"""
Câu hỏi Q&A → câu mô tả, trước khi mã hoá làm probe truy xuất.

Ý lấy từ NII-UIT tại VBS2025 (mục 2.7). Lý do: tháp văn bản jina-clip huấn luyện trên
**caption**, còn câu nghi vấn hỏi về thứ TA CHƯA BIẾT — cụm "màu gì" là chỗ trống, không
mô tả khung hình nào cả.

⚠️ Phép biến đổi này CHƯA có phép đo trên dữ liệu thật (⑥ chưa có bộ eval). Nên bài kiểm
ở đây tập trung vào **an toàn khi thất bại**: không bao giờ trả rỗng, không cắt quá tay,
và không đụng vào câu trần thuật.
"""

import pytest

from src.retrieval.probe import declarativize


# ── việc chính: bỏ cụm hỏi, giữ phần mô tả ───────────────────────────────────

@pytest.mark.parametrize("hoi,mong_doi", [
    ("Người phụ nữ mặc váy đỏ đang cầm ly màu gì?",
     "Người phụ nữ mặc váy đỏ đang cầm ly"),
    ("Biển số xe của chiếc xe màu đỏ là gì?",
     "Biển số xe của chiếc xe màu đỏ"),
    ("Người đàn ông áo trắng đang đứng ở đâu?",
     "Người đàn ông áo trắng đang đứng"),
    ("Sự kiện này diễn ra khi nào?",
     "Sự kiện này diễn ra"),
    ("Có bao nhiêu người đứng trên sân khấu?",
     "người đứng trên sân khấu"),
    ("Mấy chiếc xe đang chạy trên đường?",
     "chiếc xe đang chạy trên đường"),
])
def test_bo_cum_hoi_giu_mo_ta(hoi, mong_doi):
    assert declarativize(hoi) == mong_doi


def test_bo_cum_dan_dau():
    assert declarativize("Trong video quay cảnh bữa tiệc, người đàn ông cầm cốc nước") \
        == "người đàn ông cầm cốc nước"


# ── AN TOÀN KHI THẤT BẠI — phần quan trọng hơn, vì chưa đo được ─────────────

def test_cau_tran_thuat_khong_bi_dung_toi():
    """KIS/TRAKE không đi qua hàm này, nhưng lỡ có thì cũng không được hỏng."""
    for s in ("Tìm cảnh hai người đàn ông ký văn bản",
              "Tìm hình ảnh ruộng hành lá xanh tốt với nhiều luống cây"):
        assert declarativize(s) == s


def test_khong_bao_gio_tra_chuoi_rong():
    """Probe rỗng làm ② chấm mọi khung bằng nhau — hỏng lặng lẽ, phải chặn."""
    for s in ("Ai?", "Gì?", "Ở đâu?", "màu gì", "?", "bao nhiêu"):
        assert declarativize(s).strip() != ""


def test_cat_qua_tay_thi_giu_nguyen():
    """Còn dưới 3 từ thì vector mã hoá ra vô nghĩa — thà giữ câu gốc."""
    assert declarativize("Con mèo màu gì?") == "Con mèo màu gì?"


def test_chuoi_rong_va_none_khong_no():
    assert declarativize("") == ""
    assert declarativize(None) == ""


def test_tat_dinh():
    q = "Người phụ nữ mặc váy đỏ đang cầm ly màu gì?"
    assert declarativize(q) == declarativize(q)


def test_khong_con_dau_hoi_o_cuoi():
    for s in ("Người đàn ông đang cầm vật gì?",
              "Có bao nhiêu người đứng trên sân khấu?"):
        assert not declarativize(s).endswith("?")


def test_khong_lam_dai_them():
    """Chỉ được BỎ bớt, không được thêm chữ — thêm là bịa nội dung không có trong đề."""
    for s in ("Người phụ nữ mặc váy đỏ đang cầm ly màu gì?",
              "Có bao nhiêu người đứng trên sân khấu?",
              "Tìm cảnh hai người đàn ông ký văn bản"):
        assert len(declarativize(s)) <= len(s)
