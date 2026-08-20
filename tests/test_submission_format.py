"""
Định dạng nộp bài CHÍNH THỨC AIC 2026 — mỗi luật của thể lệ một ca
==================================================================

Nguồn: "Hướng dẫn nộp bài sơ tuyển".

Vì sao file này tồn tại: mọi lỗi định dạng ở đây đều CÂM. File vẫn ghi ra, vẫn mở
được bằng Notepad, vẫn trông đúng — và điểm về 0 vì bộ chấm tách cột khác ta nghĩ.
Thể lệ cho **3 lượt nộp mỗi gói**, và "khi nộp sai định dạng vẫn tính là 01 lần
nộp", nên một lỗi định dạng tiêu mất một phần ba số lượt.
"""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import pytest

from src.submission.writer import (
    MAX_ANSWER_CHARS, SUBMISSION_DIRNAME, SubmissionError, TaskSubmission,
    clean_answer, format_task_csv, pack_submission_zip, parse_task_csv,
    task_type_from_filename, write_task_csv,
)


def _kis(*rows):
    return TaskSubmission("query-1-kis", "kis",
                          tuple((v, (f,), None) for v, f in rows), n_moments=1)


def _qa(answer, video="L01_V028", frame=3450):
    return TaskSubmission("query-3-qa", "qa", ((video, (frame,), answer),), n_moments=1)


# ============================================================
# Hậu tố tên file quyết định loại truy vấn
# ============================================================


def test_hau_to_ten_file_quyet_dinh_loai():
    assert task_type_from_filename("query-1-kis") == "kis"
    assert task_type_from_filename("query-3-qa") == "qa"
    assert task_type_from_filename("query-4-trake") == "trake"


def test_hau_to_la_lay_theo_DUOI_chu_khong_phai_tim_chuoi():
    """`"qa" in "query-2-kis"` là chuyện khác với "hậu tố là qa"."""
    assert task_type_from_filename("query-2-kis") == "kis"
    with pytest.raises(SubmissionError, match="hậu tố"):
        task_type_from_filename("query-5")
    with pytest.raises(SubmissionError, match="hậu tố"):
        task_type_from_filename("query-6-vqa")


# ============================================================
# Ba định dạng dòng
# ============================================================


def test_kis_hai_cot_khong_khoang_trang_khong_header():
    text, _ = format_task_csv(_kis(("L00_V000", 1234), ("L01_V028", 25300)))
    assert text == "L00_V000,1234\nL01_V028,25300\n"


def test_trake_dung_N_cot_theo_so_su_kien():
    sub = TaskSubmission("query-4-trake", "trake",
                         (("L10_V001", (1200, 1850, 2100, 2450), None),), n_moments=4)
    text, _ = format_task_csv(sub)
    assert text == "L10_V001,1200,1850,2100,2450\n"


def test_trake_lech_so_moc_thi_NO_chu_khong_ghi_ra():
    """"TRAKE sai số frame" là lỗi thứ 5 trong danh sách của BTC."""
    sub = TaskSubmission("query-4-trake", "trake",
                         (("L10_V001", (1200, 1850), None),), n_moments=4)
    with pytest.raises(SubmissionError, match="đề yêu cầu đúng 4"):
        write_task_csv(sub, "/tmp/khong-bao-gio-ghi")


# ============================================================
# 🔴 Q&A: bọc ngoặc kép — lỗi thứ 4 trong "5 lỗi thường gặp nhất"
# ============================================================


def test_dap_an_co_dau_phay_PHAI_duoc_boc_ngoac_kep():
    text, _ = format_task_csv(_qa("Có 3 người, bao gồm nam và nữ"))
    assert text == 'L01_V028,3450,"Có 3 người, bao gồm nam và nữ"\n'
    # và đọc ngược ra ĐÚNG BA cột — đây mới là điều thật sự quan trọng
    assert parse_task_csv(text, "qa", 1) == [
        ["L01_V028", "3450", "Có 3 người, bao gồm nam và nữ"]]


def test_noi_chuoi_bang_dau_phay_lam_HONG_dong_that():
    """Chứng minh vì sao phải dùng `csv.writer`: bản nối tay tách ra BỐN cột."""
    tho = "L01_V028,3450," + "Có 3 người, bao gồm nam và nữ"
    assert len(next(csv.reader(io.StringIO(tho)))) == 4      # sai
    text, _ = format_task_csv(_qa("Có 3 người, bao gồm nam và nữ"))
    assert len(next(csv.reader(io.StringIO(text)))) == 3     # đúng


def test_dap_an_co_ngoac_kep_duoc_escape_bang_ngoac_kep_doi():
    text, _ = format_task_csv(_qa('Anh ấy nói "Xin chào"'))
    assert text == 'L01_V028,3450,"Anh ấy nói ""Xin chào"""\n'
    assert parse_task_csv(text, "qa", 1)[0][2] == 'Anh ấy nói "Xin chào"'


def test_dap_an_don_gian_KHONG_bi_boc_ngoac_kep():
    """Thể lệ: ngoặc kép chỉ BẮT BUỘC khi có ký tự đặc biệt. Cả hai cách đều hợp lệ."""
    assert format_task_csv(_qa("Năm người"))[0] == "L01_V028,3450,Năm người\n"


# ============================================================
# Q&A: chuẩn hoá đáp án
# ============================================================


def test_bo_khoang_trang_dau_cuoi_vi_bo_cham_KHONG_tu_trim():
    """Thể lệ nói hai điều cạnh nhau: "không tự động trim" + "so sánh chính xác"."""
    a, notes = clean_answer("  Năm người  ")
    assert a == "Năm người" and notes


def test_xuong_dong_bi_gop_thanh_khoang_trang():
    assert clean_answer("Dòng 1\nDòng 2")[0] == "Dòng 1 Dòng 2"


def test_dap_an_qua_100_ky_tu_bi_CAT_va_co_ghi_chu_chu_khong_im_lang():
    a, notes = clean_answer("x" * 250)
    assert len(a) == MAX_ANSWER_CHARS
    assert any("ĐÃ CẮT" in m for m in notes)


def test_dap_an_rong_thi_NO():
    with pytest.raises(SubmissionError, match="rỗng"):
        format_task_csv(_qa("   "))


# ============================================================
# Tên video và frame_id
# ============================================================


def test_ten_video_con_duoi_mp4_thi_NO():
    with pytest.raises(SubmissionError, match=r"\.mp4"):
        format_task_csv(_kis(("L01_V028.mp4", 25300)))


def test_frame_id_luon_la_so_nguyen_thuan():
    text, _ = format_task_csv(_kis(("L01_V028", 25300)))
    assert "25300" in text and " 25300" not in text
    with pytest.raises(SubmissionError, match="số nguyên"):
        parse_task_csv("L01_V028,25 300\n", "kis", 1)


def test_doc_nguoc_bat_duoc_so_cot_sai():
    with pytest.raises(SubmissionError, match="tách ra 3 cột"):
        parse_task_csv("L01_V028,3450,5\n", "kis", 1)


# ============================================================
# Ghi file
# ============================================================


def test_ten_file_khop_ten_truy_van_va_ma_hoa_utf8_khong_BOM(tmp_path):
    p, _ = write_task_csv(_qa("Màu đỏ"), tmp_path)
    assert p.name == "query-3-qa.csv"
    raw = p.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")          # BOM làm hỏng cột đầu
    assert raw.decode("utf-8") == "L01_V028,3450,Màu đỏ\n"


def test_vuot_100_dong_thi_NO(tmp_path):
    sub = _kis(*[("L00_V000", i) for i in range(101)])
    with pytest.raises(SubmissionError, match="vượt ngân sách"):
        write_task_csv(sub, tmp_path)


# ============================================================
# 🔴 Đóng gói — "thiếu thư mục submission" là lỗi thứ 2 của BTC
# ============================================================


def _ba_file(d: Path):
    write_task_csv(_kis(("L00_V000", 1)), d)
    write_task_csv(_qa("Màu đỏ"), d)
    write_task_csv(TaskSubmission("query-4-trake", "trake",
                                  (("L10_V001", (1, 2), None),), n_moments=2), d)


def test_zip_co_dung_thu_muc_submission_o_trong(tmp_path):
    d = tmp_path / "out"; _ba_file(d)
    z = pack_submission_zip(d, tmp_path / "team_ABC_round1.zip")
    with zipfile.ZipFile(z) as zf:
        names = sorted(zf.namelist())
    assert names == [f"{SUBMISSION_DIRNAME}/query-1-kis.csv",
                     f"{SUBMISSION_DIRNAME}/query-3-qa.csv",
                     f"{SUBMISSION_DIRNAME}/query-4-trake.csv"]


def test_zip_KHONG_gom_report_json_hay_npz(tmp_path):
    """`zip -r` gói cả `_rerank_scores.npz` (hàng chục MB) và `_report.json`."""
    d = tmp_path / "out"; _ba_file(d)
    (d / "_report.json").write_text("{}", encoding="utf-8")
    (d / "_rerank_scores.npz").write_bytes(b"\x00" * 1000)
    z = pack_submission_zip(d, tmp_path / "nop.zip")
    with zipfile.ZipFile(z) as zf:
        assert all(n.endswith(".csv") for n in zf.namelist())


def test_ten_zip_co_ky_tu_la_thi_NO(tmp_path):
    d = tmp_path / "out"; _ba_file(d)
    with pytest.raises(SubmissionError, match="chữ hoặc số"):
        pack_submission_zip(d, tmp_path / "bài nộp v2.zip")


def test_thieu_hoac_thua_file_so_voi_goi_de_thi_NO(tmp_path):
    d = tmp_path / "out"; _ba_file(d)
    want = ["query-1-kis", "query-3-qa", "query-4-trake", "query-5-kis"]
    with pytest.raises(SubmissionError, match="query-5-kis"):
        pack_submission_zip(d, tmp_path / "a.zip", expected=want)
    with pytest.raises(SubmissionError, match="thừa"):
        pack_submission_zip(d, tmp_path / "b.zip", expected=["query-1-kis"])


def test_thu_muc_rong_thi_NO_chu_khong_ra_zip_rong(tmp_path):
    with pytest.raises(SubmissionError, match="rỗng"):
        pack_submission_zip(tmp_path, tmp_path / "c.zip")


# ============================================================
# 🔴 CỔNG TẦNG 0 — lệch quy ước frame_id là lỗi CÂM tuyệt đối
# ============================================================
#
# Định lý 5: lệch một hằng số kéo `final` 1,00 → 0,50 trong khi `best` vẫn 1,0 —
# không chỉ số nội bộ nào báo động, và bài nộp vẫn qua sạch mọi phép kiểm khác.
#
# Cổng này TỪNG BỊ BỎ SÓT ở `write_task_csv`: khi BTC công bố mẫu CSV, hàm mới được
# viết mà không mang theo cổng. Vô hại đúng lúc đó vì δ = 0 — tức đúng kiểu bỏ sót
# chỉ lộ ra khi δ ≠ 0, lúc không sửa lại được nữa.


def _calib(tmp_path, delta):
    import json as _json
    (tmp_path / "frame_index_calibration.json").write_text(
        _json.dumps({"delta": delta, "method": "test", "n_samples": 1, "agreement": 1.0}),
        encoding="utf-8")
    return tmp_path


def test_delta_khac_0_thi_duoc_CONG_vao_moi_frame_id(tmp_path):
    d = _calib(tmp_path, 7)
    p, _ = write_task_csv(_kis(("L00_V000", 100), ("L00_V001", 200)),
                          tmp_path / "out", calibration_dir=d)
    assert p.read_text(encoding="utf-8") == "L00_V000,107\nL00_V001,207\n"


def test_delta_ap_cho_MOI_moc_cua_TRAKE(tmp_path):
    d = _calib(tmp_path, -3)
    sub = TaskSubmission("query-4-trake", "trake",
                         (("L10_V001", (10, 20, 30), None),), n_moments=3)
    p, _ = write_task_csv(sub, tmp_path / "out", calibration_dir=d)
    assert p.read_text(encoding="utf-8") == "L10_V001,7,17,27\n"


def test_thieu_hieu_chinh_thi_NEM_chu_khong_ghi_bua(tmp_path):
    """Không có bằng chứng δ thì KHÔNG được đoán δ = 0."""
    with pytest.raises(Exception):
        write_task_csv(_kis(("L00_V000", 1)), tmp_path / "out",
                       calibration_dir=tmp_path / "trong-rong")


def test_delta_0_thi_khong_doi_gi(tmp_path):
    d = _calib(tmp_path, 0)
    p, _ = write_task_csv(_kis(("L00_V000", 1234)), tmp_path / "out", calibration_dir=d)
    assert p.read_text(encoding="utf-8") == "L00_V000,1234\n"
