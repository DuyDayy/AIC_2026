"""
Kho ảnh keyframe — chốt rằng ảnh hiện ra ĐÚNG là khung chỉ mục nói
==================================================================

Đây là lỗi câm nguy hiểm nhất của tầng hiển thị: nếu số trong tên file `.webp` không
phải `n` của chỉ mục, operator nhìn MỘT khung rồi bấm chốt một khung KHÁC. `frame_id`
nộp đi lấy từ chỉ mục chứ không lấy từ ảnh, nên không có gì báo.

Test cần kho ảnh và video gốc nên tự bỏ qua khi thiếu.
"""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

import numpy as np
import pytest

from src.feedback.thumbs import KhoAnh, cat_bang_ffmpeg

KHO_DIR = Path("data/keyframes")
VIDEO = Path("data/video/L21_V001.mp4")
pytestmark = pytest.mark.skipif(
    not (KHO_DIR / "L21-L25.zip").is_file(), reason="chưa có kho ảnh")


@pytest.fixture(scope="module")
def kho():
    k = KhoAnh(KHO_DIR)
    k.mo_san()
    return k


def _mang(b: bytes, canh: int = 64) -> np.ndarray:
    from PIL import Image
    im = Image.open(io.BytesIO(b)).convert("L").resize((canh, canh))
    return np.asarray(im, dtype=np.float32)


def test_bang_tra_khop_dung_so_khung_cua_chi_muc(kho):
    n_chi_muc = len(np.load("data/embed/frame_idx.npy"))
    assert len(kho) == n_chi_muc, (len(kho), n_chi_muc)


def test_moi_khoa_cua_chi_muc_deu_co_anh(kho):
    ids = np.load("data/embed/ids.npy", allow_pickle=True)
    thieu = [(str(v), int(n)) for v, n in ids[::997]
             if (str(v), int(n)) not in kho.bang]
    assert not thieu, f"{len(thieu)} khoá thiếu ảnh, ví dụ {thieu[:5]}"


@pytest.mark.skipif(not VIDEO.is_file(), reason="chưa có video gốc")
@pytest.mark.parametrize("vt", [0, 0.5, 0.95],
                         ids=["dau", "giua", "cuoi"])
def test_anh_trong_kho_LA_khung_ma_chi_muc_noi(kho, vt):
    """
    So ảnh kho với khung cắt từ video tại `pts_time` của CHÍNH `n` đó.

    Không so byte: kho là WebP nén mất mát, ffmpeg trả JPEG — khác codec, khác chất
    lượng. So bằng ảnh xám 64×64: cùng khung thì sai khác trung bình nhỏ, khác khung
    thì lớn. Ngưỡng lấy từ chính dữ liệu — so thêm với một khung CÁCH XA để biết
    "khác khung" trông như thế nào.
    """
    import json
    ids = np.load("data/embed/ids.npy", allow_pickle=True)
    FI = np.load("data/embed/frame_idx.npy")
    lo, hi = json.load(open("data/embed/ranges.json"))["L21_V001"]
    # `n` LẤY TỪ CHỈ MỤC, không mã cứng: `n` không liên tục (1,2,3,6,7…) nên đoán
    # một số bất kỳ sẽ trúng khoảng trống và test hỏng vì lý do sai.
    vi = lo + int((hi - lo - 1) * vt)
    n = int(ids[vi][1])
    fi = int(FI[vi])

    a = kho.doc("L21_V001", n)
    assert a, f"kho thiếu (L21_V001, n={n})"
    b = cat_bang_ffmpeg("L21_V001", fi / 30.0)
    assert b, "ffmpeg không cắt được khung đối chứng"

    dung = float(np.abs(_mang(a) - _mang(b)).mean())
    xa = float(np.abs(_mang(a) - _mang(
        cat_bang_ffmpeg("L21_V001", fi / 30.0 + 60.0))).mean())
    assert dung < xa / 2, (
        f"n={n}: lệch với khung ĐÚNG ({dung:.1f}) không nhỏ hơn hẳn lệch với khung "
        f"cách 60 s ({xa:.1f}) — số trong tên file có thể KHÔNG phải `n`")


def test_khoa_khong_co_thi_tra_rong_chu_khong_NO(kho):
    assert kho.doc("KHONG_CO_V999", 1) == b""
    assert kho.doc("L21_V001", 999999) == b""


def test_doc_nhieu_giu_dung_thu_tu(kho):
    keys = [("L21_V001", 1), ("L21_V001", 3), ("L21_V001", 6)]
    r = kho.doc_nhieu(keys, workers=3)
    assert len(r) == 3 and all(r)
    import base64
    for k, b64 in zip(keys, r):
        assert base64.b64decode(b64) == kho.doc(*k)


@pytest.mark.skipif(not VIDEO.is_file(), reason="chưa có video gốc")
def test_ffmpeg_duong_lui_van_chay():
    assert cat_bang_ffmpeg("L21_V001", 129.466667)
    assert cat_bang_ffmpeg("KHONG_CO_V999", 1.0) == b""
