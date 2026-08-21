"""
Lưu phiên — chốt "chết rồi khôi phục ra ĐÚNG trạng thái cũ", không chốt "ghi được file"
=======================================================================================

Nguồn: `ke_hoach_hien_thuc_rocchio_rf_aic_2026.docx` §12 (log để replay), §14 (DoD).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.feedback.rocchio import RocchioConfig, l2
from src.feedback.session import SearchSession
from src.feedback.store import SessionStore, liet_ke, phat_lai

RNG = np.random.default_rng(11)
N, D = 300, 16


class _Idx:
    def __init__(self, emb):
        self.emb = emb
        self.n_frames = emb.shape[0]
        self.frame_idx = np.arange(emb.shape[0]) * 7


def _moi(tmp, sid="s1", luu=True):
    emb = np.stack([l2(v) for v in RNG.normal(size=(N, D)).astype(np.float32)])
    vid = np.array([f"V{i // 10:03d}" for i in range(N)])
    t = np.array([float(i % 10) for i in range(N)])
    txt = {"ocr": RNG.random(N).astype(np.float32),
           "asr": RNG.random(N).astype(np.float32)}
    s = SearchSession(_Idx(emb), txt, t, vid,
                      {"visual": 1 / 3, "ocr": 1 / 3, "asr": 1 / 3},
                      cfg=RocchioConfig(radius_sec=1.5, max_frames=5))
    s.session_id = sid
    q0 = l2(RNG.normal(size=D).astype(np.float32))
    s.khoi_tao(q0)
    if luu:
        s.store = SessionStore(sid, tmp)
        s.store.khoi_tao({"query_text": "thử", "task": "kis"}, {0: q0}, txt)
    return s, q0, txt, emb, vid, t


def _khoi_phuc(tmp, sid, emb, vid, t):
    """Dựng lại phiên từ đĩa — mô phỏng server khởi động sau khi chết."""
    st = SessionStore(sid, tmp)
    meta, q0, txt, ev = st.doc()
    s = SearchSession(_Idx(emb), txt, t, vid,
                      {"visual": 1 / 3, "ocr": 1 / 3, "asr": 1 / 3},
                      cfg=RocchioConfig(radius_sec=1.5, max_frames=5),
                      task=meta["task"], query_text=meta["query_text"])
    for e, v in q0.items():
        s.khoi_tao(v, event_id=e)
    phat_lai(s, ev)          # store=None ⟹ phát lại KHÔNG ghi lại vào nhật ký
    return s, ev


# ============================================================
# Bất biến trung tâm: khôi phục ra ĐÚNG xếp hạng
# ============================================================


def test_chet_giua_phien_khoi_phuc_ra_dung_xep_hang(tmp_path):
    s, *rest = _moi(tmp_path)
    emb, vid, t = rest[2], rest[3], rest[4]
    for r in (5, 77, 140):
        s.feedback(r)
    s.submit(77)
    s.dat_answer("hơn 14,5 tỷ đồng")
    truoc = s.rank().copy()

    lai, _ = _khoi_phuc(tmp_path, "s1", emb, vid, t)
    assert np.allclose(lai.rank(), truoc, atol=1e-6)
    assert lai.state().positives == s.state().positives
    assert lai.chot == s.chot
    assert lai.answer == "hơn 14,5 tỷ đồng"


def test_phat_lai_theo_THAO_TAC_nen_undo_cung_dung(tmp_path):
    """Gán thẳng trạng thái cuối sẽ đúng ở ca dễ và SAI ở ca có undo."""
    s, *rest = _moi(tmp_path, "s2")
    emb, vid, t = rest[2], rest[3], rest[4]
    s.feedback(10)
    s.feedback(20)
    s.undo()
    s.feedback(30)
    truoc = s.rank().copy()
    lai, _ = _khoi_phuc(tmp_path, "s2", emb, vid, t)
    assert np.allclose(lai.rank(), truoc, atol=1e-6)
    assert lai.state().positives == [10, 30]


def test_reset_cung_duoc_phat_lai(tmp_path):
    s, *rest = _moi(tmp_path, "s3")
    emb, vid, t = rest[2], rest[3], rest[4]
    s.feedback(10)
    s.reset()
    s.feedback(99)
    lai, _ = _khoi_phuc(tmp_path, "s3", emb, vid, t)
    assert lai.state().positives == [99]
    assert np.allclose(lai.rank(), s.rank(), atol=1e-6)


# ============================================================
# Chết GIỮA LÚC GHI — dòng cuối cụt
# ============================================================


def test_dong_nhat_ky_cut_bi_bo_qua_chu_khong_lam_MAT_CA_PHIEN(tmp_path):
    """Đây là lý do chọn nối thêm thay vì ghi đè ảnh chụp trạng thái."""
    s, *rest = _moi(tmp_path, "s4")
    emb, vid, t = rest[2], rest[3], rest[4]
    s.feedback(10)
    s.feedback(20)
    p = tmp_path / "s4" / "events.jsonl"
    p.write_text(p.read_text(encoding="utf-8") + '{"t": 1.0, "op": "feed',
                 encoding="utf-8")          # cụt đúng như khi bị kill
    lai, ev = _khoi_phuc(tmp_path, "s4", emb, vid, t)
    assert len(ev) == 2                      # hai thao tác nguyên vẹn còn nguyên
    assert lai.state().positives == [10, 20]


def test_nhat_ky_thuc_su_nam_tren_dia_ngay_sau_moi_thao_tac(tmp_path):
    """`flush`+`fsync`: không có nó thì dữ liệu còn trong bộ đệm OS, kill là mất."""
    s, *_ = _moi(tmp_path, "s5")
    s.feedback(1)
    dong = (tmp_path / "s5" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(dong) == 1 and json.loads(dong[0])["op"] == "feedback"


# ============================================================
# Phát lại KHÔNG được ghi lại — nếu không nhật ký tự nhân đôi
# ============================================================


def test_phat_lai_khong_lam_nhat_ky_phinh_them(tmp_path):
    s, *rest = _moi(tmp_path, "s6")
    emb, vid, t = rest[2], rest[3], rest[4]
    s.feedback(10)
    s.feedback(20)
    n1 = len((tmp_path / "s6" / "events.jsonl").read_text(encoding="utf-8").splitlines())
    for _ in range(3):
        _khoi_phuc(tmp_path, "s6", emb, vid, t)
    n2 = len((tmp_path / "s6" / "events.jsonl").read_text(encoding="utf-8").splitlines())
    assert n1 == n2 == 2


# ============================================================
# Liệt kê để UI cho chọn khôi phục
# ============================================================


def test_liet_ke_moi_nhat_truoc(tmp_path):
    for sid in ("a", "b"):
        s, *_ = _moi(tmp_path, sid)
        s.feedback(3)
    ds = liet_ke(tmp_path)
    assert {x["session_id"] for x in ds} == {"a", "b"}
    assert all(x["n_events"] == 1 and x["task"] == "kis" for x in ds)
    assert ds[0]["created_at"] >= ds[1]["created_at"]


def test_thu_muc_rong_thi_tra_danh_sach_rong_chu_khong_NO(tmp_path):
    assert liet_ke(tmp_path / "khong-ton-tai") == []


def test_khong_luu_thi_khong_tao_file(tmp_path):
    s, *_ = _moi(tmp_path, "s7", luu=False)
    s.feedback(1)
    assert not (tmp_path / "s7").exists()


def test_doc_phien_khong_ton_tai_thi_NEM(tmp_path):
    with pytest.raises(FileNotFoundError):
        SessionStore("khong-co", tmp_path).doc()
