"""
Phiên phản hồi — mỗi test chốt một điều tài liệu Rocchio RF ĐÒI, không phải "chạy được"
=======================================================================================

Nguồn: `ke_hoach_hien_thuc_rocchio_rf_aic_2026.docx` §4.1, §5.3, §8, §12.1, §14.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.feedback.rocchio import RocchioConfig, l2
from src.feedback.session import SearchSession

RNG = np.random.default_rng(7)
N, D = 400, 24


class _Idx:
    """Chỉ mục giả tối thiểu: `SearchSession` đọc `.emb` và `.frame_idx`."""

    def __init__(self, emb):
        self.emb = emb
        self.n_frames = emb.shape[0]
        self.frame_idx = np.arange(emb.shape[0]) * 13


def _phien(**kw):
    emb = np.stack([l2(v) for v in RNG.normal(size=(N, D)).astype(np.float32)])
    # 40 video × 10 khung, mốc thời gian cách nhau 1 giây trong mỗi video
    vid = np.array([f"V{i // 10:03d}" for i in range(N)])
    t = np.array([float(i % 10) for i in range(N)])
    txt = {"ocr": np.zeros(N, np.float32), "asr": np.zeros(N, np.float32)}
    s = SearchSession(_Idx(emb), txt, t, vid,
                      {"visual": 1 / 3, "ocr": 1 / 3, "asr": 1 / 3},
                      cfg=kw.pop("cfg", RocchioConfig(radius_sec=1.5, max_frames=7)),
                      **kw)
    s.khoi_tao(l2(RNG.normal(size=D).astype(np.float32)))
    return s


# ============================================================
# §5.3 — Reset là thao tác BẮT BUỘC, và phải khôi phục CHÍNH XÁC
# ============================================================


def test_reset_khoi_phuc_dung_thu_hang_vong_0():
    s = _phien()
    goc = s.rank().copy()
    for r in (5, 123, 300):
        s.feedback(r)
    assert not np.array_equal(np.argsort(-s.rank()), np.argsort(-goc))
    s.reset()
    assert np.array_equal(s.rank(), goc)
    assert s.round == 0 and not s.history


# ============================================================
# §7 — Undo chỉ bỏ lần click GẦN NHẤT
# ============================================================


def test_undo_chi_bo_click_cuoi():
    s = _phien()
    s.feedback(10)
    sau1 = s.rank().copy()
    s.feedback(20)
    assert s.undo() is True
    assert np.allclose(s.rank(), sau1, atol=1e-6)
    assert s.state().positives == [10]
    s.undo()
    assert s.undo() is False        # hết lịch sử


def test_click_lai_cung_khung_la_BO_CHON_khong_cong_don():
    s = _phien()
    s.feedback(10)
    s.feedback(10)
    assert s.state().positives == []


# ============================================================
# §8 — TRAKE: feedback của E₂ KHÔNG được đụng E₁/E₃
# ============================================================


def test_feedback_mot_event_khong_cham_event_khac():
    s = _phien(task="trake")
    q2 = l2(RNG.normal(size=D).astype(np.float32))
    q3 = l2(RNG.normal(size=D).astype(np.float32))
    s.khoi_tao(q2, event_id=1)
    s.khoi_tao(q3, event_id=2)
    r1_truoc, r2_truoc, r3_truoc = s.rank(0).copy(), s.rank(1).copy(), s.rank(2).copy()
    s.feedback(77, event_id=1)
    # E₁ và E₃ phải y NGUYÊN…
    assert np.array_equal(s.rank(0), r1_truoc)
    assert np.array_equal(s.rank(2), r3_truoc)
    # …còn E₂ phải THẬT SỰ đổi, nếu không thì test trên đúng một cách vô nghĩa
    assert not np.array_equal(s.rank(1), r2_truoc)
    assert s.state(1).positives == [77]
    assert s.state(0).positives == [] and s.state(2).positives == []


def test_event_chua_khoi_tao_thi_NEM_chu_khong_tao_ngam():
    s = _phien()
    with pytest.raises(KeyError, match="chưa khởi tạo"):
        s.feedback(1, event_id=9)


# ============================================================
# §2 — Rocchio phải chạm MỌI khung, không chỉ sắp lại danh sách cũ
# ============================================================


def test_khung_ngoai_top_van_len_duoc():
    """Nếu chỉ rerank danh sách cũ thì một khung hạng chót vĩnh viễn ở chót."""
    s = _phien()
    goc = np.argsort(-s.rank())
    chot = int(goc[-1])
    s.feedback(chot)                       # click đúng khung tệ nhất
    moi = np.argsort(-s.rank())
    assert int(np.flatnonzero(moi == chot)[0]) < len(goc) - 1


# ============================================================
# §12.1 — Drift guard hạ β, và ghi lại để log được
# ============================================================


def test_drift_guard_ha_beta_khi_query_troi_qua_xa():
    s = _phien(cfg=RocchioConfig(beta=50.0, drift_threshold=0.95,
                                 radius_sec=1.5, max_frames=7))
    r = s.feedback(42)
    assert r["beta"] < 50.0
    assert s.state().beta_effective == pytest.approx(50.0 * 0.5 ** 3)


def test_khong_troi_thi_beta_giu_nguyen():
    s = _phien(cfg=RocchioConfig(beta=0.2, drift_threshold=0.1,
                                 radius_sec=1.5, max_frames=7))
    r = s.feedback(42)
    assert r["beta"] == pytest.approx(0.2) and not r["drift"]


# ============================================================
# §3.2 — prototype gộp khung cùng video trong bán kính, có trần
# ============================================================


def test_prototype_chi_lay_cung_video_va_trong_ban_kinh():
    s = _phien(cfg=RocchioConfig(radius_sec=1.5, max_frames=7))
    rows = s._prototype_rows(15)            # video V001, t = 5.0
    assert rows.size and all(r // 10 == 1 for r in rows.tolist())
    assert all(abs(float(r % 10) - 5.0) <= 1.5 for r in rows.tolist())


def test_prototype_ton_trong_tran_max_frames():
    s = _phien(cfg=RocchioConfig(radius_sec=100.0, max_frames=3))
    assert s._prototype_rows(15).size == 3


# ============================================================
# §6.1 — hợp đồng độ trễ: một vòng phải ĐO ĐƯỢC
# ============================================================


def test_moi_vong_ghi_lai_do_tre():
    s = _phien()
    s.feedback(1)
    s.feedback(2)
    assert len(s.latency_ms) == 2 and all(x >= 0 for x in s.latency_ms)


# ============================================================
# ✓ Chốt — bật/tắt, và KHÔNG được sinh bản sao
# ============================================================


def test_chot_bam_lai_la_BO_GHIM_khong_cong_don():
    """🔴 Lỗi đã xảy ra: so `(video, int)` mà lưu `(video, (int,))` ⟹ `in` không bao
    giờ khớp, bấm lại chỉ thêm bản sao. Mỗi bản sao là một dòng bài nộp vứt đi vì
    thể lệ chấm `R@k = max`."""
    s = _phien()
    assert s.submit(10) == 1
    assert s.submit(10) == 0, "bấm lại phải BỎ ghim"
    assert s.chot == []
    s.submit(10); s.submit(10); s.submit(10)
    assert len(s.chot) == 1


def test_chot_khong_bao_gio_co_dong_trung_trong_bai_nop():
    s = _phien()
    for r in (10, 20, 10, 30, 20):
        s.submit(r)
    lines = s.dong_nop(100)
    assert len(lines) == len(set(lines)), "bài nộp có dòng trùng"


def test_chot_duoc_GHIM_len_dau_bai_nop():
    s = _phien()
    xa = int(np.argsort(-s.rank())[-1])          # khung tệ nhất
    s.submit(xa)
    assert s.dong_nop(100)[0] == (str(s.video_of[xa]),
                                  (int(s.index.frame_idx[xa]),))


# ============================================================
# Run THỊ GIÁC mở rộng — tĩnh, và KHÔNG được tăng quyền vote của modality
# ============================================================


def test_them_run_thi_giac_KHONG_tang_quyen_vote_cua_modality():
    """Bất biến ③: `Σ_j β_mj = 1` trong từng modality.

    Nếu thêm bản mở rộng mà cứ cộng thẳng vào thì thị giác tự nhân đôi lá phiếu —
    đúng lỗi mà RRF PHẲNG mắc phải và E3 đo được là mất 0,085 MRR.
    """
    s1 = _phien()
    q0 = s1.state().q_original
    vx = np.zeros(N, np.float32)                 # mở rộng đóng góp 0
    s2 = SearchSession(s1.index, s1.text_rrf, s1.times_s, s1.video_of, s1.alpha,
                       cfg=s1.cfg, visual_extra=vx, n_visual_run=3)
    s2.khoi_tao(q0)
    # với `visual_extra = 0` và 3 run, phần thị giác phải bằng ĐÚNG 1/3 bản 1 run
    r1 = s1.rank() - sum(s1.alpha[m] * r for m, r in s1.text_rrf.items())
    r2 = s2.rank() - sum(s2.alpha[m] * r for m, r in s2.text_rrf.items())
    assert np.allclose(r2, r1 / 3, atol=1e-6)


def test_rocchio_chi_doi_run_GOC_khong_doi_ban_mo_rong():
    """Bản mở rộng là câu KHÁC, không có `q₀` để cập nhật — nó phải đứng yên."""
    s = _phien()
    vx = RNG.random(N).astype(np.float32) * 1e-3
    s2 = SearchSession(s.index, s.text_rrf, s.times_s, s.video_of, s.alpha,
                       cfg=s.cfg, visual_extra=vx, n_visual_run=3)
    s2.khoi_tao(s.state().q_original)
    truoc = s2.rank().copy()
    s2.feedback(42)
    sau = s2.rank()
    assert not np.allclose(truoc, sau)           # có đổi…
    assert s2.visual_extra is vx                 # …nhưng phần mở rộng y nguyên
    assert np.array_equal(s2.visual_extra, vx)
