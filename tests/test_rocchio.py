"""
Rocchio RF — mỗi test chốt MỘT tính chất toán học, không chốt "chạy được"
=========================================================================

Nguồn công thức: `ke_hoach_hien_thuc_rocchio_rf_aic_2026.docx` §3.1; Rocchio (1971).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.feedback.rocchio import (
    RocchioConfig, drift_guard, l2, shot_balanced_centroid, update,
)

RNG = np.random.default_rng(0)


def _v(n=1, d=32):
    x = RNG.normal(size=(n, d)).astype(np.float32)
    return np.stack([l2(r) for r in x]) if n > 1 else l2(x[0])


# ============================================================
# (1) Chuẩn hoá L2 KHÔNG đổi thứ hạng trong một vòng
# ============================================================


def test_chuan_hoa_khong_doi_thu_hang():
    """`q′·v = (u·v)/‖u‖`, mà `‖u‖` là hằng số theo `v` ⟹ argsort không đổi."""
    q0, P, V = _v(), _v(4), _v(200)
    cfg = RocchioConfig(beta=0.65)
    q1 = update(q0, P, cfg=cfg)
    u = cfg.alpha * l2(q0) + cfg.beta * shot_balanced_centroid(P)   # CHƯA chuẩn hoá
    assert np.array_equal(np.argsort(-(V @ q1)), np.argsort(-(V @ u)))
    assert abs(float(np.linalg.norm(q1)) - 1.0) < 1e-5


# ============================================================
# (2) Chỉ TỈ LỆ β/α có nghĩa — nên quét β với α=1 là quét đủ
# ============================================================


def test_chi_ti_le_beta_tren_alpha_co_nghia():
    q0, P, V = _v(), _v(3), _v(200)
    a = update(q0, P, cfg=RocchioConfig(alpha=1.0, beta=0.6))
    b = update(q0, P, cfg=RocchioConfig(alpha=2.0, beta=1.2))     # cùng tỉ lệ
    assert np.allclose(a, b, atol=1e-6)
    assert np.array_equal(np.argsort(-(V @ a)), np.argsort(-(V @ b)))


# ============================================================
# (3) Một vòng = trộn tuyến tính hai độ tương đồng — hai ca biên
# ============================================================


def test_beta_0_tra_ve_dung_thu_hang_goc():
    """Không có β thì Rocchio phải là phép ĐỒNG NHẤT. Nếu ca này hỏng thì mọi
    so sánh 'trước/sau feedback' đều vô nghĩa vì nền đã trôi."""
    q0, P, V = _v(), _v(5), _v(300)
    q1 = update(q0, P, cfg=RocchioConfig(beta=0.0))
    assert np.allclose(q1, l2(q0), atol=1e-6)
    assert np.array_equal(np.argsort(-(V @ q1)), np.argsort(-(V @ l2(q0))))


def test_beta_rat_lon_tra_ve_thu_hang_theo_centroid():
    q0, P, V = _v(), _v(5), _v(300)
    q1 = update(q0, P, cfg=RocchioConfig(beta=1e6))
    c = shot_balanced_centroid(P)
    assert np.array_equal(np.argsort(-(V @ q1)), np.argsort(-(V @ c)))


def test_diem_la_to_hop_tuyen_tinh_dung_he_so():
    """`u·v = α(q₀·v) + β(c·v)` — kiểm bằng số, không chỉ bằng thứ hạng."""
    q0, P, V = _v(), _v(4), _v(50)
    cfg = RocchioConfig(alpha=1.0, beta=0.65)
    q1 = update(q0, P, cfg=cfg)
    c = shot_balanced_centroid(P)
    u_dot = cfg.alpha * (V @ l2(q0)) + cfg.beta * (V @ c)
    assert np.allclose(V @ q1, u_dot / np.linalg.norm(cfg.alpha * l2(q0) + cfg.beta * c),
                       atol=1e-5)


# ============================================================
# (4) Cập nhật từ q₀, KHÔNG từ vòng trước
# ============================================================


def test_cap_nhat_day_chuyen_lam_q0_suy_giam_theo_CAP_SO_NHAN():
    """Vì sao chữ ký hàm nhận `q_original`: lặp từ `q` vòng trước làm trọng số của
    `q₀` tụt theo cấp số nhân, không tham số nào chặn được."""
    q0, P = _v(), _v(3)
    cfg = RocchioConfig(beta=0.65)
    day, tu_goc = l2(q0), l2(q0)
    for _ in range(6):
        day = update(day, P, cfg=cfg)          # SAI: nối vòng
        tu_goc = update(q0, P, cfg=cfg)        # ĐÚNG: luôn từ gốc
    c_day = float(np.dot(l2(q0), day))
    c_goc = float(np.dot(l2(q0), tu_goc))
    assert c_day < c_goc - 0.05, (c_day, c_goc)
    # và bản từ gốc là ĐIỂM BẤT ĐỘNG: lặp bao nhiêu vòng cũng ra một kết quả
    assert np.allclose(tu_goc, update(q0, P, cfg=cfg), atol=1e-6)


# ============================================================
# Prototype cân bằng theo shot (§3.2)
# ============================================================


def test_can_bang_shot_chan_mot_shot_lan_at():
    """5 khung của shot A và 1 khung của shot B phải nặng NGANG nhau."""
    d = 16
    a, b = l2(RNG.normal(size=d).astype(np.float32)), l2(RNG.normal(size=d).astype(np.float32))
    V = np.stack([a] * 5 + [b])
    g = np.array([0] * 5 + [1])
    can = shot_balanced_centroid(V, g)
    phang = shot_balanced_centroid(V, None)
    assert abs(float(np.dot(can, a)) - float(np.dot(can, b))) < 1e-5
    assert float(np.dot(phang, a)) > float(np.dot(phang, b)) + 0.2


# ============================================================
# Chốt an toàn
# ============================================================


def test_tron_nham_khong_gian_thi_NEM():
    """Tài liệu §4.3: không được cộng vector từ hai encoder khác nhau."""
    with pytest.raises(ValueError, match="không gian"):
        update(_v(d=32), _v(3, d=64))


def test_gamma_duong_ma_khong_co_negative_thi_NEM():
    with pytest.raises(ValueError, match="γ > 0"):
        update(_v(), _v(3), cfg=RocchioConfig(gamma=0.15))


def test_he_so_am_thi_NEM():
    with pytest.raises(ValueError, match="không được âm"):
        update(_v(), _v(2), cfg=RocchioConfig(beta=-0.1))


def test_drift_guard_do_dung_cosine():
    q0, P = _v(), _v(3)
    nhe = update(q0, P, cfg=RocchioConfig(beta=0.05))
    nang = update(q0, P, cfg=RocchioConfig(beta=50.0))
    _, c_nhe = drift_guard(q0, nhe)
    troi, c_nang = drift_guard(q0, nang, RocchioConfig(drift_threshold=0.60))
    assert c_nhe > c_nang and troi
