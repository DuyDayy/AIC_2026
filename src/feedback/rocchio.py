"""
Rocchio Relevance Feedback — tinh chỉnh TRUY VẤN, không phải xếp lại danh sách cũ
=================================================================================

Nguồn: `ke_hoach_hien_thuc_rocchio_rf_aic_2026.docx` §3, và Rocchio (1971).

    q′ = normalize( α·q₀ + β·centroid(D⁺) − γ·centroid(D⁻) )

BỐN TÍNH CHẤT TOÁN HỌC, và chúng quyết định cách cài
----------------------------------------------------

**(1) Chuẩn hoá L2 KHÔNG đổi thứ hạng trong một vòng.** Đặt
`u = α·q₀ + β·c⁺ − γ·c⁻`. Với mọi khung `d`:

    q′ · v_d = (u · v_d) / ‖u‖

`‖u‖ > 0` là hằng số theo `d`, nên `argsort(q′·v) ≡ argsort(u·v)`. Chuẩn hoá vẫn
BẮT BUỘC vì hai lý do khác: cộng dồn nhiều vòng, và `drift_guard` đo `cos(q₀,q′)`
— cả hai cần độ dài chuẩn.

**(2) Chỉ TỈ LỆ `β/α` có nghĩa (khi γ=0).** `u·v = α(q₀·v) + β(c⁺·v)`; nhân cả hai
hệ số với `k>0` chỉ nhân điểm với `k`, không đổi thứ hạng. Nên quét `β` với `α=1`
cố định là quét TOÀN BỘ không gian tỉ lệ — không mất tính tổng quát. Tài liệu §11.3
quét đúng như vậy.

**(3) Rocchio một vòng = trộn TUYẾN TÍNH hai độ tương đồng.** Từ (1),(2): thứ hạng
sau feedback là thứ hạng của `(q₀·v) + (β/α)(c⁺·v)`. Tức nó KHÔNG sinh ra tín hiệu
mới; nó nội suy giữa "giống câu hỏi" và "giống thứ đã chọn". Hệ quả kiểm được: `β=0`
phải trả về đúng thứ hạng gốc, và `β→∞` phải trả về thứ hạng theo `c⁺` — hai ca này
là test.

**(4) Cập nhật LUÔN từ `q₀`, không từ `q` vòng trước.** Nếu lặp
`qₜ₊₁ = norm(α·qₜ + β·cₜ)` thì sau `T` vòng trọng số của `q₀` là `∏(α/‖·‖)`, suy
giảm theo cấp số nhân — query trôi khỏi intent gốc mà không có tham số nào chặn.
Cập nhật từ `q₀` giữ hệ số của nó cố định bằng `α`. Tài liệu §1.1 và §13 đòi điều
này; ở đây nó là CHỮ KÝ HÀM: `update()` nhận `q_original`, không nhận `q_prev`.

KHÔNG TRỘN GIỮA HAI KHÔNG GIAN NHÚNG (tài liệu §4.3). Hàm này chỉ nhận vector cùng
một encoder. Kho hiện có đúng một không gian dense (`jina-clip-v2`); OCR/ASR là BM25
nên không có vector để lấy centroid — chúng đi qua ③ RRF **không đổi**, đúng như
tài liệu: fusion chỉ xảy ra SAU khi mỗi encoder đã tự xếp hạng.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["RocchioConfig", "l2", "shot_balanced_centroid", "update", "drift_guard"]


@dataclass(frozen=True)
class RocchioConfig:
    """Cấu hình baseline theo tài liệu §15. `gamma=0` là MVP positive-only (§3.1)."""

    alpha: float = 1.0
    #: [ĐO] 249 câu gen299, bootstrap bắt cặp 4000 lần, trên 64 câu mà operator THẬT SỰ
    #: click được (khung đúng nằm trong top-100 — điều kiện §11.1):
    #:     β=0,4   ΔMRR +0,1770  KTC95 [+0,1035, +0,2575]  thắng 34 / thua 6
    #:     β=0,65  ΔMRR +0,2455  KTC95 [+0,1601, +0,3313]  thắng 38 / thua 2
    #:     β=0,8   ΔMRR +0,2557  KTC95 [+0,1680, +0,3446]  thắng 38 / thua 3
    #:     β=1,0   ΔMRR +0,2643  KTC95 [+0,1739, +0,3552]  thắng 39 / thua 2
    #: Chọn 0,8 chứ không 1,0: chênh +0,0086 nằm sâu trong nhiễu, còn β=1,0 nghĩa là
    #: prototype nặng NGANG câu hỏi gốc — mất neo khi operator lỡ click nhầm. 0,8 giữ
    #: gần hết khoản lãi mà vẫn để câu gốc nặng hơn.
    #: Giá trị §15 của tài liệu là 0,65; phép đo cho thấy cao hơn thì tốt hơn.
    beta: float = 0.8
    gamma: float = 0.0
    #: Bán kính cửa sổ thời gian khi dựng prototype, GIÂY (§15: 1.5).
    radius_sec: float = 1.5
    #: Trần số khung gộp vào một prototype (§15: 7).
    max_frames: int = 7
    #: Cân bằng theo shot trước khi trung bình (§3.2) — chặn một shot chiếm trọng số.
    balance_by_shot: bool = True
    #: Ngưỡng `cos(q₀, q′)`; dưới ngưỡng thì giảm β (§12.1).
    drift_threshold: float = 0.60


def l2(v: np.ndarray) -> np.ndarray:
    """Chuẩn hoá L2 an toàn với vector 0 (trả lại chính nó thay vì NaN)."""
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v if n < 1e-12 else (v / n).astype(np.float32)


def shot_balanced_centroid(vecs: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
    """
    Centroid có cân bằng theo nhóm — `groups[i]` là shot của `vecs[i]`.

    Tài liệu §3.2: *"tính centroid từng shot trước rồi mới trung bình các shot để
    tránh một shot chiếm trọng số"*. Không có nó thì một shot đóng góp 5 khung sẽ
    lấn một shot đóng góp 1 khung theo tỉ lệ 5:1 dù operator coi hai shot ngang nhau.

    `groups=None` ⟹ trung bình phẳng.
    """
    vecs = np.asarray(vecs, dtype=np.float32)
    if vecs.ndim != 2 or vecs.shape[0] == 0:
        raise ValueError(f"cần ma trận (n, d) với n ≥ 1, nhận {vecs.shape}")
    if groups is None:
        return l2(vecs.mean(axis=0))
    groups = np.asarray(groups)
    if groups.shape[0] != vecs.shape[0]:
        raise ValueError(f"groups {groups.shape[0]} ≠ vecs {vecs.shape[0]}")
    per = [l2(vecs[groups == g].mean(axis=0)) for g in np.unique(groups)]
    return l2(np.mean(per, axis=0))


def update(
    q_original: np.ndarray,
    positives: np.ndarray | None = None,
    negatives: np.ndarray | None = None,
    *,
    cfg: RocchioConfig = RocchioConfig(),
    pos_groups: np.ndarray | None = None,
    neg_groups: np.ndarray | None = None,
) -> np.ndarray:
    """
    `q′ = normalize(α·q₀ + β·centroid(D⁺) − γ·centroid(D⁻))`.

    Args:
        q_original: `q₀`, **luôn là truy vấn GỐC** — xem tính chất (4) ở đầu module.
        positives / negatives: ma trận `(n, d)` vector khung, CÙNG không gian với `q₀`.
        pos_groups / neg_groups: shot của từng khung, để cân bằng theo shot.

    Raises:
        ValueError: lệch số chiều, hệ số âm, hoặc `gamma > 0` mà không có negative.
    """
    q0 = l2(q_original)
    if cfg.alpha < 0 or cfg.beta < 0 or cfg.gamma < 0:
        raise ValueError("α, β, γ không được âm")
    u = cfg.alpha * q0
    if positives is not None and len(positives):
        c = shot_balanced_centroid(positives, pos_groups if cfg.balance_by_shot else None)
        if c.shape != q0.shape:
            raise ValueError(f"positive {c.shape} ≠ q₀ {q0.shape} — trộn nhầm không gian?")
        u = u + cfg.beta * c
    if cfg.gamma > 0:
        if negatives is None or not len(negatives):
            raise ValueError("γ > 0 nhưng không có negative — đặt γ=0 cho MVP")
        c = shot_balanced_centroid(negatives, neg_groups if cfg.balance_by_shot else None)
        u = u - cfg.gamma * c
    return l2(u)


def drift_guard(
    q_original: np.ndarray, q_updated: np.ndarray, cfg: RocchioConfig = RocchioConfig()
) -> tuple[bool, float]:
    """
    `(có trôi?, cos(q₀, q′))` — tài liệu §12.1.

    Bên gọi xử lý: giảm β một nửa rồi tính lại, hoặc bỏ feedback mới nhất. Hàm này
    chỉ ĐO, không tự sửa, để phép sửa nằm ở một chỗ và ghi log được.
    """
    c = float(np.dot(l2(q_original), l2(q_updated)))
    return c < cfg.drift_threshold, c
