"""
TRAKE với N=1 CHÍNH LÀ KIS — và ⑦ chỉ được có MỘT đường mã
============================================================

Hệ thức truy hồi `DP[i,t] = S[i,t] + max_{τ<t}(DP[i−1,τ] − λ(t−τ))` với `N = 1` suy biến
thành `max_t S[0,t]`, tức phép max-pool thường. Nên KIS **không cần** nhánh code riêng.

`run.py` từng có hai nhánh, và chúng KHÔNG tương đương ở khâu nộp: nhánh TRAKE gọi
`dante_over_videos`, vốn trả **một đường mỗi video** — tức khử trùng theo video, đo được
**−12,0pp**. Bài kiểm này khoá tính tương đương lại để hai nhánh không thể tái sinh:
đi qua `k_best_alignments` với `N=1` rồi sắp toàn cục phải cho **đúng** thứ tự mà phép
sắp xếp thẳng cho ra.
"""

import numpy as np
import pytest

from src.retrieval.dante import DEFAULT_LAMBDA, dante
from src.submission.kbest import k_best_alignments


def unified(frames_by_video, scores_by_video, n_mom, k_per_video):
    """Đúng thuật toán ⑦ trong run.py, tách ra để kiểm được."""
    scored = []
    for vid in frames_by_video:
        cand = frames_by_video[vid]
        sc = scores_by_video[vid]
        if len(cand) < n_mom:
            continue
        for al in k_best_alignments(cand, sc, k=min(k_per_video, len(cand)),
                                    pacing_penalty=DEFAULT_LAMBDA):
            scored.append((float(al.score), vid, tuple(al.frames)))
    scored.sort(key=lambda x: (-x[0], x[1], x[2]))
    return scored


def test_n1_qua_kbest_cho_dung_thu_tu_nhu_sap_thang():
    """Bài kiểm cốt lõi: N=1 phải ra đúng bảng xếp hạng của phép sắp thẳng."""
    rng = np.random.default_rng(7)
    for _ in range(30):
        vids = {}
        scores = {}
        thang = []
        for v in ("A", "B", "C"):
            m = int(rng.integers(2, 9))
            f = np.sort(rng.choice(np.arange(1000), size=m, replace=False)).tolist()
            s = rng.normal(size=m).round(4).tolist()
            vids[v] = f
            scores[v] = [s]
            thang += [(s[i], v, (f[i],)) for i in range(m)]
        thang.sort(key=lambda x: (-x[0], x[1], x[2]))
        got = unified(vids, scores, n_mom=1, k_per_video=10**9)
        assert [(round(a, 4), b, c) for a, b, c in got] == \
               [(round(a, 4), b, c) for a, b, c in thang]


def test_n1_khong_khu_trung_theo_video():
    """Lỗi cũ: một đường mỗi video. Video thắng phải giữ được NHIỀU suất."""
    vids = {"A": [10, 20, 30], "B": [40]}
    scores = {"A": [[9.0, 8.0, 7.0]], "B": [[1.0]]}
    got = unified(vids, scores, n_mom=1, k_per_video=10**9)
    assert [v for _s, v, _f in got] == ["A", "A", "A", "B"]


def test_kbest_k1_trung_khop_dante_tren_cung_du_lieu():
    """`k_best_alignments(k=1)` phải cho cùng ĐIỂM với `dante()` — cùng một phép DP."""
    rng = np.random.default_rng(3)
    for _ in range(20):
        n, m = int(rng.integers(1, 4)), int(rng.integers(4, 12))
        S = rng.normal(size=(n, m)).astype(np.float64)
        times = np.sort(rng.choice(np.arange(5000), size=m, replace=False)).astype(float)
        cand = list(range(m))
        best = k_best_alignments(cand, S.tolist(), k=1, pacing_penalty=0.0)
        p = dante(S, times, lam=0.0)
        assert best[0].score == pytest.approx(float(p.score), abs=1e-6)


def test_video_it_khung_hon_N_bi_bo_qua():
    """Video không đủ N khung trong rổ thì không thể tạo bộ hợp lệ."""
    vids = {"A": [1, 2], "B": [5, 6, 7]}
    scores = {"A": [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
              "B": [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]}
    got = unified(vids, scores, n_mom=3, k_per_video=1)
    assert [v for _s, v, _f in got] == ["B"]


def test_bo_tra_ve_luon_tang_dan_nghiem_ngat():
    rng = np.random.default_rng(11)
    vids = {"A": sorted(rng.choice(np.arange(500), size=10, replace=False).tolist())}
    scores = {"A": rng.normal(size=(3, 10)).round(3).tolist()}
    for _s, _v, fr in unified(vids, scores, n_mom=3, k_per_video=5):
        assert list(fr) == sorted(set(fr))


def test_sap_toan_cuc_tat_dinh_giua_hai_lan_goi():
    rng = np.random.default_rng(5)
    vids = {"A": [1, 4, 9], "B": [2, 3, 8]}
    scores = {v: rng.normal(size=(1, 3)).round(3).tolist() for v in vids}
    a = unified(vids, scores, 1, 10**9)
    b = unified(vids, scores, 1, 10**9)
    assert a == b
