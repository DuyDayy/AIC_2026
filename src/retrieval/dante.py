"""
Tầng ④ — DANTE: quy hoạch động căn N mốc theo thứ tự thời gian
================================================================

Nguồn: arXiv:2512.13169. Hệ thức truy hồi:

    DP[i,t] = S[i,t] + max_{τ < t} ( DP[i−1,τ] − λ·(t − τ) )

Đọc: mốc `i` xảy ra tại khung `t` được điểm `S[i,t]`, cộng đường tốt nhất kết thúc ở
mốc trước tại một khung `τ` **thực sự sớm hơn**, trừ hình phạt tỉ lệ với khoảng cách
thời gian giữa hai mốc.

Hai ràng buộc nằm luôn trong công thức, không cần kiểm riêng:

* `τ < t` **ngặt** ⟹ thứ tự thời gian đúng, và không mốc nào dùng chung một khung.
* `−λ(t−τ)` ⟹ các mốc phải **gần nhau**. Đo được thì phạt này **có hại** ở kho này;
  `λ = 0` — xem khối bình luận ở `DEFAULT_LAMBDA` để biết cơ chế.

=============================================================================
`O(N·T)` — VÌ SAO KHÔNG PHẢI `O(N·T²)`
=============================================================================

Viết thẳng thì mỗi ô phải quét mọi `τ`, thành `O(N·T²)`. Tách `t` ra khỏi max:

    max_{τ<t} ( DP[i−1,τ] − λt + λτ )  =  [ max_{τ<t} ( DP[i−1,τ] + λτ ) ] − λt

Ngoặc vuông **không phụ thuộc `t`** ngoài chặn trên, nên nó là một **max luỹ tiến**
tính một lượt. Mỗi mốc còn `O(T)`.

=============================================================================
TRỤC THỜI GIAN LÀ MILI-GIÂY, KHÔNG PHẢI `n`, CŨNG KHÔNG PHẢI `frame_idx`
=============================================================================

`n` là số thứ tự keyframe và **bước của nó không đều**: p10 = 19 khung, p90 = 105
khung. Phạt `λ(t−τ)` trên trục `n` nghĩa là coi hai keyframe liền nhau luôn cách nhau
như nhau — sai theo hệ số hơn 5 lần.

`frame_idx` đều trong MỘT video nhưng **không so được giữa các video**: 25 fps và 30
fps cho cùng một giây ra số khung khác nhau. Vì `λ` là **một hằng số chung cho mọi
video**, trục phải mang cùng đơn vị vật lý ở mọi nơi ⟹ **mili-giây**.

Đơn vị của `λ`: *điểm trên giây*. `λ = 0,1` nghĩa là hai mốc cách nhau 10 giây bị trừ
1,0 — cùng cỡ **một độ lệch chuẩn** của điểm sau chuẩn hoá z ở ③, tức phạt rất nặng.
Chính vì thang đó mà λ mặc định là 0.

=============================================================================
`N = 1` SUY BIẾN, KHÔNG CÓ NHÁNH RIÊNG CHO KIS
=============================================================================

`N = 1` thì vòng lặp mốc không chạy lần nào, `DP[0] = S[0]`, và kết quả là
`argmax_t S[0,t]` — đúng bằng KIS. Nên KIS và TRAKE **dùng chung một engine**; khác
biệt duy nhất là số hàng của `S`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NEG = -np.inf

# =============================================================================
# λ = 0 — ĐÃ QUÉT, VÀ MỌI λ > 0 ĐỀU TỆ HƠN
# =============================================================================
#
# [ĐO] 6 truy vấn TRAKE tự sinh (`data/eval/trake_gt.json`), span 6→45 giây:
#
#     λ       hạng video (trung vị)   mốc đúng CHẶT   mốc đúng ±2 giây
#     0,0                         6            0,50               0,61   ← chọn
#     0,01                       24            0,50               0,56
#     0,05                      206            0,11               0,39
#     0,2                       133            0,11               0,39
#     1,0                       131            0,06               0,17
#
# CƠ CHẾ HỎNG, không phải chuyện chỉnh số. Ở λ = 1,0 **mọi** truy vấn đều trả về
# 0,4s / 1,0s / 1,8s — ba keyframe đầu của một video nào đó. Vì `−λ(t−τ)` là hàm
# của khoảng cách TUYỆT ĐỐI, nó cộng cùng một khoản handicap cho mọi đường có span
# lớn, ở MỌI video. Trong 873 video luôn tồn tại vài video có ba khung kề nhau ghi
# điểm tàm tạm, và chúng thắng chỉ nhờ span ~1 giây.
#
# Nói cách khác λ can thiệp vào việc XẾP HẠNG VIDEO, trong khi nó chỉ được thiết kế
# để định hình đường TRONG một video. Kể cả khi đã cố định đúng video, λ > 0 vẫn hạ
# độ chính xác mốc từ 0,50 xuống 0,11 vì nó dồn ba mốc vào ba khung liền nhau.
#
# CẢNH BÁO: 6 truy vấn, cả 6 từ chương trình dạy nấu ăn, các mốc đều nằm trong một
# chuỗi cảnh liền. λ có thể có ích cho truy vấn mà các mốc rải xa và có nhiều mồi
# nhiễu. Ý tưởng chưa thử: thay phạt tuyến tính bằng **chặn cứng span**, vì chặn
# cứng không bóp méo điểm nên không đụng tới xếp hạng video.
DEFAULT_LAMBDA = 0.0


@dataclass(frozen=True)
class Path:
    """Một phương án: `cols[i]` là chỉ số cột (trong lát video) của mốc thứ `i`."""

    score: float
    cols: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.cols)


def _running_argmax(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    `(M, A)` với `M[t] = max_{τ≤t} v[τ]` và `A[t]` là một chỉ số đạt max đó.

    `np.maximum.accumulate` cho `M`; `A` lấy bằng cách chỉ giữ chỉ số ở những vị trí
    LẬP KỶ LỤC rồi cũng luỹ tiến — chỗ không lập kỷ lục sẽ kế thừa chỉ số trước.
    """
    m = np.maximum.accumulate(v)
    idx = np.arange(v.shape[0])
    a = np.maximum.accumulate(np.where(v >= m, idx, 0))
    return m, a


def dante(S: np.ndarray, times_ms: np.ndarray, lam: float = DEFAULT_LAMBDA) -> Path:
    """
    Chạy DP trên **một video**. `S` là `(N, T)`, `times_ms` là `(T,)` TĂNG DẦN.

    Trả đường tốt nhất. `T < N` ⟹ không tồn tại đường hợp lệ (cần N khung phân biệt
    theo thứ tự) ⟹ `score = -inf`, `cols = ()`.

    Raises:
        ValueError: `S` không phải 2 chiều · `times_ms` lệch độ dài · `times_ms` không
            tăng dần (thứ tự sai thì ràng buộc `τ < t` mất nghĩa) · `lam` âm.
    """
    S = np.asarray(S, dtype=np.float64)
    t_ms = np.asarray(times_ms, dtype=np.float64)
    if S.ndim != 2:
        raise ValueError(f"S phải 2 chiều (N,T), nhận {S.shape}")
    n, T = S.shape
    if t_ms.shape != (T,):
        raise ValueError(f"times_ms {t_ms.shape} ≠ số cột của S ({T},)")
    if lam < 0:
        raise ValueError(f"lam phải ≥ 0, nhận {lam}")
    if T and np.any(np.diff(t_ms) < 0):
        raise ValueError("times_ms phải TĂNG DẦN — DP giả định cột sắp theo thời gian")
    if n == 0 or T < n:
        return Path(NEG, ())

    # Dời gốc thời gian về 0 và đổi sang GIÂY: hiệu (t − τ) không đổi khi dời, còn
    # `λ·t` với t cỡ hàng triệu mili-giây thì mất chính xác dấu phẩy động.
    t = (t_ms - t_ms[0]) / 1000.0

    dp = np.full((n, T), NEG)
    back = np.zeros((n, T), dtype=np.int64)
    dp[0] = S[0]
    for i in range(1, n):
        prev = dp[i - 1] + lam * t                 # ngoặc vuông ở docstring
        m, a = _running_argmax(prev)
        # Cần max trên τ ≤ t−1 (NGẶT), nên dịch phải một ô; cột 0 không có τ nào.
        ms = np.empty(T)
        ms[0] = NEG
        ms[1:] = m[:-1]
        ab = np.zeros(T, dtype=np.int64)
        ab[1:] = a[:-1]
        with np.errstate(invalid="ignore"):
            dp[i] = np.where(np.isfinite(ms), S[i] + ms - lam * t, NEG)
        back[i] = ab

    end = int(np.argmax(dp[n - 1]))
    best = float(dp[n - 1, end])
    if not np.isfinite(best):
        return Path(NEG, ())
    cols = [end]
    for i in range(n - 1, 0, -1):
        cols.append(int(back[i, cols[-1]]))
    return Path(best, tuple(reversed(cols)))


def dante_over_videos(S_all: np.ndarray, ranges: dict[str, tuple[int, int]],
                      times_ms: np.ndarray, lam: float = DEFAULT_LAMBDA,
                      videos: list[str] | None = None) -> list[tuple[str, Path]]:
    """
    Chạy `dante` trên từng lát video, trả `[(video_id, Path)]` sắp giảm dần theo điểm.

    `S_all` là `(N, tổng số khung)` theo đúng thứ tự hàng của chỉ mục phẳng; `ranges`
    là `{video → (lo, hi)}` của `FlatIndex`. `Path.cols` là chỉ số **trong lát**, nên
    khung tuyệt đối là `lo + col`.

    `videos=None` thì quét mọi video. Truyền danh sách ngắn để chỉ chạy trên tập ứng
    viên — DP là `O(N·T)` nên quét cả 873 video vẫn rẻ, nhưng khi đã có tầng lọc thì
    không có lý do chạy thừa.

    **`n` KHÔNG sắp theo thời gian ở mọi video.** [ĐO] 20/873 video có ít nhất một chỗ
    `pts_time` tụt lại khi `n` tăng (tụt 0,0–0,3 giây, 1 chỗ mỗi video). Chỉ mục phẳng
    sắp theo `(video_id, n)`, nên với các video đó thứ tự cột **không** là thứ tự thời
    gian — mà `τ < t` của DP nói về THỜI GIAN.

    Nên ở đây sắp lại từng lát theo `times_ms` rồi ánh xạ `cols` ngược về chỉ số trong
    lát gốc. `dante()` giữ nguyên cổng chặn nghiêm ngặt: nó có quyền tin đầu vào đã
    sắp, và mọi chỗ gọi khác vẫn bị bắt lỗi.
    """
    S_all = np.asarray(S_all)
    if S_all.ndim != 2:
        raise ValueError(f"S_all phải 2 chiều (N, tổng khung), nhận {S_all.shape}")
    out: list[tuple[str, Path]] = []
    for vid in (videos if videos is not None else ranges.keys()):
        lo, hi = ranges[vid]
        t = np.asarray(times_ms[lo:hi], dtype=np.float64)
        order = np.argsort(t, kind="stable")
        p = dante(S_all[:, lo:hi][:, order], t[order], lam)
        if not np.isfinite(p.score):
            continue
        out.append((vid, Path(p.score, tuple(int(order[c]) for c in p.cols))))
    out.sort(key=lambda x: -x[1].score)
    return out
