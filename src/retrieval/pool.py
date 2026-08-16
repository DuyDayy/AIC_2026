"""
Rổ ứng viên — HỢP top riêng của từng nguồn
==========================================

Hai cách sinh ứng viên cho ⑤, và chúng hỏng theo hai kiểu ngược nhau.

**Hợp ĐIỂM** (`fuse` rồi lấy top): nguồn thị giác mang hệ số `1,0` còn OCR/ASR/vật thể
mang `0,09–0,13`, nên một khung chỉ có bằng chứng **thuần OCR** gần như không bao giờ nổi
lên. Đo được: 10 câu trong bộ 100 có video đúng **vắng mặt hoàn toàn** khỏi bài nộp, mà
9/10 trong số đó nằm ở hạng mốc 16–62 — với tới được, chỉ là không được với tới.

**Hợp ỨNG VIÊN** (module này): mỗi nguồn tự đề cử top `n` của riêng nó, hợp lại thành rổ.
OCR chỉ cần xếp một khung #1 *trong bảng của chính nó* là khung đó vào rổ, bất kể điểm
thị giác. Độ phủ video đúng ở ngân sách 16 mốc, riêng 10 câu nói trên: **10% → 30%**.

Cái giá, và vì sao module này KHÔNG tự xếp hạng
-----------------------------------------------
Nếu lấy rổ này rồi xếp hạng bằng vòng tròn theo hạng của từng nguồn thì **thua nặng**:
đo được `−0,1430` Final, vì chia đều ô đầu bảng cho 4 nguồn nghĩa là ba phần tư số ô giao
cho nguồn kém. Trọng số `0,089 / 0,132 / 0,106` **chính là tỉ lệ độ tin cậy đã rút từ dữ
liệu**, và xếp hạng vòng tròn vứt bỏ đúng thông tin ấy.

Nên rổ này chỉ có **một** việc: quyết định *ai được vào vòng trong*. Thứ hạng cuối do ⑤
rerank quyết — đó là tầng duy nhất thật sự nhìn vào pixel của khung. `provenance` trả kèm
để ⑤ và phần chẩn đoán biết mỗi ứng viên vào rổ nhờ nguồn nào.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from src.retrieval.sources import SourceScores

__all__ = ["PoolResult", "pool_mask", "union_pool"]


class PoolResult:
    """Rổ ứng viên kèm lai lịch.

    Attributes:
        rows: hàng khung trong rổ, **đã sắp theo điểm nền giảm dần** — chỉ để tất định
            và tiện đọc log; thứ hạng thật do ⑤ quyết.
        provenance: `{hàng: (tên nguồn đã đề cử, …)}`, theo thứ tự nguồn được truyền vào.
        per_source_n: số ứng viên mỗi nguồn thực sự đóng góp (sau khi bỏ khung không phủ).
    """

    __slots__ = ("rows", "provenance", "per_source_n")

    def __init__(self, rows: np.ndarray, provenance: dict[int, tuple[str, ...]],
                 per_source_n: dict[str, int]):
        self.rows = rows
        self.provenance = provenance
        self.per_source_n = per_source_n

    def __len__(self) -> int:
        return int(self.rows.shape[0])

    def __repr__(self) -> str:
        return (f"PoolResult({len(self)} ứng viên, "
                f"nguồn={dict(sorted(self.per_source_n.items()))})")


def union_pool(sources: Sequence[SourceScores], per_source: int,
               base: np.ndarray | None = None,
               weights: Mapping[str, float] | None = None,
               cap: int | None = None,
               ranges: Mapping[str, tuple[int, int]] | None = None,
               per_video: int | None = None) -> PoolResult:
    """Hợp top-`per_source` của từng nguồn thành một rổ ứng viên.

    Args:
        sources: các nguồn đã chấm. Khung **không được nguồn nào phủ** (`covered=False`)
            không bao giờ được nguồn đó đề cử — nếu không thì một nguồn thưa sẽ đề cử
            toàn khung điểm 0, đúng lỗi đã giết RRF.
        per_source: mỗi nguồn đề cử bao nhiêu khung.
        base: điểm nền để sắp rổ cho tất định. `None` thì tính bằng `fuse` với `weights`.
        weights: trọng số dùng khi `base is None`.
        cap: chặn trên kích thước rổ; `None` là không chặn. Cắt theo `base` giảm dần —
            nhưng **sau khi** mỗi nguồn đã có suất, nên nguồn yếu không bị xoá sạch.
        ranges: `{video_id: (đầu, cuối)}` của chỉ mục. Bắt buộc khi có `per_video`.
        per_video: **hạn ngạch mỗi nguồn được đề cử tối đa bao nhiêu khung của CÙNG một
            video.** `None` là không hạn.

    Vì sao cần `per_video`
    ----------------------
    Một nguồn có thể cho điểm **phẳng trong cả một video**: OCR khớp logo kênh, mà logo
    hiện ở mọi khung. Khi ấy nó không sai — nó chỉ **không định vị được**, và top-40 của
    nó là 40 khung tuỳ tiện của cùng một video. Đo được một câu như thế: video xếp hạng 1
    mà khung lệch **5.068 khung**.

    Hạn ngạch biến một nguồn phẳng từ "chiếm 40 suất bằng nhiễu" thành "bỏ một phiếu cho
    video này". Nó chữa cùng lúc hai bệnh: nguồn phẳng thôi làm ngập rổ, và rổ phủ được
    nhiều video hơn — đúng thứ 10 câu sai video đang thiếu.

    Returns:
        `PoolResult`.

    Raises:
        ValueError: `per_source < 1` · không có nguồn nào · các nguồn lệch số khung ·
            thiếu `base` lẫn `weights` · có `per_video` mà thiếu `ranges` · `per_video<1`.
    """
    if per_source < 1:
        raise ValueError(f"per_source phải ≥ 1, nhận {per_source}")
    if not sources:
        raise ValueError("không có nguồn nào — rổ rỗng phải là LỖI, không phải mặc định")
    if per_video is not None:
        if per_video < 1:
            raise ValueError(f"per_video phải ≥ 1, nhận {per_video}")
        if ranges is None:
            raise ValueError("có `per_video` thì phải có `ranges` để biết khung nào "
                             "thuộc video nào")
    n = sources[0].scores.shape[0]
    if any(s.scores.shape[0] != n for s in sources):
        raise ValueError("các nguồn lệch số khung — nguy cơ ghép nhầm vị trí")

    if base is None:
        if weights is None:
            raise ValueError("cần `base` hoặc `weights` để sắp rổ cho tất định")
        from src.retrieval.score_matrix import fuse
        base = fuse(sources, weights)
    base = np.asarray(base, dtype=np.float32)
    if base.shape[0] != n:
        raise ValueError(f"`base` có {base.shape[0]} khung, nguồn có {n}")

    vid_of = None
    if per_video is not None:
        vid_of = np.full(n, -1, dtype=np.int32)
        for i, (_v, (lo, hi)) in enumerate(sorted(ranges.items())):   # type: ignore[union-attr]
            vid_of[lo:hi] = i

    prov: dict[int, list[str]] = {}
    per_n: dict[str, int] = {}
    for s in sources:
        cov = np.flatnonzero(s.covered)
        if cov.size == 0:
            per_n[s.name] = 0
            continue
        # `z_normalize` để so trong CÙNG một nguồn là thừa (đơn điệu), nhưng dùng
        # `s.scores` thô là đúng: ta chỉ xếp hạng nội bộ nguồn, không so chéo nguồn.
        v = s.scores[cov]
        if vid_of is None:
            k = min(per_source, cov.size)
            pick = cov[np.argpartition(-v, k - 1)[:k]] if k < cov.size else cov
        else:
            # Sắp TOÀN BỘ phần được phủ, không cắt cửa sổ trước: một nguồn phẳng có thể
            # có hàng trăm khung cùng điểm trong một video, và cửa sổ hẹp sẽ khiến ta
            # trả về ít hơn `per_source` mà không biết vì sao.
            order = cov[np.argsort(-v, kind="stable")]
            cnt: dict[int, int] = {}
            chosen: list[int] = []
            for r in order.tolist():
                b = int(vid_of[r])
                if cnt.get(b, 0) >= per_video:
                    continue
                cnt[b] = cnt.get(b, 0) + 1
                chosen.append(r)
                if len(chosen) >= per_source:
                    break
            pick = np.asarray(chosen, dtype=np.int64)
        per_n[s.name] = int(pick.size)
        for r in pick.tolist():
            prov.setdefault(int(r), []).append(s.name)

    rows = np.fromiter(prov.keys(), dtype=np.int64, count=len(prov))
    rows = rows[np.argsort(-base[rows], kind="stable")]
    if cap is not None and rows.size > cap:
        rows = rows[:cap]
        keep = set(rows.tolist())
        prov = {r: v for r, v in prov.items() if r in keep}
    return PoolResult(rows, {r: tuple(v) for r, v in prov.items()}, per_n)


def pool_mask(rows: np.ndarray, n_frames: int) -> np.ndarray:
    """`rows` → mặt nạ boolean độ dài `n_frames`, để làm `covered` cho ⑤."""
    m = np.zeros(n_frames, dtype=bool)
    m[rows] = True
    return m
