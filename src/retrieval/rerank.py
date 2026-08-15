"""
Tầng ⑤ — RERANK BẰNG MẢNH CẮT: buộc màu và chi tiết vào ĐÚNG vật
==================================================================

Embedding một khung trộn màu của **mọi** thứ trong khung. Người mặc áo vàng đứng giữa
ruộng thì vector bị màu xanh của ruộng chi phối, và truy vấn *"áo màu vàng"* thua khung
có nhiều vàng ở chỗ khác.

Cắt vật ra rồi mã hoá riêng **buộc thuộc tính vào vật bằng cấu trúc**: khi mảnh cắt chỉ
chứa chiếc áo, màu của áo là màu của cả ảnh.

=============================================================================
VÌ SAO CẦN — ĐO ĐƯỢC Ở PHÍA CHỮ, KHÔNG PHẢI SUY ĐOÁN
=============================================================================

Hai câu giống hệt nhau, chỉ **đảo màu giữa hai vật**, rồi đo độ trùng 20 kết quả đầu:

    đảo màu giữa hai vật   cosine chữ 0,974   trùng top-20 0,35
    chỉ viết lại câu       cosine chữ 0,985   trùng top-20 0,85
    hai câu khác hẳn       cosine chữ 0,42    trùng top-20 0,00

Đảo màu chỉ làm vector chữ đổi **0,026** — gần bằng mức đổi khi chỉ viết lại câu. Tháp
chữ mã hoá `{áo, xe, đỏ, trắng}` như một **túi khái niệm**. Đây là lỗi *attribute
binding* của họ CLIP, và nó **có sẵn ở phía chữ**, nên không cách nào sửa được bằng cách
đổi phía ảnh nếu vẫn so cả khung.

[ĐO] chấm bằng mắt, 8 truy vấn màu × 6 khung đầu: toàn khung **26/48 = 54%** đúng màu,
rerank mảnh cắt **37/48 = 77%**, không ca nào tệ đi.

=============================================================================
MẢNH CẮT PHỦ MỘT PHẦN — NÊN NÓ LÀ MỘT `SourceScores`, KHÔNG PHẢI PHÉP SẮP LẠI
=============================================================================

Khung không có vật nào thuộc lớp đang hỏi thì **không có mảnh cắt**. Nếu coi điểm mảnh
cắt của chúng là 0 rồi cộng thẳng, mọi khung có mảnh cắt được lợi *bất kể liên quan* —
đúng lỗi đã giết RRF (−0,2927 FINAL).

Nên tầng này trả về `SourceScores(scores, covered)` và đi qua **cùng một phép chuẩn hoá
z** của ③. Khung không có mảnh cắt nhận đúng 0, bằng kỳ vọng của khung có mảnh cắt.

=============================================================================
GIỚI HẠN PHẢI BIẾT
=============================================================================

* **Trần là recall của tầng trước.** Rerank đổi thứ tự, không thêm ứng viên. [ĐO] hoa
  tím thật vốn đã nằm ở hạng 12, 3, 8 — rerank chỉ kéo chúng lên.
* **Không tạo ra được thứ không có.** *"Xe hơi màu vàng"* không cải thiện vì kho không
  có xe vàng.
* **Hộp sai thì mảnh cắt sai.** [ĐO] hộp `Helmet` của detector thực chất bao cả khuôn
  mặt, nên mảnh cắt mang màu da chứ không mang màu mũ.
* Con số 54% → 77% là **chấm bằng mắt trên 8 truy vấn**, người chấm cũng là người đề
  xuất phương pháp. Đủ để nói *đáng làm*, chưa đủ để công bố.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from src.retrieval.sources import FrameKey, SourceScores

# Hộp nhỏ hơn 1% diện tích khung thì mảnh cắt chỉ còn vài chục pixel — phóng to lên
# 336px là nội suy ra hư vô, và màu trung bình của nó là màu nhiễu nén.
MIN_AREA = 0.01

# Lấy mấy hộp lớn nhất mỗi khung. 2 là đủ cho ca "hai người, một đỏ một xanh"; nhiều
# hơn thì chi phí mã hoá tăng tuyến tính mà hộp thứ ba thường đã quá nhỏ.
MAX_PER_FRAME = 2

# Nới hộp trước khi cắt: hộp detector thường bó sát, và vài phần trăm ngữ cảnh giúp
# model nhận ra ĐÓ LÀ CÁI GÌ thay vì chỉ thấy một mảng màu.
PAD = 0.06


@dataclass(frozen=True)
class CropRef:
    """Một mảnh cắt cần mã hoá. `row` là chỉ số hàng trong `FlatIndex`."""

    row: int
    video_id: str
    n: int
    entity: str
    box: tuple[float, float, float, float]      # ymin, xmin, ymax, xmax
    area: float

    def pixel_box(self, width: int, height: int) -> tuple[int, int, int, int]:
        """`(left, top, right, bottom)` đã nới `PAD` và kẹp trong khung."""
        y0, x0, y1, x1 = self.box
        mx, my = (x1 - x0) * PAD, (y1 - y0) * PAD
        return (int(max(0.0, x0 - mx) * width), int(max(0.0, y0 - my) * height),
                int(min(1.0, x1 + mx) * width), int(min(1.0, y1 + my) * height))


def collect_crops(rows: Sequence[int], ids: Sequence[FrameKey],
                  objects_root: str | Path,
                  classes: Iterable[str] | None = None,
                  max_per_frame: int = MAX_PER_FRAME,
                  min_area: float = MIN_AREA) -> list[CropRef]:
    """
    `[CropRef]` cho các khung ứng viên, lấy hộp lớn nhất trước.

    `classes=None` ⟹ **mọi lớp**. Truyền danh sách lớp khi truy vấn nói rõ vật nào —
    nó cắt số mảnh phải mã hoá xuống nhiều lần. Nhưng mặc định phải là "mọi lớp", vì
    đoán sai lớp thì mảnh đúng không bao giờ được xét.
    """
    root = Path(objects_root)
    want = set(classes) if classes is not None else None
    out: list[CropRef] = []
    for r in rows:
        vid, n = ids[r]
        p = root / vid / f"{int(n):03d}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        cand = []
        for e, b in zip(d["detection_class_entities"], d["detection_boxes"]):
            if want is not None and e not in want:
                continue
            y0, x0, y1, x1 = (float(t) for t in b)
            a = (y1 - y0) * (x1 - x0)
            if a >= min_area:
                cand.append((a, e, (y0, x0, y1, x1)))
        cand.sort(key=lambda x: -x[0])
        for a, e, box in cand[:max_per_frame]:
            out.append(CropRef(int(r), vid, int(n), e, box, float(a)))
    return out


def crop_scores(n_frames: int, refs: Sequence[CropRef], crop_vecs: np.ndarray,
                query_vec: np.ndarray, name: str = "crop") -> SourceScores:
    """
    `[CropRef]` + vector mảnh cắt + vector truy vấn → `SourceScores` trên MỌI khung.

    Một khung có nhiều mảnh thì lấy **max**: đủ để một vật khớp là khung đáng lên. Lấy
    trung bình sẽ phạt khung đông vật, mà đông vật không phải lỗi.

    Khung không có mảnh nào ⟹ `covered=False`, điểm 0. Xem docstring module về vì sao
    phân biệt đó là bắt buộc chứ không phải tiện.

    Raises:
        ValueError: số vector ≠ số `CropRef` · số chiều lệch · `n_frames` quá nhỏ.
    """
    cv = np.asarray(crop_vecs, dtype=np.float32)
    q = np.asarray(query_vec, dtype=np.float32).ravel()
    if cv.ndim != 2 or cv.shape[0] != len(refs):
        raise ValueError(f"{cv.shape[0] if cv.ndim == 2 else '?'} vector ≠ {len(refs)} "
                         f"mảnh cắt — hai mảng song song lệch thì điểm gắn nhầm khung")
    if len(refs) and cv.shape[1] != q.shape[0]:
        raise ValueError(f"mảnh cắt {cv.shape[1]} chiều ≠ truy vấn {q.shape[0]} chiều")
    s = np.zeros(n_frames, dtype=np.float32)
    cov = np.zeros(n_frames, dtype=bool)
    if not len(refs):
        return SourceScores(name, s, cov)
    sims = cv @ q
    for ref, sim in zip(refs, sims):
        if ref.row >= n_frames or ref.row < 0:
            raise ValueError(f"CropRef trỏ hàng {ref.row}, ngoài chỉ mục {n_frames} khung")
        if not cov[ref.row] or sim > s[ref.row]:
            s[ref.row] = sim
        cov[ref.row] = True
    return SourceScores(name, s, cov)
