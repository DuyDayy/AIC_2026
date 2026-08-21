"""
Mã hoá bằng jina-clip-v2 — hai tháp, một không gian
====================================================

Tháp ẢNH và tháp VĂN BẢN của `jina-clip-v2` nhả vector vào **cùng một không gian**, nên
`cos(vector_câu_hỏi, vector_khung)` là một con số có nghĩa. Đó là thứ mở khoá toàn bộ
tầng truy vấn: không có nó thì không có ma trận `S`, không có DANTE.

    tháp ẢNH     304M tham số   chạy MỘT LẦN offline cho 173.426 keyframe
    tháp VĂN BẢN 561M tham số   chạy 1 lần mỗi truy vấn (~300 lần cả cuộc thi)

Chọn model này thay CLIP gốc vì tháp văn bản đa ngữ **89 thứ tiếng, có tiếng Việt** —
truy vấn đi thẳng vào không gian chung, không qua một bước dịch nào (mỗi lần dịch là
một lần mất thông tin).

=============================================================================
MATRYOSHKA: CẮT CHIỀU RỒI MỚI CHUẨN HOÁ — KHÔNG PHẢI NGƯỢC LẠI
=============================================================================

Model huấn luyện kiểu Matryoshka: các chiều được sắp theo **độ quan trọng giảm dần**,
nên 512 chiều đầu đã mang gần hết thông tin của 1024 chiều. Nhờ vậy đổi số chiều là
**cắt đuôi**, không phải mã hoá lại 173.426 khung.

Nhưng thứ tự hai phép toán KHÔNG hoán đổi được, và đây là chỗ sai âm thầm:

    chuẩn hoá(1024) rồi cắt → 512     ‖v‖ ≈ 0.7–0.9, KHÔNG phải 1
    cắt → 512 rồi chuẩn hoá           ‖v‖ = 1  ✓

Vì `‖v[:512]‖² = Σ_{i<512} v_i² ≤ Σ_{i<1024} v_i² = 1`, dấu bằng chỉ xảy ra khi đuôi
toàn số 0. Vector không đơn vị làm hỏng mọi thứ ở hạ nguồn cùng một lúc: tích vô hướng
không còn là cosine, và độ dài vector (một đại lượng vô nghĩa về ngữ nghĩa) lẻn vào
điểm số — khung nào tình cờ có đuôi nhỏ sẽ được chấm cao hơn.

`truncate_and_normalize` làm đúng thứ tự, và `test_jina_encoder.py` khoá nó lại.

=============================================================================
VÌ SAO PHẦN LÕI TÁCH KHỎI MODEL
=============================================================================

Mọi phép toán ở đây (`l2_normalize`, `truncate_and_normalize`) là **thuần tuý** và test
được mà không cần tải 0.9B tham số. `JinaEncoder` chỉ là lớp vỏ gọi model, nạp lười.
Cùng lý do `assign_embeddings` tách khỏi vòng decode video: chỗ dễ sai nhất là phép ghép
và phép chuẩn hoá, không phải lời gọi model.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - chỉ để type checker, không nạp lúc chạy
    from PIL.Image import Image

logger = logging.getLogger(__name__)

MODEL_NAME = "jinaai/jina-clip-v2"

# Số chiều model nhả ra trước khi cắt.
FULL_DIM = 1024

# Chiều mặc định của chỉ mục. 512 vì đo được: 173.426 × 512 fp16 = 177,6 MB, quét phẳng
# 5,4 ms/truy vấn — nghĩa là KHÔNG cần ANN index, nên không có tham số xấp xỉ nào phải
# chỉnh và không mất recall. Model cho phép xuống tới 64.
DEFAULT_DIM = 512

# Ngưỡng kiểm "đã chuẩn hoá chưa". fp16 lưu ~3 chữ số thập phân có nghĩa, nên sau vòng
# fp32 → fp16 → fp32 chuẩn ‖v‖ lệch cỡ 1e-3. Đặt 1e-2 để không báo động giả, vẫn đủ chặt
# để bắt lỗi thật (quên chuẩn hoá cho ‖v‖ lệch hàng chục phần trăm).
NORM_TOLERANCE = 1e-2


def l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Chuẩn hoá từng HÀNG về độ dài 1, trả mảng mới (không sửa tại chỗ).

    Sau phép này tích vô hướng **chính là** cosine, nên cả phép tìm kiếm rút về một
    lệnh `emb @ q`. Chuẩn hoá lúc GHI chứ không lúc đọc: làm lúc đọc là lặp lại
    173.426 phép chia cho mỗi truy vấn.

    `eps` chặn chia cho 0 với hàng toàn số 0 — hàng đó ra vector 0, và vector 0 cho
    cosine 0 với mọi truy vấn, tức "không khớp gì", đúng nghĩa mong muốn.
    """
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim != 2:
        raise ValueError(f"cần ma trận 2 chiều (N, D), nhận shape {mat.shape}")
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, eps)


def truncate_and_normalize(mat: np.ndarray, dim: int = DEFAULT_DIM) -> np.ndarray:
    """
    Cắt Matryoshka xuống `dim` chiều **rồi mới** chuẩn hoá. Xem docstring module.

    Raises:
        ValueError: `dim` lớn hơn số chiều đang có — cắt không thể nới rộng, và đệm
            số 0 sẽ tạo ra vector trông hợp lệ mà sai nghĩa.
    """
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim != 2:
        raise ValueError(f"cần ma trận 2 chiều (N, D), nhận shape {mat.shape}")
    if dim <= 0:
        raise ValueError(f"dim phải dương, nhận {dim}")
    if dim > mat.shape[1]:
        raise ValueError(
            f"không cắt được {mat.shape[1]} chiều thành {dim} — Matryoshka chỉ bỏ đuôi, "
            f"không nới rộng"
        )
    return l2_normalize(mat[:, :dim])


def is_normalized(mat: np.ndarray, tol: float = NORM_TOLERANCE) -> bool:
    """`True` nếu MỌI hàng có độ dài 1 trong sai số `tol`. Hàng toàn 0 coi là hợp lệ."""
    mat = np.asarray(mat, dtype=np.float32)
    if mat.size == 0:
        return True
    norms = np.linalg.norm(mat, axis=1)
    zero = norms < tol
    return bool(np.all(zero | (np.abs(norms - 1.0) <= tol)))


class JinaEncoder:
    """
    Lớp vỏ quanh `jina-clip-v2`. Nạp model **lười** — dựng đối tượng không tải gì.

    Nạp lười để bên gọi lập kế hoạch, phân mảnh, kiểm tra đường dẫn xong xuôi rồi mới
    trả tiền cho lần tải 0.9B tham số. Đã mắc lỗi ngược lại một lần ở đường OWLv2: model
    nạp trong worker trước khi kiểm dữ liệu, nên mọi lỗi cấu hình đều lộ ra SAU khi đã
    khởi động xong container.
    """

    #: Cặp sha mà vector KHUNG đã dùng — `artifacts/embed/embed/manifest.json`, và
    #: `PipelineConfig` của `frame_extracting` dùng đúng cặp này.
    MODEL_SHA = "e10d47f5691d0454a0fb5d13f46f2199b74cb436"
    CODE_SHA = "39e6a55ae971b59bea6e44675d237c99762e7ee2"

    def __init__(self, model_name: str = MODEL_NAME, dim: int = DEFAULT_DIM,
                 device: str | None = None, revision: str | None = None,
                 code_revision: str | None = None) -> None:
        if dim > FULL_DIM:
            raise ValueError(f"dim {dim} vượt {FULL_DIM} chiều model nhả ra")
        self.model_name = model_name
        self.dim = dim
        self.device = device
        self.revision = revision or self.MODEL_SHA
        self.code_revision = code_revision or self.CODE_SHA
        self._model = None

    @property
    def model(self):
        """Nạp lần đầu khi thật sự cần."""
        if self._model is None:
            from transformers import AutoModel

            logger.info(f"nạp {self.model_name} (lần đầu, có thể mất vài phút)")
            # GHIM REVISION — xem khối bình luận ở `MODEL_SHA` trong `scripts/run.py`.
            # Không ghim thì truy vấn và khung có thể đến từ hai bản mô hình khác nhau,
            # và sai lệch đó KHÔNG có triệu chứng nào.
            self._model = AutoModel.from_pretrained(
                self.model_name, trust_remote_code=True,
                revision=self.revision, code_revision=self.code_revision)
            if self.device:
                self._model = self._model.to(self.device)
            self._model.eval()
        return self._model

    def encode_images(self, images: Sequence["Image"], batch_size: int = 32) -> np.ndarray:
        """
        `(N, dim)` đã cắt và chuẩn hoá — tháp ẢNH, chạy offline.

        Trả `(0, dim)` cho danh sách rỗng thay vì `(0, 0)`, để `np.concatenate` với các
        lô khác không nổ vì lệch chiều. Lô rỗng là chuyện thường khi phân mảnh.
        """
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = self.model.encode_image(list(images), batch_size=batch_size)
        return truncate_and_normalize(np.asarray(out, dtype=np.float32), self.dim)

    def encode_texts(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """
        `(N, dim)` đã cắt và chuẩn hoá — tháp VĂN BẢN, chạy lúc truy vấn.

        Dùng cho CÂU HỎI, không dùng cho OCR/ASR/object: ba nguồn đó khớp bằng BM25 ở
        phía chỉ mục. Nhúng chúng là một đường thứ ba, và là một khoản THÊM phải tự
        chứng minh bằng đo đạc trước — xem `CONG_NGHE_TRUY_VAN.md`.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = self.model.encode_text(list(texts), batch_size=batch_size)
        return truncate_and_normalize(np.asarray(out, dtype=np.float32), self.dim)
