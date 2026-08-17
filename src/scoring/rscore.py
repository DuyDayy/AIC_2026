"""
R-Score & Final Score — cài đặt nguyên văn công thức chấm điểm AIC 2026
========================================================================

Nguồn: "Thong tin vong So tuyen AIC2026.pdf", mục 2.

  Textual KIS   R(rᵢ) = I(vᵢ = GTᵥ ∧ idᵢ ∈ [s, e])
  Q&A           R(rᵢ) = I(vᵢ = GTᵥ ∧ idᵢ ∈ [s, e] ∧ aᵢ = GTₐ)
  TRAKE         R(rᵢ) = 0                                    nếu vᵢ ≠ GTᵥ
                R(rᵢ) = (1/N)·Σⱼ I(id_{i,j} ∈ [sⱼ, eⱼ])      nếu vᵢ = GTᵥ

  R@k   = max_{1≤i≤k} R(rᵢ),  k ∈ {1, 5, 20, 50, 100}
  Final = (1/5)·Σ_k R@k

Module này là HÀM MỤC TIÊU của toàn hệ thống: allocator (Định lý 2/4) tối ưu
chính con số `final_score` trả về ở đây. Mọi thay đổi phải kèm test.

Lưu ý về biên:
  - Đoạn [s, e] là ĐÓNG hai đầu (PDF: "505 ∈ [500, 510] → Đúng").
  - Nộp ít hơn k câu trả lời: R@k lấy max trên số câu thực có (PDF không cấm
    nộp thiếu; thiếu chỉ làm mất cơ hội, không phải lỗi).
  - Nộp 0 câu trả lời: mọi R@k = 0.
"""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

logger = logging.getLogger(__name__)

# Các mốc xếp hạng dùng cho Final Score (PDF mục 2.2).
K_THRESHOLDS: tuple[int, ...] = (1, 5, 20, 50, 100)

# Số câu trả lời tối đa mỗi truy vấn (PDF mục 2). Allocator dùng làm ngân sách B.
MAX_ANSWERS: int = 100


# ============================================================
# KIỂU DỮ LIỆU
# ============================================================


@dataclass(frozen=True)
class Interval:
    """Đoạn khung hình đáp án [s, e], đóng hai đầu."""

    s: int
    e: int

    def __post_init__(self) -> None:
        if self.e < self.s:
            raise ValueError(f"Interval rỗng: [{self.s}, {self.e}]")

    def contains(self, frame_id: int) -> bool:
        return self.s <= frame_id <= self.e

    @property
    def length(self) -> int:
        """L = e − s + 1. Đây là `L` trong Định lý 1 (phủ lưới)."""
        return self.e - self.s + 1


@dataclass(frozen=True)
class KISAnswer:
    video_id: str
    frame_id: int


@dataclass(frozen=True)
class QAAnswer:
    video_id: str
    frame_id: int
    answer: str


@dataclass(frozen=True)
class TrakeAnswer:
    video_id: str
    frame_ids: tuple[int, ...]


@dataclass(frozen=True)
class KISGroundTruth:
    video_id: str
    window: Interval


@dataclass(frozen=True)
class QAGroundTruth:
    video_id: str
    window: Interval
    # Tập đáp án được chấp nhận (PDF: "5" hoặc "Năm" đều đúng).
    accepted_answers: frozenset[str]


@dataclass(frozen=True)
class TrakeGroundTruth:
    video_id: str
    windows: tuple[Interval, ...]

    @property
    def n_moments(self) -> int:
        """N — tổng số khoảnh khắc trong truy vấn."""
        return len(self.windows)


Answer = KISAnswer | QAAnswer | TrakeAnswer
GroundTruth = KISGroundTruth | QAGroundTruth | TrakeGroundTruth


# ============================================================
# SO KHỚP ĐÁP ÁN Q&A
# ============================================================


def normalize_answer(text: str) -> str:
    """
    Chuẩn hoá chuỗi đáp án trước khi so khớp: bỏ dấu cách thừa, hạ chữ thường,
    chuẩn hoá Unicode NFC (tiếng Việt có tổ hợp dấu nhiều cách biểu diễn).
    """
    return unicodedata.normalize("NFC", " ".join(text.split())).casefold()


def exact_answer_match(predicted: str, accepted: frozenset[str]) -> bool:
    """
    So khớp mặc định: bằng nhau sau chuẩn hoá.

    PDF yêu cầu khớp "về mặt ngữ nghĩa" — việc đó do giám khảo quyết định, ta
    không mô phỏng được. Đây là cận DƯỚI bi quan: dùng để mô phỏng chấm điểm
    (V5), luôn cho điểm ≤ điểm thật. Muốn dùng LLM-judge thì truyền hàm khác
    qua tham số `answer_match`.
    """
    norm = normalize_answer(predicted)
    return any(norm == normalize_answer(a) for a in accepted)


AnswerMatcher = Callable[[str, frozenset[str]], bool]


# ============================================================
# R-SCORE
# ============================================================


def r_score(
    answer: Answer,
    gt: GroundTruth,
    *,
    answer_match: AnswerMatcher = exact_answer_match,
) -> float:
    """
    Điểm Tương Quan của MỘT câu trả lời. Trả về giá trị trong [0, 1].

    Raises:
        TypeError: nếu kiểu answer và kiểu ground-truth không cùng loại truy vấn.
    """
    if isinstance(answer, KISAnswer) and isinstance(gt, KISGroundTruth):
        return float(answer.video_id == gt.video_id and gt.window.contains(answer.frame_id))

    if isinstance(answer, QAAnswer) and isinstance(gt, QAGroundTruth):
        return float(
            answer.video_id == gt.video_id
            and gt.window.contains(answer.frame_id)
            and answer_match(answer.answer, gt.accepted_answers)
        )

    if isinstance(answer, TrakeAnswer) and isinstance(gt, TrakeGroundTruth):
        # Điều kiện tiên quyết: sai video ⟹ 0 điểm ngay lập tức.
        if answer.video_id != gt.video_id:
            return 0.0
        n = gt.n_moments
        if n == 0:
            raise ValueError("TrakeGroundTruth phải có ít nhất 1 khoảnh khắc")
        # Nộp thiếu/thừa mốc: chỉ chấm trên N mốc mà đáp án quy định. Mốc thiếu
        # coi như trượt — mẫu số luôn là N (PDF: "N là tổng số khoảnh khắc
        # trong truy vấn", không phải số frame ta nộp).
        hits = sum(
            1
            for j, window in enumerate(gt.windows)
            if j < len(answer.frame_ids) and window.contains(answer.frame_ids[j])
        )
        return hits / n

    raise TypeError(f"Không khớp loại truy vấn: {type(answer).__name__} vs {type(gt).__name__}")


# ============================================================
# R@k VÀ FINAL SCORE
# ============================================================


def r_at_k(r_scores: Sequence[float], k: int) -> float:
    """R@k = max trong k câu trả lời ĐẦU TIÊN. Nộp thiếu thì lấy max phần có."""
    if k < 1:
        raise ValueError(f"k phải ≥ 1, nhận {k}")
    prefix = r_scores[:k]
    return max(prefix) if prefix else 0.0


@dataclass(frozen=True)
class ScoreReport:
    """Kết quả chấm điểm một truy vấn, kèm chẩn đoán."""

    final: float
    per_k: Mapping[int, float]
    r_scores: tuple[float, ...]

    @property
    def best(self) -> float:
        """max R-Score trên TOÀN BỘ danh sách — cận trên của Final (Định lý 2)."""
        return max(self.r_scores) if self.r_scores else 0.0

    @property
    def best_rank(self) -> int | None:
        """Vị trí (1-based) của câu trả lời tốt nhất. None nếu không có câu nào."""
        if not self.r_scores:
            return None
        return max(range(len(self.r_scores)), key=lambda i: self.r_scores[i]) + 1

    @property
    def ranking_loss(self) -> float:
        """
        Phần điểm mất do XẾP HẠNG sai, không phải do tìm sai: `best − final`.

        Chẩn đoán quan trọng (Định lý 2): nếu `ranking_loss` lớn thì vấn đề nằm
        ở calibration (câu đúng bị xếp sau), không phải ở coverage. Nếu
        `ranking_loss ≈ 0` mà `final` vẫn thấp thì vấn đề là coverage — phải
        sửa tầng cắt frame, không phải tầng xếp hạng.
        """
        return self.best - self.final


def final_score(
    answers: Sequence[Answer],
    gt: GroundTruth,
    *,
    answer_match: AnswerMatcher = exact_answer_match,
    k_thresholds: Sequence[int] = K_THRESHOLDS,
    max_answers: int = MAX_ANSWERS,
) -> ScoreReport:
    """
    Điểm Cuối Cùng cho một truy vấn: `Final = (1/5)·Σ_k R@k`.

    Args:
        answers: danh sách câu trả lời ĐÃ SẮP THEO THỨ TỰ NỘP (rank 1 trước).
        gt: đáp án của BTC.
        answer_match: hàm so khớp answer cho Q&A.
        k_thresholds: các mốc k. Mặc định (1, 5, 20, 50, 100) theo PDF.
        max_answers: cắt bớt nếu nộp quá ngân sách. Mặc định 100 theo PDF.

    Returns:
        ScoreReport — có `final`, `per_k`, và chẩn đoán `ranking_loss`.
    """
    if len(answers) > max_answers:
        logger.warning(
            "Nộp %d câu trả lời, vượt ngân sách %d — cắt bớt phần dư.",
            len(answers),
            max_answers,
        )
        answers = answers[:max_answers]

    scores = tuple(r_score(a, gt, answer_match=answer_match) for a in answers)
    per_k = {k: r_at_k(scores, k) for k in k_thresholds}
    final = sum(per_k.values()) / len(per_k)
    return ScoreReport(final=final, per_k=per_k, r_scores=scores)


def slot_weight(rank: int, k_thresholds: Sequence[int] = K_THRESHOLDS) -> float:
    """
    Trọng số thực của slot thứ `rank` (1-based) — Định lý 2.

        w(i) = |{k : k ≥ i}| / |k_thresholds|

    Với mốc mặc định: slot 1 → 1.0; slot 2–5 → 0.8; 6–20 → 0.6; 21–50 → 0.4;
    51–100 → 0.2. Allocator dùng hàm này để quyết định đặt gì ở đâu.
    """
    if rank < 1:
        raise ValueError(f"rank phải ≥ 1, nhận {rank}")
    return sum(1 for k in k_thresholds if k >= rank) / len(k_thresholds)
