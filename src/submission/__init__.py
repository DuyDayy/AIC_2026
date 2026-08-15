"""Tầng tối ưu nộp bài — Định lý 1/2/3/4 của kế hoạch AIC 2026."""

from src.submission.coverage import (
    grid_hits,
    guaranteed_span,
    half_widths,
    hit_probability,
    optimal_placement,
    uniform_grid,
)
from src.submission.kbest import (
    AlignedTuple,
    apply_pacing_penalty,
    best_alignment,
    k_best_alignments,
)

__all__ = [
    "AlignedTuple",
    "apply_pacing_penalty",
    "best_alignment",
    "grid_hits",
    "guaranteed_span",
    "half_widths",
    "hit_probability",
    "k_best_alignments",
    "optimal_placement",
    "uniform_grid",
]
