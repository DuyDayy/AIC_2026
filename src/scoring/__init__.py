"""Scoring module — cài đặt đúng công thức chấm điểm của BTC AIC 2026."""

from src.scoring.rscore import (
    K_THRESHOLDS,
    Interval,
    KISAnswer,
    KISGroundTruth,
    QAAnswer,
    QAGroundTruth,
    ScoreReport,
    TrakeAnswer,
    TrakeGroundTruth,
    final_score,
    r_at_k,
    r_score,
)

__all__ = [
    "K_THRESHOLDS",
    "Interval",
    "KISAnswer",
    "KISGroundTruth",
    "QAAnswer",
    "QAGroundTruth",
    "ScoreReport",
    "TrakeAnswer",
    "TrakeGroundTruth",
    "final_score",
    "r_at_k",
    "r_score",
]
