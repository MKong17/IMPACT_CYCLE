from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


def _clamp01(value: float) -> float:
    try:
        v = float(value)
    except Exception:
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


@dataclass(frozen=True)
class ReliabilitySignals:
    s_det: float = 0.0
    s_temp: float = 0.0
    s_struct: float = 0.0
    s_cross: float = 0.0
    s_support: float = 0.0
    s_conflict: float = 0.0


@dataclass(frozen=True)
class ReliabilityWeights:
    b1: float = 0.24
    b2: float = 0.20
    b3: float = 0.18
    b4: float = 0.14
    b5: float = 0.16
    b6: float = 0.08


@dataclass(frozen=True)
class WorkflowWeights:
    a1: float = 0.45
    a2: float = 0.30
    a3: float = 0.15
    a4: float = 0.10


def reliability_score(signals: ReliabilitySignals, weights: ReliabilityWeights = ReliabilityWeights()) -> float:
    raw = (
        weights.b1 * _clamp01(signals.s_det)
        + weights.b2 * _clamp01(signals.s_temp)
        + weights.b3 * _clamp01(signals.s_struct)
        + weights.b4 * _clamp01(signals.s_cross)
        + weights.b5 * _clamp01(signals.s_support)
        - weights.b6 * _clamp01(signals.s_conflict)
    )
    return _clamp01(raw)


def review_priority_score(
    *,
    reliability: float,
    importance_weight: float = 1.0,
    propagation_weight: float = 1.0,
) -> float:
    # P(x) = (1 - r(x)) * w_importance * w_propagation
    base = (1.0 - _clamp01(reliability)) * max(0.0, float(importance_weight)) * max(0.0, float(propagation_weight))
    return _clamp01(base)


def stage_module_score(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    num = 0.0
    den = 0.0
    for key, val in scores.items():
        w = max(0.0, float(weights.get(key, 0.0)))
        num += w * _clamp01(val)
        den += w
    if den <= 1e-8:
        return 0.0
    return _clamp01(num / den)


def workflow_score(
    *,
    s_struct: float,
    s_summary: float,
    s_mqa: float,
    s_verify: float,
    weights: WorkflowWeights = WorkflowWeights(),
) -> float:
    raw = (
        weights.a1 * _clamp01(s_struct)
        + weights.a2 * _clamp01(s_summary)
        + weights.a3 * _clamp01(s_mqa)
        + weights.a4 * _clamp01(s_verify)
    )
    return _clamp01(raw)
