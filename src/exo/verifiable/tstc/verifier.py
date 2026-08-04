"""Online TSTC verifier: normalized decision and first-mismatch localization.

The decision rule matches the P0-E2 reference verifier:
``float64(score / normalizer) > cutoff``. A boundary is a checkpoint sketch
between a reference and a challenged boundary tensor; the chain localizes the
first over-tolerance boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from exo.verifiable.tstc.sketch import capture_scalar_sketch


@dataclass(frozen=True)
class BoundaryVerdict:
    """Decision for one checked boundary."""

    boundary: str
    score: float
    threshold: float
    detected: bool
    normalized_score: float | None = None
    cutoff: float | None = None


@dataclass(frozen=True)
class ChainVerdict:
    """Trace-level decision with per-boundary results and first-mismatch."""

    detected: bool
    first_mismatch: str | None
    local_results: Mapping[str, BoundaryVerdict]


def _validate_inputs(
    reference_trace: Mapping[str, np.ndarray],
    candidate_trace: Mapping[str, np.ndarray],
    thresholds: Mapping[str, float],
) -> tuple[str, ...]:
    if set(reference_trace) != set(candidate_trace):
        raise ValueError("reference and candidate must contain the same boundaries")
    if set(thresholds) != set(reference_trace):
        raise ValueError("thresholds must cover every boundary in the trace")
    boundary_order = tuple(reference_trace.keys())
    if not boundary_order:
        raise ValueError("trace must contain at least one boundary")
    return boundary_order


def evaluate_chain(
    reference_trace: Mapping[str, np.ndarray],
    candidate_trace: Mapping[str, np.ndarray],
    thresholds: Mapping[str, float],
    *,
    normalizers: Mapping[str, float] | None = None,
    cutoffs: Mapping[str, float] | None = None,
) -> ChainVerdict:
    """Evaluate each boundary with the scalar sketch and locate the first alarm.

    When ``normalizers`` and ``cutoffs`` are supplied, detection uses the
    normalized decision rule ``float64(score/normalizer) > cutoff``; otherwise a
    raw threshold ``score > threshold`` is used. Boundary order follows the
    ``reference_trace`` mapping order.
    """
    boundary_order = _validate_inputs(reference_trace, candidate_trace, thresholds)

    local_results: dict[str, BoundaryVerdict] = {}
    first_mismatch: str | None = None
    for boundary in boundary_order:
        reference = reference_trace[boundary]
        candidate = candidate_trace[boundary]
        threshold = thresholds[boundary]
        normalizer = normalizers.get(boundary) if normalizers else None
        cutoff = cutoffs.get(boundary) if cutoffs else None

        sketch = capture_scalar_sketch(reference, candidate)
        score = sketch.score

        normalized_score: float | None = None
        if normalizer is not None and cutoff is not None:
            normalizer_f = float(normalizer)
            cutoff_f = float(cutoff)
            if not np.isfinite(normalizer_f) or normalizer_f <= 0.0:
                raise ValueError("normalizer must be a positive finite value")
            if not np.isfinite(cutoff_f) or cutoff_f < 0.0:
                raise ValueError("cutoff must be a non-negative finite value")
            normalized_score = float(score / normalizer_f)
            detected = normalized_score > cutoff_f
        else:
            detected = score > float(threshold)

        verdict = BoundaryVerdict(
            boundary=boundary,
            score=score,
            threshold=float(threshold),
            detected=detected,
            normalized_score=normalized_score,
            cutoff=cutoff,
        )
        local_results[boundary] = verdict
        if detected and first_mismatch is None:
            first_mismatch = boundary

    return ChainVerdict(
        detected=first_mismatch is not None,
        first_mismatch=first_mismatch,
        local_results=local_results,
    )
