"""Sketch capture semantics for the online TSTC module.

The sketch statistics implemented here are the same statistics as the P0-E2
reference verifier (AccountEdge/verifier.py), so the online module and the
controlled harness reconcile on the same boundary tensors.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np


def _rng(seed: int, purpose: str) -> np.random.Generator:
    material = f"accountedge-e2:{seed}:{purpose}".encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return np.random.Generator(np.random.PCG64(derived_seed))


@dataclass(frozen=True)
class ScalarSketch:
    """Scalar-coordinate TSTC sketch: selected flat coordinates + max abs gap."""

    score: float
    coordinate_indices: tuple[int, ...]
    projection_digest: str | None = None


@dataclass(frozen=True)
class ProjectedSketch:
    """Projected-token TSTC sketch (projcos or projscalar1_abs)."""

    score: float
    token_indices: tuple[int, ...]
    projection_digest: str
    metric: str  # "projcos" | "projscalar1_abs"


def _require_finite(*values: np.ndarray) -> None:
    if any(not np.all(np.isfinite(value)) for value in values):
        raise ValueError("non-finite verifier arithmetic is forbidden")


def _selected_tokens(row_count: int, token_indices: tuple[int, ...] | None, selection_seed: int) -> np.ndarray:
    if token_indices is not None:
        return np.asarray(token_indices, dtype=np.int64)
    return np.sort(
        _rng(selection_seed, "tokens").choice(
            row_count,
            size=min(16, row_count),
            replace=False,
        )
    )


def _projection(hidden_size: int, dimension: int, projection_seed: int | None, purpose: str) -> np.ndarray:
    if projection_seed is None:
        raise ValueError("projected verifier requires a projection seed")
    projection = _rng(projection_seed, purpose).normal(
        loc=0.0,
        scale=1.0 / np.sqrt(dimension),
        size=(hidden_size, dimension),
    )
    _require_finite(projection)
    return projection


def _projection_digest(projection: np.ndarray) -> str:
    canonical = np.ascontiguousarray(projection, dtype="<f8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _validated_tensors(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference_input = np.asarray(reference)
    candidate_input = np.asarray(candidate)
    if (
        reference_input.dtype.kind not in "fc"
        or candidate_input.dtype.kind not in "fc"
        or not np.all(np.isfinite(reference_input))
        or not np.all(np.isfinite(candidate_input))
    ):
        raise ValueError("reference and candidate must contain finite numeric values")
    if reference_input.shape != candidate_input.shape:
        raise ValueError("reference and candidate must have identical shape")
    return np.asarray(reference_input, dtype=np.float64), np.asarray(candidate_input, dtype=np.float64)


def capture_projected_cosine_sketch(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    projection_dimension: int,
    token_indices: tuple[int, ...] | None = None,
    projection_seed: int = 0,
    selection_seed: int = 0,
) -> ProjectedSketch:
    """Capture a projected-token cosine sketch (P0-E2 projcos_d semantics).

    Projects each selected token row to ``projection_dimension``, row-normalizes,
    and reports ``mean(1 - cos)`` across selected tokens.
    """
    reference_input, candidate_input = _validated_tensors(reference, candidate)
    hidden_size = reference_input.shape[-1]
    reference_rows = reference_input.reshape(-1, hidden_size)
    candidate_rows = candidate_input.reshape(-1, hidden_size)
    indices = _selected_tokens(reference_rows.shape[0], token_indices, selection_seed)
    projection = _projection(hidden_size, projection_dimension, projection_seed, f"projection:projcos{projection_dimension}")
    selected_tokens = tuple(int(index) for index in indices)
    digest = _projection_digest(projection)
    with np.errstate(over="ignore", invalid="ignore"):
        reference_projected = reference_rows[indices] @ projection
        candidate_projected = candidate_rows[indices] @ projection
        reference_norm = np.linalg.norm(reference_projected, axis=1, keepdims=True)
        candidate_norm = np.linalg.norm(candidate_projected, axis=1, keepdims=True)
    _require_finite(reference_projected, candidate_projected, reference_norm, candidate_norm)
    reference_unit = np.divide(
        reference_projected, reference_norm, out=np.zeros_like(reference_projected), where=reference_norm > 0.0
    )
    candidate_unit = np.divide(
        candidate_projected, candidate_norm, out=np.zeros_like(candidate_projected), where=candidate_norm > 0.0
    )
    similarities = np.sum(reference_unit * candidate_unit, axis=1)
    both_zero = (reference_norm[:, 0] == 0.0) & (candidate_norm[:, 0] == 0.0)
    identical_projection = np.all(reference_projected == candidate_projected, axis=1)
    similarities[both_zero | identical_projection] = 1.0
    _require_finite(similarities)
    score = float(np.mean(1.0 - np.clip(similarities, -1.0, 1.0)))
    return ProjectedSketch(score=score, token_indices=selected_tokens, projection_digest=digest, metric="projcos")


def capture_projected_scalar_sketch(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    token_indices: tuple[int, ...] | None = None,
    projection_seed: int = 0,
    selection_seed: int = 0,
) -> ProjectedSketch:
    """Capture a 1-D projected scalar sketch (P0-E2 projscalar1_abs semantics).

    Projects each selected token row to dimension 1 and reports the mean
    absolute gap, so it remains norm-sensitive (scale is visible).
    """
    reference_input, candidate_input = _validated_tensors(reference, candidate)
    hidden_size = reference_input.shape[-1]
    reference_rows = reference_input.reshape(-1, hidden_size)
    candidate_rows = candidate_input.reshape(-1, hidden_size)
    indices = _selected_tokens(reference_rows.shape[0], token_indices, selection_seed)
    projection = _projection(hidden_size, 1, projection_seed, "projection:projscalar1_abs")
    selected_tokens = tuple(int(index) for index in indices)
    digest = _projection_digest(projection)
    with np.errstate(over="ignore", invalid="ignore"):
        reference_projected = reference_rows[indices] @ projection
        candidate_projected = candidate_rows[indices] @ projection
        differences = np.abs(reference_projected - candidate_projected)
    _require_finite(reference_projected, candidate_projected, differences)
    score = float(np.mean(differences))
    return ProjectedSketch(score=score, token_indices=selected_tokens, projection_digest=digest, metric="projscalar1_abs")


def capture_scalar_sketch(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    coordinate_indices: tuple[int, ...] | None = None,
    selection_seed: int = 0,
) -> ScalarSketch:
    """Capture a scalar-coordinate sketch between reference and candidate.

    ``coordinate_indices`` are flat indices into the tensors; when omitted they
    are chosen deterministically from ``selection_seed`` (up to 16 coordinates),
    matching the P0-E2 reference selection semantics.
    """
    reference_input = np.asarray(reference)
    candidate_input = np.asarray(candidate)
    if (
        reference_input.dtype.kind not in "fc"
        or candidate_input.dtype.kind not in "fc"
        or not np.all(np.isfinite(reference_input))
        or not np.all(np.isfinite(candidate_input))
    ):
        raise ValueError("reference and candidate must contain finite numeric values")
    if reference_input.shape != candidate_input.shape:
        raise ValueError("reference and candidate must have identical shape")

    reference_flat = np.asarray(reference_input, dtype=np.float64).reshape(-1)
    candidate_flat = np.asarray(candidate_input, dtype=np.float64).reshape(-1)

    if coordinate_indices is not None:
        indices = np.asarray(coordinate_indices, dtype=np.int64)
    else:
        indices = np.sort(
            _rng(selection_seed, "coordinates").choice(
                reference_flat.size,
                size=min(16, reference_flat.size),
                replace=False,
            )
        )
    selected = tuple(int(index) for index in indices)
    with np.errstate(over="ignore", invalid="ignore"):
        differences = np.abs(reference_flat[indices] - candidate_flat[indices])
    if not np.all(np.isfinite(differences)):
        raise ValueError("non-finite sketch arithmetic is forbidden")
    score = float(np.max(differences))
    return ScalarSketch(score=score, coordinate_indices=selected)
