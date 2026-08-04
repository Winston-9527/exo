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
