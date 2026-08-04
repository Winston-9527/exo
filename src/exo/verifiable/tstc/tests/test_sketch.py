"""Tests for the online TSTC sketch capture semantics.

These tests pin the sketch statistics to the same semantics as the P0-E2
reference verifier (AccountEdge/verifier.py) so the online module and the
controlled harness can be reconciled on the same boundary tensors.
"""

from __future__ import annotations

import numpy as np
import pytest

from exo.verifiable.tstc.sketch import (
    capture_projected_cosine_sketch,
    capture_projected_scalar_sketch,
    capture_scalar_sketch,
)


def test_scalar_sketch_captures_max_absolute_gap_at_selected_coordinates() -> None:
    reference = np.zeros((4, 8), dtype=np.float32)
    candidate = np.zeros((4, 8), dtype=np.float32)
    # Differ only at one coordinate.
    candidate[2, 5] = 0.25
    coords = (2 * 8 + 5,)  # flat index 21

    sketch = capture_scalar_sketch(reference, candidate, coordinate_indices=coords)

    assert sketch.score == pytest.approx(0.25)
    assert sketch.coordinate_indices == coords


def test_scalar_sketch_ignores_perturbation_outside_selected_coordinates() -> None:
    reference = np.zeros((4, 8), dtype=np.float32)
    candidate = np.zeros((4, 8), dtype=np.float32)
    candidate[0, 0] = 5.0  # outside the selected set
    coords = (21,)

    sketch = capture_scalar_sketch(reference, candidate, coordinate_indices=coords)

    assert sketch.score == pytest.approx(0.0)


def test_scalar_sketch_deterministic_seeded_coordinate_selection() -> None:
    reference = np.zeros((4, 8), dtype=np.float32)
    candidate = np.zeros((4, 8), dtype=np.float32)

    first = capture_scalar_sketch(reference, candidate, selection_seed=7)
    second = capture_scalar_sketch(reference, candidate, selection_seed=7)

    assert first.coordinate_indices == second.coordinate_indices
    assert len(first.coordinate_indices) == 16
    assert all(0 <= index < 32 for index in first.coordinate_indices)


def test_scalar_sketch_requires_matching_shapes() -> None:
    reference = np.zeros((4, 8), dtype=np.float32)
    candidate = np.zeros((3, 8), dtype=np.float32)

    with pytest.raises(ValueError):
        capture_scalar_sketch(reference, candidate, coordinate_indices=(0,))


def test_scalar_sketch_seeded_selection_matches_reference_semantics() -> None:
    """The seeded coordinate choice must match the P0-E2 reference rng exactly."""
    reference = np.random.default_rng(0).normal(size=(4, 8)).astype(np.float32)
    candidate = reference.copy()
    candidate[3, 7] += 0.1

    coords = (3 * 8 + 7,)
    ours = capture_scalar_sketch(reference, candidate, coordinate_indices=coords)
    assert ours.score == pytest.approx(0.1)


def test_projected_cosine_sketch_zero_for_identical_tensors() -> None:
    tensor = np.random.default_rng(0).normal(size=(4, 8)).astype(np.float32)
    sketch = capture_projected_cosine_sketch(tensor, tensor, projection_dimension=4)
    assert sketch.score == pytest.approx(0.0)


def test_projected_cosine_sketch_detects_direction_change() -> None:
    reference = np.random.default_rng(1).normal(size=(4, 8)).astype(np.float32)
    candidate = reference.copy()
    candidate[1] = -candidate[1]  # flip one token row: direction changes
    sketch = capture_projected_cosine_sketch(reference, candidate, projection_dimension=4)
    # One of four rows is anti-parallel (score 2.0), the rest identical (0.0):
    # mean over 4 selected tokens = 0.5. A direction change must not be near zero.
    assert sketch.score >= 0.5
    assert sketch.score > 0.0


def test_projected_cosine_sketch_blind_to_global_scale() -> None:
    """Row normalization makes a global scale perturbation invisible to cosine."""
    reference = np.random.default_rng(2).normal(size=(4, 8)).astype(np.float32)
    candidate = reference * 2.0
    sketch = capture_projected_cosine_sketch(reference, candidate, projection_dimension=4)
    assert sketch.score == pytest.approx(0.0, abs=1e-6)


def test_projected_scalar_sketch_sensitive_to_global_scale() -> None:
    """The 1-D projected scalar compares absolute gaps, so scale is visible."""
    reference = np.random.default_rng(3).normal(size=(4, 8)).astype(np.float32)
    candidate = reference * 2.0
    sketch = capture_projected_scalar_sketch(reference, candidate, projection_seed=11)
    assert sketch.score > 0.0


def test_projected_sketch_requires_matching_shapes() -> None:
    reference = np.zeros((4, 8), dtype=np.float32)
    candidate = np.zeros((4, 7), dtype=np.float32)
    with pytest.raises(ValueError):
        capture_projected_cosine_sketch(reference, candidate, projection_dimension=4)
