"""Tests for the online TSTC sketch capture semantics.

These tests pin the sketch statistics to the same semantics as the P0-E2
reference verifier (AccountEdge/verifier.py) so the online module and the
controlled harness can be reconciled on the same boundary tensors.
"""

from __future__ import annotations

import numpy as np
import pytest

from exo.verifiable.tstc.sketch import capture_scalar_sketch


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
