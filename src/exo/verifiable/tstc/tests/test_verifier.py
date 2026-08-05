"""Tests for the online TSTC verifier decision and first-mismatch localization.

The decision rule is the same normalized float64 rule as P0-E2:
``float64(score / normalizer) > cutoff``.
"""

from __future__ import annotations

import numpy as np
import pytest

from exo.verifiable.tstc.sketch import capture_scalar_sketch
from exo.verifiable.tstc.verifier import evaluate_chain


def test_chain_accepts_when_all_boundaries_within_tolerance() -> None:
    reference = {
        "B0": np.random.default_rng(0).normal(size=(2, 4)).astype(np.float32),
        "B1": np.random.default_rng(1).normal(size=(2, 4)).astype(np.float32),
    }
    candidate = {name: tensor.copy() for name, tensor in reference.items()}
    verdict = evaluate_chain(reference, candidate, thresholds={"B0": 0.5, "B1": 0.5})
    assert verdict.detected is False
    assert verdict.first_mismatch is None


def test_chain_localizes_first_mismatch() -> None:
    reference = {
        "B0": np.random.default_rng(0).normal(size=(2, 4)).astype(np.float32),
        "B1": np.random.default_rng(1).normal(size=(2, 4)).astype(np.float32),
        "B2": np.random.default_rng(2).normal(size=(2, 4)).astype(np.float32),
    }
    candidate = {name: tensor.copy() for name, tensor in reference.items()}
    # Tamper only boundary B1.
    candidate["B1"][0, 0] += 10.0
    verdict = evaluate_chain(reference, candidate, thresholds={"B0": 0.5, "B1": 0.5, "B2": 0.5})
    assert verdict.detected is True
    assert verdict.first_mismatch == "B1"


def test_chain_uses_first_alarm_not_later_tamper() -> None:
    reference = {
        "B0": np.random.default_rng(0).normal(size=(2, 4)).astype(np.float32),
        "B1": np.random.default_rng(1).normal(size=(2, 4)).astype(np.float32),
    }
    candidate = {name: tensor.copy() for name, tensor in reference.items()}
    candidate["B0"][0, 0] += 10.0  # earlier alarm
    candidate["B1"][0, 0] += 10.0
    verdict = evaluate_chain(reference, candidate, thresholds={"B0": 0.5, "B1": 0.5})
    assert verdict.first_mismatch == "B0"


def test_chain_boundary_order_follows_input_order() -> None:
    reference = {
        "B2": np.random.default_rng(0).normal(size=(2, 4)).astype(np.float32),
        "B1": np.random.default_rng(1).normal(size=(2, 4)).astype(np.float32),
    }
    candidate = {name: tensor.copy() for name, tensor in reference.items()}
    candidate["B1"][0, 0] += 10.0
    candidate["B2"][0, 0] += 10.0
    # Both tampered; first in input order is "B2".
    verdict = evaluate_chain(reference, candidate, thresholds={"B2": 0.5, "B1": 0.5})
    assert verdict.first_mismatch == "B2"


def test_chain_requires_all_boundaries_in_reference() -> None:
    reference = {"B0": np.zeros((2, 4), dtype=np.float32)}
    candidate = {"B0": np.zeros((2, 4), dtype=np.float32), "B1": np.zeros((2, 4), dtype=np.float32)}
    with pytest.raises(ValueError):
        evaluate_chain(reference, candidate, thresholds={"B0": 0.5, "B1": 0.5})


def test_chain_returns_per_boundary_verdicts() -> None:
    reference = {
        "B0": np.random.default_rng(0).normal(size=(2, 4)).astype(np.float32),
        "B1": np.random.default_rng(1).normal(size=(2, 4)).astype(np.float32),
    }
    candidate = {name: tensor.copy() for name, tensor in reference.items()}
    candidate["B1"][0, 0] += 10.0
    verdict = evaluate_chain(reference, candidate, thresholds={"B0": 0.5, "B1": 0.5})
    assert set(verdict.local_results) == {"B0", "B1"}
    assert verdict.local_results["B0"].detected is False
    assert verdict.local_results["B1"].detected is True
    assert verdict.local_results["B1"].score > verdict.local_results["B0"].score


def test_normalized_decision_rule() -> None:
    """Score/normalizer > cutoff must gate detection (P0-E2 decision rule)."""
    reference = np.random.default_rng(0).normal(size=(2, 4)).astype(np.float32)
    candidate = reference.copy()
    candidate[0, 0] += 1.0  # max abs gap = 1.0
    coords = (0,)
    score = capture_scalar_sketch(reference, candidate, coordinate_indices=coords).score
    assert score == pytest.approx(1.0)
    # normalizer=2.0, cutoff=0.4 -> 1.0/2.0 = 0.5 > 0.4 -> detected
    verdict = evaluate_chain(
        {"B0": reference},
        {"B0": candidate},
        thresholds={"B0": 0.5},
        normalizers={"B0": 2.0},
        cutoffs={"B0": 0.4},
    )
    assert verdict.local_results["B0"].detected is True
    # normalizer=2.0, cutoff=0.9 -> 0.5 <= 0.9 -> accepted
    verdict2 = evaluate_chain(
        {"B0": reference},
        {"B0": candidate},
        thresholds={"B0": 0.5},
        normalizers={"B0": 2.0},
        cutoffs={"B0": 0.9},
    )
    assert verdict2.local_results["B0"].detected is False
