"""Integration tests: P0-E2 attack tensors fed through the online TSTC chain.

This validates the D3 injection path semantics at module level: an honest
boundary tensor is attacked with a P0-E2 Attack.generate candidate, and the
online TSTC verifier must localize the tampered boundary while accepting
honest boundaries. Uses explicit seeds/indices so the reconciliation with the
reference verifier remains exact.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from exo.verifiable.tstc.verifier import evaluate_chain

_NDSS_ROOT = Path(__file__).resolve().parents[6]
_P0E2_SRC = _NDSS_ROOT / "workspace" / "AdversarialEvaluation" / "src"
sys.path.insert(0, str(_P0E2_SRC))

from accountedge_e2.attacks import Attack, AttackContext  # noqa: E402

pytestmark = pytest.mark.skipif(
    not _P0E2_SRC.exists(),
    reason="P0-E2 attack generators are not present on this host",
)


def _honest_trace() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "B0": rng.normal(size=(4, 8)).astype(np.float32),
        "B1": rng.normal(size=(4, 8)).astype(np.float32),
        "B2": rng.normal(size=(4, 8)).astype(np.float32),
    }


def test_gaussian_tamper_at_boundary_is_localized() -> None:
    reference = _honest_trace()
    candidate = {name: tensor.copy() for name, tensor in reference.items()}
    # Tamper only B1 with a strong Gaussian attack.
    attacked = Attack.generate(
        reference["B1"],
        AttackContext(kind="gaussian_relative_std", strength=0.5, seed=5),
    ).tensor
    candidate["B1"] = np.asarray(attacked, dtype=np.float32)

    verdict = evaluate_chain(reference, candidate, thresholds={"B0": 0.2, "B1": 0.2, "B2": 0.2})
    assert verdict.detected is True
    assert verdict.first_mismatch == "B1"
    assert verdict.local_results["B0"].detected is False
    assert verdict.local_results["B2"].detected is False


def test_global_scale_tamper_is_localized_by_scalar_sketch() -> None:
    reference = _honest_trace()
    candidate = {name: tensor.copy() for name, tensor in reference.items()}
    attacked = Attack.generate(
        reference["B2"],
        AttackContext(kind="positive_global_scale", strength=2.0),
    ).tensor
    candidate["B2"] = np.asarray(attacked, dtype=np.float32)

    verdict = evaluate_chain(reference, candidate, thresholds={"B0": 0.2, "B1": 0.2, "B2": 0.2})
    assert verdict.detected is True
    assert verdict.first_mismatch == "B2"


def test_honest_control_is_accepted() -> None:
    reference = _honest_trace()
    candidate = {name: tensor.copy() for name, tensor in reference.items()}
    # Honest attack (strength 0) must not change the tensor.
    attacked = Attack.generate(
        reference["B0"], AttackContext(kind="honest", strength=0.0, seed=None)
    ).tensor
    candidate["B0"] = np.asarray(attacked, dtype=np.float32)
    assert np.array_equal(candidate["B0"], reference["B0"])

    verdict = evaluate_chain(reference, candidate, thresholds={"B0": 0.2, "B1": 0.2, "B2": 0.2})
    assert verdict.detected is False
