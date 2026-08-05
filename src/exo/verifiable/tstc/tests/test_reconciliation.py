"""Reconciliation tests: the online TSTC module must match the P0-E2 reference.

This is the Phase 0.5 hard gate: the same (candidate, reference) boundary
tensors must produce the same score and the same detected decision in the
online module (exo.verifiable.tstc) and the controlled P0-E2 verifier
(AccountEdge/accountedge_e2.verifier). If this fails, the paper's two sets of
results cannot be cross-referenced.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from exo.verifiable.tstc.sketch import (
    capture_projected_cosine_sketch,
    capture_projected_scalar_sketch,
    capture_scalar_sketch,
)

# Make the P0-E2 reference verifier importable from the sibling workspace.
# tests/ is at <ndss2027>/verifiable-exo/src/exo/verifiable/tstc/tests/
_NDSS_ROOT = Path(__file__).resolve().parents[6]
_P0E2_SRC = _NDSS_ROOT / "workspace" / "AdversarialEvaluation" / "src"
sys.path.insert(0, str(_P0E2_SRC))

from accountedge_e2.verifier import (  # noqa: E402
    NORMALIZED_DECISION_RULE,
    Verifier,
    VerifierPolicy,
)

pytestmark = pytest.mark.skipif(
    not _P0E2_SRC.exists(),
    reason="P0-E2 reference verifier is not present on this host",
)


def _make_policy(mode: str, threshold: float, **kwargs: object) -> VerifierPolicy:
    return VerifierPolicy(mode=mode, threshold=threshold, **kwargs)


def test_reconcile_scalar16_matches_reference_semantics() -> None:
    reference = np.random.default_rng(0).normal(size=(4, 8)).astype(np.float32)
    candidate = reference.copy()
    candidate[2, 5] += 0.25
    coords = (2 * 8 + 5,)

    ours = capture_scalar_sketch(reference, candidate, coordinate_indices=coords)
    policy = _make_policy(
        "scalar16",
        threshold=0.2,
        coordinate_indices=coords,
        decision_rule=NORMALIZED_DECISION_RULE,
        normalizer=1.0,
        cutoff=0.2,
    )
    ref_result = Verifier.evaluate(reference, candidate, policy)

    assert ours.score == pytest.approx(ref_result.score)
    assert ours.coordinate_indices == ref_result.coordinate_indices
    assert ref_result.detected == (ours.score / 1.0 > 0.2)


def test_reconcile_projcos4_matches_reference_semantics() -> None:
    reference = np.random.default_rng(1).normal(size=(4, 8)).astype(np.float32)
    candidate = reference.copy()
    candidate[1] = -candidate[1]  # direction change at token 1
    tokens = (0, 1, 2, 3)
    projection_seed = 42

    ours = capture_projected_cosine_sketch(
        reference,
        candidate,
        projection_dimension=4,
        token_indices=tokens,
        projection_seed=projection_seed,
    )
    policy = _make_policy(
        "projcos4",
        threshold=0.4,
        token_indices=tokens,
        projection_seed=projection_seed,
        decision_rule=NORMALIZED_DECISION_RULE,
        normalizer=1.0,
        cutoff=0.4,
    )
    ref_result = Verifier.evaluate(reference, candidate, policy)

    assert ours.score == pytest.approx(ref_result.score)
    assert ours.token_indices == ref_result.token_indices
    assert ours.projection_digest == ref_result.projection_digest


def test_reconcile_projscalar1_abs_matches_reference_semantics() -> None:
    reference = np.random.default_rng(2).normal(size=(4, 8)).astype(np.float32)
    candidate = reference * 2.0  # global scale
    tokens = (0, 1, 2, 3)
    projection_seed = 7

    ours = capture_projected_scalar_sketch(
        reference,
        candidate,
        token_indices=tokens,
        projection_seed=projection_seed,
    )
    policy = _make_policy(
        "projscalar1_abs",
        threshold=0.3,
        token_indices=tokens,
        projection_seed=projection_seed,
        decision_rule=NORMALIZED_DECISION_RULE,
        normalizer=1.0,
        cutoff=0.3,
    )
    ref_result = Verifier.evaluate(reference, candidate, policy)

    assert ours.score == pytest.approx(ref_result.score)
    assert ours.token_indices == ref_result.token_indices
    assert ours.projection_digest == ref_result.projection_digest


def test_reconcile_detection_verdict_matches_across_modes() -> None:
    """The online detected decision must equal the P0-E2 detected decision."""
    reference = np.random.default_rng(3).normal(size=(4, 8)).astype(np.float32)
    tampered = reference.copy()
    tampered[0, 0] += 10.0  # clearly over any sane threshold
    coords = (0,)
    tokens = (0, 1, 2, 3)

    for mode, ours_fn in (
        (
            "scalar16",
            lambda r, c: capture_scalar_sketch(r, c, coordinate_indices=coords).score,
        ),
        (
            "projcos4",
            lambda r, c: capture_projected_cosine_sketch(
                r, c, projection_dimension=4, token_indices=tokens, projection_seed=1
            ).score,
        ),
        (
            "projscalar1_abs",
            lambda r, c: capture_projected_scalar_sketch(
                r, c, token_indices=tokens, projection_seed=1
            ).score,
        ),
    ):
        ours_score = ours_fn(reference, tampered)
        policy = _make_policy(
            mode,
            threshold=0.2,
            coordinate_indices=coords if mode == "scalar16" else None,
            token_indices=tokens if mode != "scalar16" else None,
            projection_seed=None if mode == "scalar16" else 1,
            decision_rule=NORMALIZED_DECISION_RULE,
            normalizer=1.0,
            cutoff=0.2,
        )
        ref_result = Verifier.evaluate(reference, tampered, policy)
        assert ours_score == pytest.approx(ref_result.score)
        assert (ours_score / 1.0 > 0.2) == ref_result.detected
