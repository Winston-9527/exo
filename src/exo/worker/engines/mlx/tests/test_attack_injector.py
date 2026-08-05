"""Tests for the attack injector factory (Phase 2b / D3 option 1).

The injector adapts a P0-E2 Attack.generate candidate into the BoundaryHook
inject_fn contract (activation, rank) -> activation. It is opt-in and lazily
imports the P0-E2 reference package so the verifiable-exo core has no hard
dependency on the sibling workspace.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from exo.worker.engines.mlx.attack_injector import make_attack_injector

_NDSS_ROOT = Path(__file__).resolve().parents[7]
_P0E2_SRC = _NDSS_ROOT / "workspace" / "AdversarialEvaluation" / "src"
sys.path.insert(0, str(_P0E2_SRC))

_P0E2_AVAILABLE = _P0E2_SRC.exists()

pytestmark = pytest.mark.skipif(
    not _P0E2_AVAILABLE,
    reason="P0-E2 attack generators are not present on this host",
)


def _activation() -> np.ndarray:
    return np.random.default_rng(0).normal(size=(4, 8)).astype(np.float32)


def test_make_attack_injector_returns_callable() -> None:
    injector = make_attack_injector(
        attack_kind="positive_global_scale", strength=2.0, seed=7
    )
    assert callable(injector)


def test_injector_applies_attack_to_activation() -> None:
    x = _activation()
    injector = make_attack_injector(
        attack_kind="positive_global_scale", strength=2.0, seed=7
    )
    result = injector(x, rank=1)
    assert result.shape == x.shape
    assert not np.allclose(result, x)  # tampered


def test_injector_global_scale_multiplies() -> None:
    x = np.ones((2, 3), dtype=np.float32)
    injector = make_attack_injector(
        attack_kind="positive_global_scale", strength=2.0, seed=7
    )
    result = injector(x, rank=1)
    assert np.allclose(result, 2.0)  # scale by 2.0


def test_injector_gaussian_changes_tensor() -> None:
    x = _activation()
    injector = make_attack_injector(
        attack_kind="gaussian_relative_std", strength=0.3, seed=5
    )
    result = injector(x, rank=1)
    assert not np.allclose(result, x)


def test_injector_honest_returns_original() -> None:
    x = _activation()
    injector = make_attack_injector(attack_kind="honest", strength=0.0, seed=None)
    result = injector(x, rank=1)
    assert np.allclose(result, x)


def test_injector_metadata_records_attack_kind() -> None:
    injector = make_attack_injector(
        attack_kind="gaussian_relative_std", strength=0.1, seed=1
    )
    metadata = getattr(injector, "metadata", None)
    assert metadata is not None
    assert metadata["attack_kind"] == "gaussian_relative_std"
    assert metadata["strength"] == 0.1
