"""Tests for env-driven boundary hook configuration.

P0E3_BOUNDARY_HOOK lets an operator enable capture and/or injection without
code changes: "capture" enables boundary capture, "inject:<kind>:<strength>"
builds an attack injector, and empty/unset means the hook is off.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from exo.worker.engines.mlx.boundary_hook import BoundaryHook
from exo.worker.engines.mlx.boundary_hook_config import boundary_hook_from_env

_NDSS_ROOT = Path(__file__).resolve().parents[7]
_P0E2_SRC = _NDSS_ROOT / "workspace" / "AdversarialEvaluation" / "src"
sys.path.insert(0, str(_P0E2_SRC))

_P0E2_AVAILABLE = _P0E2_SRC.exists()

pytestmark = pytest.mark.skipif(
    not _P0E2_AVAILABLE,
    reason="P0-E2 attack generators are not present on this host",
)


def test_env_empty_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("P0E3_BOUNDARY_HOOK", raising=False)
    assert boundary_hook_from_env() is None


def test_env_off_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("P0E3_BOUNDARY_HOOK", "off")
    assert boundary_hook_from_env() is None


def test_env_capture_creates_capture_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("P0E3_BOUNDARY_HOOK", "capture")
    hook = boundary_hook_from_env()
    assert hook is not None
    assert hook.capture_callback is not None
    assert hook.inject_fn is None


def test_env_inject_creates_inject_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("P0E3_BOUNDARY_HOOK", "inject:positive_global_scale:2.0")
    hook = boundary_hook_from_env()
    assert hook is not None
    assert hook.inject_fn is not None
    x = np.ones((2, 3), dtype=np.float32)
    result = hook.inject_fn(x, rank=1)
    assert np.allclose(result, 2.0)


def test_env_capture_and_inject(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "P0E3_BOUNDARY_HOOK", "capture,inject:gaussian_relative_std:0.3"
    )
    hook = boundary_hook_from_env()
    assert hook is not None
    assert hook.capture_callback is not None
    assert hook.inject_fn is not None


def test_env_invalid_kind_raises_on_invoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("P0E3_BOUNDARY_HOOK", "inject:unknown_kind:1.0")
    hook = boundary_hook_from_env()
    assert hook is not None and hook.inject_fn is not None
    with pytest.raises(Exception):
        hook.inject_fn(np.ones((2, 2), dtype=np.float32), rank=1)
