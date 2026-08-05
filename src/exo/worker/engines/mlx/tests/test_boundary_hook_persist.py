"""Tests for persistent boundary-activation capture to disk.

When P0E3_CAPTURE_DIR is set, the boundary hook saves the honest activation
(H) and the injected activation (H̃) as .npz files so an experiment can later
feed both to the P0-E2 verifier and confirm first-mismatch localization.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from exo.worker.engines.mlx.boundary_hook_config import (
    boundary_hook_from_env,
    make_persistent_capture,
)

_NDSS_ROOT = Path(__file__).resolve().parents[7]
_P0E2_SRC = _NDSS_ROOT / "workspace" / "AdversarialEvaluation" / "src"
sys.path.insert(0, str(_P0E2_SRC))

_P0E2_AVAILABLE = _P0E2_SRC.exists()

pytestmark = pytest.mark.skipif(
    not _P0E2_AVAILABLE,
    reason="P0-E2 attack generators are not present on this host",
)


def test_make_persistent_capture_saves_honest_and_injected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "exo.worker.engines.mlx.boundary_hook_config.uuid4",
        lambda: "00000000-0000-4000-8000-000000000000",
    )
    capture_cb, after_cb = make_persistent_capture(
        rank=1, boundary="B1", out_dir=tmp_path
    )
    h = np.random.default_rng(0).normal(size=(4, 8)).astype(np.float32)
    h_tilde = np.full((4, 8), 3.0, dtype=np.float32)

    capture_cb(h, 1)
    after_cb(h_tilde, 1)

    files = sorted(tmp_path.glob("*.npz"))
    assert len(files) == 2
    honest = np.load(files[0])["activation"]
    injected = np.load(files[1])["activation"]
    assert np.allclose(honest, h)
    assert np.allclose(injected, h_tilde)


def test_persistent_capture_via_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("P0E3_BOUNDARY_HOOK", "capture,inject:positive_global_scale:2.0")
    monkeypatch.setenv("P0E3_CAPTURE_DIR", str(tmp_path))
    hook = boundary_hook_from_env()
    assert hook is not None
    assert hook.capture_callback is not None
    assert hook.inject_fn is not None
    assert hook.after_inject_callback is not None

    h = np.ones((2, 3), dtype=np.float32)
    result = hook.inject_fn(h, rank=1)
    assert np.allclose(result, 2.0)

    # Call the full hook path via maybe_hook_activation.
    from exo.worker.engines.mlx.boundary_hook import maybe_hook_activation

    result2 = maybe_hook_activation(h, hook=hook, rank=1)
    assert np.allclose(result2, 2.0)

    files = sorted(tmp_path.glob("*.npz"))
    assert len(files) == 2  # honest + injected


def test_no_persist_when_no_capture_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("P0E3_BOUNDARY_HOOK", "capture")
    monkeypatch.delenv("P0E3_CAPTURE_DIR", raising=False)
    hook = boundary_hook_from_env()
    assert hook is not None
    assert hook.after_inject_callback is None
