"""Tests for the mx.array boundary hook conversion in auto_parallel.

These pin that _apply_boundary_hook converts a received mx.array to a numpy
float32 view, runs the hook, and converts the (possibly replaced) result back
to an mx.array with the original dtype.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from exo.worker.engines.mlx.auto_parallel import _apply_boundary_hook
from exo.worker.engines.mlx.boundary_hook import BoundaryHook


def test_apply_hook_preserves_activation_when_no_injection() -> None:
    x = mx.array(np.random.default_rng(0).normal(size=(4, 8)).astype(np.float32))
    hook = BoundaryHook()  # off
    result = _apply_boundary_hook(x, hook, rank=1)
    assert isinstance(result, mx.array)
    assert np.allclose(np.asarray(result), np.asarray(x))


def test_apply_hook_replaces_with_injected_tensor() -> None:
    x = mx.array(np.random.default_rng(1).normal(size=(4, 8)).astype(np.float32))
    tampered = np.full((4, 8), 5.0, dtype=np.float32)
    hook = BoundaryHook(inject_fn=lambda activation, rank: tampered)
    result = _apply_boundary_hook(x, hook, rank=1)
    assert np.allclose(np.asarray(result), tampered)


def test_apply_hook_captures_honest_activation_before_inject() -> None:
    x = mx.array(np.random.default_rng(2).normal(size=(4, 8)).astype(np.float32))
    captured: list[np.ndarray] = []
    tampered = np.full((4, 8), 2.0, dtype=np.float32)

    def on_capture(activation: np.ndarray, rank: int) -> None:
        captured.append(activation.copy())

    hook = BoundaryHook(capture_callback=on_capture, inject_fn=lambda a, r: tampered)
    result = _apply_boundary_hook(x, hook, rank=1)

    assert len(captured) == 1
    assert np.allclose(captured[0], np.asarray(x))  # captured the honest tensor
    assert np.allclose(np.asarray(result), tampered)  # consumed the injected one
