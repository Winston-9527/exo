"""Tests for the shard-boundary activation capture and injection hook.

The hook runs at the PipelineFirstLayer recv seam: after receiving the
upstream activation, it may (a) capture the honest boundary tensor for the
online TSTC and (b) inject a tampered tensor that replaces the received one.
Both are opt-in and off by default.
"""

from __future__ import annotations

import numpy as np

from exo.worker.engines.mlx.boundary_hook import (
    BoundaryHook,
    capture_boundary_activation,
    maybe_hook_activation,
)


def _activation() -> np.ndarray:
    return np.random.default_rng(0).normal(size=(4, 8)).astype(np.float32)


def test_noop_when_hook_disabled() -> None:
    x = _activation()
    result = maybe_hook_activation(x, hook=None, rank=1)
    assert result is x
    assert np.array_equal(result, x)


def test_noop_when_hook_empty() -> None:
    x = _activation()
    result = maybe_hook_activation(x, hook=BoundaryHook(), rank=1)
    assert result is x


def test_capture_records_activation() -> None:
    x = _activation()
    captured: list[np.ndarray] = []

    def on_capture(activation: np.ndarray, rank: int) -> None:
        captured.append(activation.copy())

    hook = BoundaryHook(capture_callback=on_capture)
    result = maybe_hook_activation(x, hook=hook, rank=1)

    assert len(captured) == 1
    assert np.array_equal(captured[0], x)
    assert result is x  # capture does not alter the tensor


def test_inject_replaces_activation() -> None:
    x = _activation()
    tampered = np.full_like(x, 7.0)

    hook = BoundaryHook(inject_fn=lambda activation, rank: tampered)
    result = maybe_hook_activation(x, hook=hook, rank=1)

    assert np.array_equal(result, tampered)
    assert not np.array_equal(result, x)


def test_capture_and_inject_both_run() -> None:
    x = _activation()
    captured: list[np.ndarray] = []
    tampered = np.full_like(x, 3.0)

    def on_capture(activation: np.ndarray, rank: int) -> None:
        captured.append(activation.copy())

    hook = BoundaryHook(capture_callback=on_capture, inject_fn=lambda a, r: tampered)
    result = maybe_hook_activation(x, hook=hook, rank=1)

    assert len(captured) == 1
    assert np.array_equal(captured[0], x)  # captured the honest activation
    assert np.array_equal(result, tampered)  # returned the injected one


def test_capture_boundary_activation_saves_to_store() -> None:
    store: dict[tuple[str, int], np.ndarray] = {}

    def sink(activation: np.ndarray, rank: int, boundary: str = "B1") -> None:
        store[(boundary, rank)] = activation.copy()

    x = _activation()
    capture_boundary_activation(x, rank=1, boundary="B1", sink=sink)
    assert np.array_equal(store[("B1", 1)], x)


def test_inject_fn_receives_honest_activation() -> None:
    x = _activation()
    received: list[np.ndarray] = []

    def inject_fn(activation: np.ndarray, rank: int) -> np.ndarray:
        received.append(activation.copy())
        return activation

    hook = BoundaryHook(inject_fn=inject_fn)
    maybe_hook_activation(x, hook=hook, rank=1)
    assert np.array_equal(received[0], x)
