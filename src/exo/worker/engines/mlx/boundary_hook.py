"""Shard-boundary activation capture and tamper-injection hook.

Runs at the ``PipelineFirstLayer`` recv seam. After a downstream shard
receives the upstream boundary activation, this hook may:

- **capture** the honest boundary tensor so the online TSTC can compare a
  challenged boundary against a reference (Phase 2a);
- **inject** a tampered tensor that replaces the received one (Phase 2b /
  D3 option 1), e.g. an ``Attack.generate`` candidate from the P0-E2 family.

Both behaviors are opt-in and off by default so normal execution is never
altered.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

BoundaryName = str
Activation = NDArray[np.float32]


@dataclass(frozen=True)
class BoundaryHook:
    """Configures capture/injection at one shard boundary.

    ``capture_callback`` is invoked with the honest activation when capture is
    enabled. ``inject_fn``, when provided, returns the tensor that replaces the
    received activation (default None = no injection).
    """

    capture_callback: Callable[[Activation, int], None] | None = None
    inject_fn: Callable[[Activation, int], Activation] | None = None


def maybe_hook_activation(
    activation: Activation,
    *,
    hook: BoundaryHook | None,
    rank: int,
) -> Activation:
    """Apply capture and/or injection at a shard boundary.

    Returns the activation that downstream execution should consume: the
    original when no injection is configured, otherwise the injected tensor.
    """
    if hook is None:
        return activation

    if hook.capture_callback is not None:
        hook.capture_callback(activation, rank)

    if hook.inject_fn is not None:
        return hook.inject_fn(activation, rank)

    return activation


def capture_boundary_activation(
    activation: Activation,
    *,
    rank: int,
    boundary: BoundaryName,
    sink: Callable[[Activation, int, BoundaryName], None],
) -> None:
    """Convenience wrapper that records one boundary activation into a sink."""
    sink(activation, rank, boundary)
