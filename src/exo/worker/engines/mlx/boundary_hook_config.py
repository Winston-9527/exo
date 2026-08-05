"""Environment-driven boundary hook configuration.

The ``P0E3_BOUNDARY_HOOK`` env var lets an operator enable capture and/or
injection without code changes:

- unset / ``off``      -> no hook (normal execution)
- ``capture``           -> capture the honest boundary activation
- ``inject:<kind>:<strength>`` -> tamper via a P0-E2 attack injector
- ``capture,inject:<kind>:<strength>`` -> both

The injector lazily imports the P0-E2 reference package; if it is not
available a clear error is raised (never a silent no-op).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from exo.worker.engines.mlx.attack_injector import make_attack_injector
from exo.worker.engines.mlx.boundary_hook import BoundaryHook

Activation = NDArray[np.float32]


def _parse_inject_spec(spec: str) -> tuple[str, float, int | None]:
    parts = spec.split(":")
    if len(parts) < 2 or len(parts) > 3:
        raise ValueError(
            f"invalid inject spec {spec!r}; expected inject:<kind>:<strength>[:seed]"
        )
    kind = parts[0]
    try:
        strength = float(parts[1])
    except ValueError as error:
        raise ValueError(f"invalid inject strength {parts[1]!r}") from error
    seed = int(parts[2]) if len(parts) == 3 else 0
    return kind, strength, seed


def make_persistent_capture(
    *,
    rank: int,
    boundary: str,
    out_dir: Path,
) -> tuple[Callable[[Activation, int], None], Callable[[Activation, int], None]]:
    """Return (capture, after_inject) callbacks that persist H and H̃ as .npz.

    The honest activation is saved with a ``honest`` suffix and the injected
    (tampered) activation with a ``injected`` suffix so an experiment can pair
    them and feed both to the P0-E2 verifier for first-mismatch localization.
    """

    def _save(activation: Activation, kind: str) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"boundary_{boundary}_rank{rank}_{uuid4()}_{kind}.npz"
        np.savez_compressed(path, activation=np.asarray(activation, dtype=np.float32))

    def _capture(activation: Activation, callback_rank: int) -> None:
        _save(activation, "honest")

    def _after_inject(activation: Activation, callback_rank: int) -> None:
        _save(activation, "injected")

    return _capture, _after_inject


def boundary_hook_from_env() -> BoundaryHook | None:
    """Build a BoundaryHook from the P0E3_BOUNDARY_HOOK env var, or None if off."""
    raw = os.environ.get("P0E3_BOUNDARY_HOOK", "").strip()
    if not raw or raw == "off":
        return None

    capture_callback = None
    after_inject_callback = None
    inject_fn = None

    capture_dir = os.environ.get("P0E3_CAPTURE_DIR", "").strip()
    persistent = bool(capture_dir)
    if persistent:
        capture_callback, after_inject_callback = make_persistent_capture(
            rank=int(os.environ.get("P0E3_BOUNDARY_RANK", "1")),
            boundary=os.environ.get("P0E3_BOUNDARY_NAME", "B1"),
            out_dir=Path(capture_dir),
        )

    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if token == "capture":
            if not persistent:
                from exo.worker.runner.bootstrap import logger

                def _log_capture(activation: object, rank: int) -> None:
                    logger.info(
                        f"[P0E3-BOUNDARY] captured activation rank={rank} "
                        f"shape={getattr(activation, 'shape', None)}"
                    )

                capture_callback = _log_capture
        elif token.startswith("inject:"):
            kind, strength, seed = _parse_inject_spec(token[len("inject:") :])
            inject_fn = make_attack_injector(
                attack_kind=kind, strength=strength, seed=seed
            )
        else:
            raise ValueError(f"unknown P0E3_BOUNDARY_HOOK token {token!r}")

    return BoundaryHook(
        capture_callback=capture_callback,
        inject_fn=inject_fn,
        after_inject_callback=after_inject_callback,
    )
