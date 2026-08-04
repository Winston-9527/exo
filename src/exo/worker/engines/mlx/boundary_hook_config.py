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

from exo.worker.engines.mlx.attack_injector import make_attack_injector
from exo.worker.engines.mlx.boundary_hook import BoundaryHook


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


def boundary_hook_from_env() -> BoundaryHook | None:
    """Build a BoundaryHook from the P0E3_BOUNDARY_HOOK env var, or None if off."""
    raw = os.environ.get("P0E3_BOUNDARY_HOOK", "").strip()
    if not raw or raw == "off":
        return None

    capture_callback = None
    inject_fn = None
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if token == "capture":

            def _capture(activation: object, rank: int) -> None:
                # Default capture just verifies the hook path; the online TSTC
                # supplies the sink. Log for audibility.
                from exo.worker.runner.bootstrap import logger

                logger.info(
                    f"[P0E3-BOUNDARY] captured activation rank={rank} "
                    f"shape={getattr(activation, 'shape', None)}"
                )

            capture_callback = _capture
        elif token.startswith("inject:"):
            kind, strength, seed = _parse_inject_spec(token[len("inject:") :])
            inject_fn = make_attack_injector(
                attack_kind=kind, strength=strength, seed=seed
            )
        else:
            raise ValueError(f"unknown P0E3_BOUNDARY_HOOK token {token!r}")

    return BoundaryHook(capture_callback=capture_callback, inject_fn=inject_fn)
