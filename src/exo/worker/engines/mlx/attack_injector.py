"""Factory for boundary tamper-injection functions (Phase 2b / D3 option 1).

Adapts a P0-E2 ``Attack.generate`` candidate into the ``BoundaryHook.inject_fn``
contract ``(activation, rank) -> activation``. The P0-E2 reference package is
imported lazily at call time so the verifiable-exo core has no hard dependency
on the sibling workspace; if it is unavailable the injector raises a clear
error rather than silently no-op'ing.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

Activation = NDArray[np.float32]


def _load_p0e2_attack() -> tuple[object, object]:
    try:
        from accountedge_e2.attacks import (  # type: ignore[reportMissingImports]
            Attack,
            AttackContext,
        )

        return (Attack, AttackContext)  # type: ignore[reportUnknownVariableType]
    except ImportError as error:  # pragma: no cover - depends on sibling repo
        raise RuntimeError(
            "P0-E2 attack generators are not importable; cannot build an injector"
        ) from error


def make_attack_injector(
    *,
    attack_kind: str,
    strength: float,
    seed: int | None = 0,
    variant: str = "default",
) -> Callable[[Activation, int], Activation]:
    """Return an inject_fn that tampers a boundary activation via Attack.generate.

    The returned callable carries a ``metadata`` attribute recording the attack
    configuration for auditability.
    """
    metadata = {
        "attack_kind": attack_kind,
        "strength": strength,
        "seed": seed,
        "variant": variant,
    }
    attack_class, context_class = _load_p0e2_attack()

    def inject_fn(activation: Activation, rank: int) -> Activation:
        del rank  # the boundary rank is recorded by the hook, not the attack
        context = context_class(  # type: ignore[reportUnknownMemberType]
            kind=attack_kind,
            strength=strength,
            seed=seed,
            variant=variant,
        )
        result = attack_class.generate(np.asarray(activation), context)  # type: ignore[reportUnknownMemberType]
        return np.asarray(result.tensor, dtype=np.float32)  # type: ignore[reportUnknownMemberType]

    inject_fn.metadata = metadata  # type: ignore[attr-defined]
    return inject_fn
