"""Pure synchronization policies for verifiable distributed generation."""

from collections.abc import Callable, Sequence


class VerifiableDistributedPreparationError(RuntimeError):
    """A sanitized error raised identically by every distributed rank."""


def synchronize_rank_preparation[T](
    prepare: Callable[[], T],
    gather_readiness: Callable[[int], Sequence[int]],
) -> T:
    """Run local preparation and join readiness before any rank may raise.

    Local exception details are deliberately discarded because the resulting
    error can be forwarded through shared task events.
    """
    try:
        prepared = prepare()
    except Exception:
        _ = gather_readiness(0)
        raise VerifiableDistributedPreparationError(
            "Verifiable private input preparation failed"
        ) from None

    readiness = gather_readiness(1)
    if not readiness or any(ready != 1 for ready in readiness):
        raise VerifiableDistributedPreparationError(
            "Verifiable private input preparation failed"
        ) from None
    return prepared


def canonical_rank_local_value[T](
    *,
    rank: int,
    source_rank: int,
    sample: Callable[[], T],
    placeholder: Callable[[], T],
) -> T:
    """Sample only on the canonical rank; contribute a placeholder elsewhere."""
    return sample() if rank == source_rank else placeholder()


def canonical_rank_slice(
    *, source_rank: int, world_size: int, values_per_rank: int
) -> slice:
    """Return the flattened all-gather slice contributed by the source rank."""
    if not 0 <= source_rank < world_size:
        raise ValueError("Canonical source rank is outside the distributed group")
    if values_per_rank <= 0:
        raise ValueError("Canonical rank must contribute at least one value")
    start = source_rank * values_per_rank
    return slice(start, start + values_per_rank)


def call_nonfatal[T](callback: Callable[[T], None], value: T) -> bool:
    """Invoke best-effort telemetry without allowing it to abort generation."""
    try:
        callback(value)
    except Exception:
        return False
    return True


def minimum_prefix_hit_length(
    *,
    prefix_cache_enabled: bool,
    system_prompt_token_count: Callable[[], int],
) -> int:
    """Avoid private prompt tokenization when no prefix cache can be consulted."""
    if not prefix_cache_enabled:
        return 1000
    return max(1000, system_prompt_token_count())


def should_run_debug_prompt_check(*, is_verifiable: bool) -> bool:
    """Debug prompt hooks are unsafe when only one rank holds the plaintext."""
    return not is_verifiable
