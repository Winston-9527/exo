"""Pure policy tests for verifiable MLX rank synchronization."""

import pytest

from exo.worker.engines.mlx.generator.verifiable_sync import (
    VerifiableDistributedPreparationError,
    call_nonfatal,
    canonical_rank_local_value,
    canonical_rank_slice,
    minimum_prefix_hit_length,
    should_run_debug_prompt_check,
    synchronize_rank_preparation,
)


def test_local_preparation_failure_still_joins_collective_and_is_sanitized() -> None:
    gathered_values: list[int] = []

    def fail_with_private_detail() -> str:
        raise ValueError("secret prompt: do not disclose")

    def gather(local_ready: int) -> list[int]:
        gathered_values.append(local_ready)
        return [local_ready, 1]

    with pytest.raises(VerifiableDistributedPreparationError) as raised:
        synchronize_rank_preparation(fail_with_private_detail, gather)

    assert gathered_values == [0]
    assert str(raised.value) == "Verifiable private input preparation failed"
    assert "secret prompt" not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__ is True


def test_remote_preparation_failure_makes_successful_rank_fail_generically() -> None:
    with pytest.raises(
        VerifiableDistributedPreparationError,
        match="^Verifiable private input preparation failed$",
    ):
        synchronize_rank_preparation(lambda: "prepared", lambda _ready: [0, 1])


def test_preparation_returns_value_only_when_every_rank_is_ready() -> None:
    prepared = synchronize_rank_preparation(
        lambda: "prepared",
        lambda local_ready: [local_ready, 1],
    )

    assert prepared == "prepared"


def test_only_canonical_rank_invokes_sampler() -> None:
    sampled = 0
    placeholders = 0

    def sample() -> str:
        nonlocal sampled
        sampled += 1
        return "sampled"

    def placeholder() -> str:
        nonlocal placeholders
        placeholders += 1
        return "placeholder"

    local_value = canonical_rank_local_value(
        rank=0,
        source_rank=0,
        sample=sample,
        placeholder=placeholder,
    )

    assert local_value == "sampled"
    assert sampled == 1
    assert placeholders == 0


def test_noncanonical_rank_contributes_placeholder_without_sampling() -> None:
    sampled = 0

    def sample() -> str:
        nonlocal sampled
        sampled += 1
        return "must-not-be-used"

    local_value = canonical_rank_local_value(
        rank=1,
        source_rank=0,
        sample=sample,
        placeholder=lambda: "placeholder",
    )

    assert local_value == "placeholder"
    assert sampled == 0


def test_canonical_rank_slice_selects_only_source_contribution() -> None:
    selected = canonical_rank_slice(
        source_rank=1,
        world_size=3,
        values_per_rank=2,
    )

    assert selected == slice(2, 4)


def test_canonical_rank_slice_rejects_invalid_source_rank() -> None:
    with pytest.raises(ValueError, match="outside the distributed group"):
        canonical_rank_slice(source_rank=2, world_size=2, values_per_rank=1)


def test_receipt_callback_failure_is_nonfatal() -> None:
    def fail(_value: int) -> None:
        raise RuntimeError("telemetry channel closed")

    assert call_nonfatal(fail, 17) is False
    assert call_nonfatal(lambda _value: None, 17) is True


def test_no_prefix_cache_skips_system_prompt_tokenization() -> None:
    token_count_requested = False

    def token_count() -> int:
        nonlocal token_count_requested
        token_count_requested = True
        raise AssertionError("private system prompt was tokenized")

    threshold = minimum_prefix_hit_length(
        prefix_cache_enabled=False,
        system_prompt_token_count=token_count,
    )

    assert threshold == 1000
    assert token_count_requested is False


def test_prefix_cache_threshold_retains_existing_minimum() -> None:
    assert (
        minimum_prefix_hit_length(
            prefix_cache_enabled=True,
            system_prompt_token_count=lambda: 1500,
        )
        == 1500
    )
    assert (
        minimum_prefix_hit_length(
            prefix_cache_enabled=True,
            system_prompt_token_count=lambda: 10,
        )
        == 1000
    )


def test_verifiable_request_skips_debug_prompt_hook() -> None:
    assert should_run_debug_prompt_check(is_verifiable=True) is False
    assert should_run_debug_prompt_check(is_verifiable=False) is True
