"""Per-request shared prefix-cache policy."""

from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import TextGenerationTaskParams


def _task_params(use_prefix_cache: bool | None = None) -> TextGenerationTaskParams:
    return TextGenerationTaskParams(
        model=ModelId("test-model"),
        input=[],
        use_prefix_cache=use_prefix_cache,
    )


def test_unspecified_prefix_cache_policy_preserves_normal_exo_caching() -> None:
    assert _task_params().allows_prefix_cache() is True


def test_explicit_prefix_cache_bypass_disables_reuse_for_one_request() -> None:
    assert _task_params(False).allows_prefix_cache() is False


def test_explicit_prefix_cache_enable_allows_reuse() -> None:
    assert _task_params(True).allows_prefix_cache() is True
