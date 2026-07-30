"""Master-side cache policy tests for text-generation tasks."""

import pytest

import exo.master.main as master_main
from exo.shared.types.common import ModelId
from exo.shared.types.state import State
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.instances import InstanceId


@pytest.mark.parametrize(
    ("use_prefix_cache", "expected_endpoint"),
    [
        (None, "prefill.test:1234"),
        (True, "prefill.test:1234"),
        (False, None),
    ],
)
def test_explicit_no_cache_suppresses_linked_remote_prefill(
    monkeypatch: pytest.MonkeyPatch,
    use_prefix_cache: bool | None,
    expected_endpoint: str | None,
) -> None:
    def fake_prefill_endpoint(_state: State, _instance_id: InstanceId) -> str:
        return "prefill.test:1234"

    monkeypatch.setattr(
        master_main,
        "_prefill_endpoint_for",
        fake_prefill_endpoint,
    )
    task_params = TextGenerationTaskParams(
        model=ModelId("test-model"),
        input=[],
        use_prefix_cache=use_prefix_cache,
    )

    endpoint = master_main._prefill_endpoint_for_task(  # pyright: ignore[reportPrivateUsage]
        State(),
        InstanceId("decode-instance"),
        task_params,
    )

    assert endpoint == expected_endpoint
