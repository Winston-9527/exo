"""Integration tests: SequentialGenerator emits TracesCollected on completion.

These drive the real SequentialGenerator.step() with a mocked mlx_generate
(so no real model is needed) and assert that a TracesCollected event is sent
on the event_sender when a task finishes and EXO_TRACING_ENABLED is set.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import cast

import pytest

from exo.shared.types.events import Event, TracesCollected
from exo.shared.types.tasks import TaskId, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceId
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.utils.channels import MpReceiver, MpSender, mp_channel
from exo.worker.engines.mlx import builder as mlx_builder
from exo.worker.engines.mlx import tracing as mlx_tracing
from exo.worker.runner.llm_inference import batch_generator as mlx_batch_generator
from exo.worker.runner.llm_inference.batch_generator import SequentialGenerator


class _EventCollector:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def send(self, event: Event) -> None:
        self.events.append(event)


def _task() -> TextGeneration:
    return TextGeneration(
        task_id=TaskId("task-trace-1"),
        instance_id=InstanceId("instance-trace-1"),
        command_id="cmd-trace-1",
        task_params=TextGenerationTaskParams(
            model="mlx-community/Qwen3-0.6B-8bit",
            input=[InputMessage(role="user", content=InputMessageContent("hi"))],
            max_output_tokens=4,
            temperature=0.0,
            seed=42,
        ),
    )


def _fake_mlx_generate(*_args: object, **_kwargs: object) -> Iterator[GenerationResponse]:
    yield GenerationResponse(
        text="hi",
        token=0,
        finish_reason="stop",
        usage=None,
    )


def test_sequential_generator_emits_traces_collected_on_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finished task must emit a TracesCollected event when tracing is enabled."""
    collector = _EventCollector()
    monkeypatch.setattr(mlx_batch_generator, "mlx_generate", _fake_mlx_generate)
    monkeypatch.setattr(mlx_batch_generator, "apply_all_parsers", lambda *a, **k: iter([]))
    monkeypatch.setattr(mlx_batch_generator, "mx_all_gather_tasks", lambda q, g: (list(q), []))
    monkeypatch.setattr(mlx_batch_generator, "mx_any", lambda x, g: False)
    monkeypatch.setattr(mlx_batch_generator, "apply_chat_template", lambda t, p: "hi")
    monkeypatch.setattr(mlx_batch_generator, "should_run_debug_prompt_check", lambda is_verifiable=False: False)
    monkeypatch.setattr(mlx_tracing, "EXO_TRACING_ENABLED", True)
    # Populate a trace span in the shared buffer so emit has something to send.
    import exo.shared.tracing as shared_tracing

    shared_tracing.clear_trace_buffer()
    monkeypatch.setattr(shared_tracing, "EXO_TRACING_ENABLED", True)
    with shared_tracing.trace(name="prefill", rank=0, category="compute"):
        pass

    # Build a SequentialGenerator with a stub model/tokenizer/group.
    gen = SequentialGenerator(
        model=cast(object, type("Model", (), {})()),
        tokenizer=cast(object, type("Tok", (), {"has_tool_calling": False})()),
        group=None,
        kv_prefix_cache=None,
        tool_parser=None,
        model_id="mlx-community/Qwen3-0.6B-8bit",
        device_rank=0,
        cancel_receiver=cast(MpReceiver[TaskId], None),
        event_sender=cast(MpSender[Event], collector),
        bound_instance=cast(object, None),
    )
    gen.submit(_task())

    # step() twice: first processes the single response, second hits StopIteration.
    list(gen.step())
    list(gen.step())

    assert len(collector.events) >= 1
    traces = [e for e in collector.events if isinstance(e, TracesCollected)]
    assert traces, "no TracesCollected emitted"
    assert traces[0].task_id == TaskId("task-trace-1")
    assert traces[0].rank == 0
    assert traces[0].traces, "trace event list is empty"
    assert any(t.name == "prefill" for t in traces[0].traces)
    shared_tracing.clear_trace_buffer()


def test_sequential_generator_no_trace_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When tracing is disabled no TracesCollected is emitted."""
    collector = _EventCollector()
    monkeypatch.setattr(mlx_batch_generator, "mlx_generate", _fake_mlx_generate)
    monkeypatch.setattr(mlx_batch_generator, "apply_all_parsers", lambda *a, **k: iter([]))
    monkeypatch.setattr(mlx_batch_generator, "mx_all_gather_tasks", lambda q, g: (list(q), []))
    monkeypatch.setattr(mlx_batch_generator, "mx_any", lambda x, g: False)
    monkeypatch.setattr(mlx_batch_generator, "apply_chat_template", lambda t, p: "hi")
    monkeypatch.setattr(mlx_batch_generator, "should_run_debug_prompt_check", lambda is_verifiable=False: False)
    monkeypatch.setattr(mlx_tracing, "EXO_TRACING_ENABLED", False)

    gen = SequentialGenerator(
        model=cast(object, type("Model", (), {})()),
        tokenizer=cast(object, type("Tok", (), {"has_tool_calling": False})()),
        group=None,
        kv_prefix_cache=None,
        tool_parser=None,
        model_id="mlx-community/Qwen3-0.6B-8bit",
        device_rank=0,
        cancel_receiver=cast(MpReceiver[TaskId], None),
        event_sender=cast(MpSender[Event], collector),
        bound_instance=cast(object, None),
    )
    gen.submit(_task())
    list(gen.step())
    list(gen.step())

    traces = [e for e in collector.events if isinstance(e, TracesCollected)]
    assert traces == []
