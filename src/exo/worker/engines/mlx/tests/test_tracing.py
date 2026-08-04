"""Tests for the MLX text-generation trace emission.

These pin the behavior of emit_traces_collected: it must send a
TracesCollected event with the buffered spans when tracing is enabled, clear
the buffer, and be a no-op otherwise. The trace buffer and the enabled flag
are monkeypatched so the test does not require a live EXO process.
"""

from __future__ import annotations

import pytest

from exo.shared.types.events import Event, TracesCollected
from exo.shared.types.tasks import TaskId
from exo.worker.engines.mlx import tracing as mlx_tracing
from exo.worker.engines.mlx.tracing import emit_traces_collected


class _EventCollector:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def send(self, event: Event) -> None:
        self.events.append(event)


def _fake_trace_event(name: str, rank: int, category: str) -> object:
    return type(
        "FakeTraceEvent",
        (),
        {
            "name": name,
            "start_us": 0,
            "duration_us": 100,
            "rank": rank,
            "category": category,
        },
    )()


def test_emit_sends_traces_collected_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = _EventCollector()
    monkeypatch.setattr(mlx_tracing, "EXO_TRACING_ENABLED", True)
    monkeypatch.setattr(
        mlx_tracing,
        "get_trace_buffer",
        lambda: [
            _fake_trace_event("prefill", 0, "compute"),
            _fake_trace_event("decode", 0, "compute"),
        ],
    )
    cleared = []

    def _clear() -> None:
        cleared.append(True)

    monkeypatch.setattr(mlx_tracing, "clear_trace_buffer", _clear)

    emit_traces_collected(collector, TaskId("task-1"), rank=0)

    assert len(collector.events) == 1
    event = collector.events[0]
    assert isinstance(event, TracesCollected)
    assert event.task_id == TaskId("task-1")
    assert event.rank == 0
    assert [t.name for t in event.traces] == ["prefill", "decode"]
    assert cleared == [True]


def test_emit_noop_when_tracing_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = _EventCollector()
    monkeypatch.setattr(mlx_tracing, "EXO_TRACING_ENABLED", False)
    monkeypatch.setattr(
        mlx_tracing,
        "get_trace_buffer",
        lambda: [_fake_trace_event("prefill", 0, "compute")],
    )

    emit_traces_collected(collector, TaskId("task-1"), rank=0)

    assert collector.events == []


def test_emit_noop_when_buffer_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = _EventCollector()
    monkeypatch.setattr(mlx_tracing, "EXO_TRACING_ENABLED", True)
    monkeypatch.setattr(mlx_tracing, "get_trace_buffer", list)

    emit_traces_collected(collector, TaskId("task-1"), rank=0)

    assert collector.events == []


def test_emit_clears_buffer_even_when_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = _EventCollector()
    monkeypatch.setattr(mlx_tracing, "EXO_TRACING_ENABLED", True)
    monkeypatch.setattr(mlx_tracing, "get_trace_buffer", list)
    cleared = []

    def _clear() -> None:
        cleared.append(True)

    monkeypatch.setattr(mlx_tracing, "clear_trace_buffer", _clear)

    emit_traces_collected(collector, TaskId("task-1"), rank=0)

    assert cleared == [True]


def test_timed_call_records_span_via_shared_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """timed_call must run the function and record a span in the shared buffer."""
    from exo.shared import tracing as shared_tracing

    monkeypatch.setattr(mlx_tracing, "EXO_TRACING_ENABLED", True)
    # The shared trace() checks the constant at import time; patch it too.
    monkeypatch.setattr(shared_tracing, "EXO_TRACING_ENABLED", True)

    shared_tracing.clear_trace_buffer()
    result = mlx_tracing.timed_call(lambda: 42, name="prefill", rank=0, category="compute")
    assert result == 42

    spans = shared_tracing.get_trace_buffer()
    assert len(spans) == 1
    assert spans[0].name == "prefill"
    assert spans[0].rank == 0
    assert spans[0].category == "compute"
    shared_tracing.clear_trace_buffer()
