"""Trace emission for MLX text-generation paths.

EXO's tracing primitives (exo.shared.tracing.trace, the trace buffer, and the
TracesCollected event) are only exercised by the image pipeline. This module
adds the emission seam so that MLX text-generation tasks produce a per-task
trace the master can persist and expose via ``/v1/traces``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from exo.shared.constants import EXO_TRACING_ENABLED
from exo.shared.tracing import clear_trace_buffer, get_trace_buffer
from exo.shared.types.events import Event, TraceEventData, TracesCollected
from exo.shared.types.tasks import TaskId
from exo.utils.channels import MpSender

T = TypeVar("T")


def emit_traces_collected(
    event_sender: MpSender[Event],
    task_id: TaskId,
    rank: int,
) -> None:
    """Send buffered trace spans as a TracesCollected event.

    No-op unless EXO_TRACING_ENABLED is set. Clears the buffer after reading
    so consecutive tasks do not accumulate spans.
    """
    if not EXO_TRACING_ENABLED:
        return

    traces = get_trace_buffer()
    if traces:
        trace_data = [
            TraceEventData(
                name=t.name,
                start_us=t.start_us,
                duration_us=t.duration_us,
                rank=t.rank,
                category=t.category,
            )
            for t in traces
        ]
        event_sender.send(
            TracesCollected(
                task_id=task_id,
                rank=rank,
                traces=trace_data,
            )
        )
    clear_trace_buffer()


def timed_call(
    fn: Callable[[], T],
    *,
    name: str,
    rank: int,
    category: str = "compute",
) -> T:
    """Invoke ``fn`` and record its wall-clock duration as a trace span.

    Uses the shared trace() context manager, which records into the shared
    buffer only when EXO_TRACING_ENABLED is set.
    """
    from exo.shared.tracing import trace

    with trace(name=name, rank=rank, category=category):
        return fn()
