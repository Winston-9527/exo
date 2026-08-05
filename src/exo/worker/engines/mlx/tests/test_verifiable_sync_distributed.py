# type: ignore
"""Real two-rank ring tests for verifiable generation synchronization.

These tests require the MLX runtime and are skipped by the ``mlx-none`` setup.
Run explicitly with ``pytest -m slow`` on an MLX-capable host.
"""

import json
import multiprocessing
import socket
import tempfile
from pathlib import Path
from queue import Empty

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

pytestmark = pytest.mark.slow


def _unused_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _hostfile(directory: str) -> str:
    ports: set[int] = set()
    while len(ports) < 2:
        ports.add(_unused_local_port())
    path = Path(directory) / "hosts.json"
    path.write_text(
        json.dumps([f"127.0.0.1:{port}" for port in sorted(ports)]),
        encoding="utf-8",
    )
    return str(path)


def _readiness_worker(rank: int, hostfile: str, result_queue) -> None:
    import os

    import mlx.core as child_mx

    from exo.worker.engines.mlx.generator.generate import (
        gather_distributed_integers,
    )
    from exo.worker.engines.mlx.generator.verifiable_sync import (
        synchronize_rank_preparation,
    )

    os.environ["MLX_HOSTFILE"] = hostfile
    os.environ["MLX_RANK"] = str(rank)
    group = child_mx.distributed.init(backend="ring", strict=True)

    def prepare() -> str:
        if rank == 0:
            raise ValueError("private prompt must not escape")
        return "ready"

    try:
        synchronize_rank_preparation(
            prepare,
            lambda ready: gather_distributed_integers(ready, group),
        )
    except Exception as error:
        result_queue.put((rank, type(error).__name__, str(error)))


def _sampler_worker(rank: int, hostfile: str, result_queue) -> None:
    import os

    import mlx.core as child_mx

    from exo.worker.engines.mlx.generator.generate import (
        make_canonical_rank_sampler,
    )

    os.environ["MLX_HOSTFILE"] = hostfile
    os.environ["MLX_RANK"] = str(rank)
    group = child_mx.distributed.init(backend="ring", strict=True)

    def source_sampler(_logprobs):
        if rank != 0:
            raise AssertionError("noncanonical rank sampled")
        return child_mx.array([77], dtype=child_mx.uint32)

    sampler = make_canonical_rank_sampler(source_sampler, group, source_rank=0)
    token = sampler(child_mx.zeros((1, 4)))
    result_queue.put((rank, token.tolist()))


def _run_two_rank_workers(target) -> list[tuple]:
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    with tempfile.TemporaryDirectory() as directory:
        hostfile = _hostfile(directory)
        processes = [
            context.Process(target=target, args=(rank, hostfile, result_queue))
            for rank in range(2)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=20)
        hanging = [process for process in processes if process.is_alive()]
        for process in hanging:
            process.terminate()
            process.join(timeout=5)
        assert not hanging, "a rank hung in a distributed collective"

        results = []
        try:
            while len(results) < 2:
                results.append(result_queue.get(timeout=2))
        except Empty:
            pass
        assert all(process.exitcode == 0 for process in processes)
        assert len(results) == 2
        return sorted(results)


def test_two_ranks_fail_together_when_ingress_preparation_fails() -> None:
    results = _run_two_rank_workers(_readiness_worker)

    assert results == [
        (
            0,
            "VerifiableDistributedPreparationError",
            "Verifiable private input preparation failed",
        ),
        (
            1,
            "VerifiableDistributedPreparationError",
            "Verifiable private input preparation failed",
        ),
    ]


def test_two_ranks_use_only_canonical_sampled_token() -> None:
    results = _run_two_rank_workers(_sampler_worker)

    assert results == [(0, [77]), (1, [77])]
