import sys
from pathlib import Path
from socket import AF_INET6, SOCK_DGRAM, SOCK_STREAM, socket
from typing import cast

import pytest


@pytest.mark.asyncio
async def test_node_rejects_invalid_bootstrap_endpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("EXO_HOME", str(tmp_path))
    monkeypatch.setenv("EXO_DASHBOARD_DIR", str(tmp_path))

    from exo.main import Args, Node

    args = Args(
        namespace="python-node-test",
        zenoh_port=unused_port(SOCK_STREAM),
        discovery_port=unused_port(SOCK_DGRAM),
        bootstrap_peers=["not-a-zenoh-endpoint"],
        no_downloads=True,
        no_worker=True,
    )

    with pytest.raises(RuntimeError, match="not-a-zenoh-endpoint"):
        await Node.create(args)


def unused_port(socket_kind: int) -> int:
    with socket(AF_INET6, socket_kind) as listener:
        listener.bind(("::1", 0))
        return cast(tuple[str, int, int, int], listener.getsockname())[1]


def test_main_inner_reaches_zenoh_endpoint_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("EXO_HOME", str(tmp_path))
    monkeypatch.setenv("EXO_DASHBOARD_DIR", str(tmp_path))

    import exo.main as exo_main

    monkeypatch.setattr(exo_main, "logger_setup", no_op_logger_setup)
    args = exo_main.Args(
        namespace="python-main-test",
        zenoh_port=52414,
        discovery_port=52413,
        bootstrap_peers=["not-a-zenoh-endpoint"],
        no_downloads=True,
        no_worker=True,
    )

    with pytest.raises(RuntimeError, match="not-a-zenoh-endpoint"):
        exo_main.main_inner(args)


def no_op_logger_setup(_path: Path, _verbosity: int) -> None:
    pass


def test_bootstrap_peer_help_describes_zenoh_endpoints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("EXO_HOME", str(tmp_path))
    monkeypatch.setenv("EXO_DASHBOARD_DIR", str(tmp_path))

    import exo.main as exo_main

    monkeypatch.setattr(sys, "argv", ["exo", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        exo_main.Args.parse()

    assert exit_info.value.code == 0
    help_text = " ".join(capsys.readouterr().out.lower().split())
    assert "zenoh tcp endpoints" in help_text
    assert "tcp/100.64.0.2:52414" in help_text
    assert "bypass namespace-based discovery filtering" in help_text
    assert "multicast discovery namespace" in help_text
    assert "different namespaces will not connect" not in help_text
    assert "libp2p multiaddrs" not in help_text
