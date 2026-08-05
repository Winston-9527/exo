import pytest

from exo.routing.router import Router


def test_router_rejects_invalid_bootstrap_endpoint() -> None:
    with pytest.raises(RuntimeError, match="not-a-zenoh-endpoint"):
        Router.create(
            "1",
            namespace="python-router-test",
            listen_port=52414,
            discovery_service_port=52413,
            bootstrap_peers=["not-a-zenoh-endpoint"],
        )
