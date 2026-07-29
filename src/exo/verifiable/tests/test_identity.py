"""Stable provider identity tests for encrypted input delivery."""

from pathlib import Path

import pytest

from exo.shared.types.common import NodeId
from exo.verifiable.crypto import delivery_public_key
from exo.verifiable.identity import (
    EXO_VERIFIABLE_KEY_PATH,
    load_or_create_delivery_private_key,
    local_delivery_identity,
)


def test_delivery_identity_is_stable_across_restarts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key_path = tmp_path / "provider" / "delivery.key"
    monkeypatch.setenv(EXO_VERIFIABLE_KEY_PATH, str(key_path))

    first_key = load_or_create_delivery_private_key()
    second_key = load_or_create_delivery_private_key()
    identity = local_delivery_identity(NodeId("provider-node"))

    assert first_key == second_key
    assert identity.node_id == NodeId("provider-node")
    assert identity.public_key == delivery_public_key(first_key)
    assert identity.provider_id.startswith("sha256:")
    assert identity.key_id == "delivery-key-v1"
    assert key_path.stat().st_mode & 0o777 == 0o600
