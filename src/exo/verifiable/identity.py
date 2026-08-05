"""Stable provider delivery identity for verifiable encrypted requests."""

import base64
import hashlib
import os
from pathlib import Path

from exo.shared.types.common import NodeId
from exo.shared.types.verifiable import VerifiableProviderIdentity
from exo.verifiable.crypto import (
    delivery_public_key,
    generate_delivery_private_key,
)

EXO_VERIFIABLE_PRIVATE_KEY = "EXO_VERIFIABLE_PRIVATE_KEY"
EXO_VERIFIABLE_KEY_PATH = "EXO_VERIFIABLE_KEY_PATH"
DELIVERY_KEY_ID = "delivery-key-v1"


def provider_id_from_public_key(public_key: str) -> str:
    raw_public_key = base64.b64decode(public_key, validate=True)
    return f"sha256:{hashlib.sha256(raw_public_key).hexdigest()}"


def load_or_create_delivery_private_key() -> str:
    """Load a stable key, creating a mode-0600 key at the configured path."""
    inline_key = os.environ.get(EXO_VERIFIABLE_PRIVATE_KEY)
    if inline_key:
        return inline_key.strip()

    configured_path = os.environ.get(EXO_VERIFIABLE_KEY_PATH)
    if not configured_path:
        raise RuntimeError(
            f"Set {EXO_VERIFIABLE_KEY_PATH} (preferred) or "
            f"{EXO_VERIFIABLE_PRIVATE_KEY} on the ingress provider"
        )

    key_path = Path(configured_path)
    try:
        return key_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        key_path.parent.mkdir(parents=True, exist_ok=True)
        private_key = generate_delivery_private_key()
        try:
            file_descriptor = os.open(
                key_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError:
            return key_path.read_text(encoding="utf-8").strip()
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as key_file:
            _ = key_file.write(private_key)
        return private_key


def load_delivery_private_key() -> str:
    """Load the local provider key without exposing it to task metadata."""
    return load_or_create_delivery_private_key()


def local_delivery_identity(node_id: NodeId) -> VerifiableProviderIdentity:
    private_key = load_or_create_delivery_private_key()
    public_key = delivery_public_key(private_key)
    return VerifiableProviderIdentity(
        node_id=node_id,
        provider_id=provider_id_from_public_key(public_key),
        key_id=DELIVERY_KEY_ID,
        public_key=public_key,
    )
