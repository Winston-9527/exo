"""Cryptographic contract tests for placement-bound private payloads."""

import pytest

from exo.shared.models.model_cards import ModelId
from exo.shared.types.common import NodeId
from exo.shared.types.verifiable import (
    VerifiableEncryptionContext,
    VerifiableRecipient,
)
from exo.shared.types.worker.instances import InstanceId
from exo.verifiable.crypto import (
    VerifiableDecryptionError,
    decrypt_private_payload,
    delivery_public_key,
    encrypt_private_payload,
    generate_delivery_private_key,
)


def _context(*, placement_digest: str) -> VerifiableEncryptionContext:
    return VerifiableEncryptionContext(
        protocol_version="verifiable-exo-v1",
        request_id="request-00000000-0000-4000-8000-000000000001",
        model=ModelId("mlx-community/Qwen3-0.6B-8bit"),
        instance_id=InstanceId(
            "instance-00000000-0000-4000-8000-000000000001"
        ),
        placement_digest=placement_digest,
        recipient=VerifiableRecipient(
            node_id=NodeId("node-ingress"),
            provider_id="sha256:" + "b" * 64,
            key_id="delivery-key-v1",
        ),
    )


def test_encrypted_payload_only_opens_for_original_placement_binding() -> None:
    """Changing the committed placement makes AEAD authentication fail."""
    private_key = generate_delivery_private_key()
    public_key = delivery_public_key(private_key)
    original = _context(placement_digest="sha256:" + "a" * 64)
    encrypted = encrypt_private_payload(
        b'{"input":[{"role":"user","content":"secret prompt"}]}',
        public_key,
        original,
    )

    assert decrypt_private_payload(encrypted, private_key, original) == (
        b'{"input":[{"role":"user","content":"secret prompt"}]}'
    )

    tampered = _context(placement_digest="sha256:" + "c" * 64)
    with pytest.raises(VerifiableDecryptionError):
        decrypt_private_payload(encrypted, private_key, tampered)
