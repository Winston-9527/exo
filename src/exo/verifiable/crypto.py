"""KEM-DEM helpers for provider-bound private task payloads."""

import base64
import binascii
import hashlib
import json
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from exo.shared.types.verifiable import (
    VerifiableEncryptedInput,
    VerifiableEncryptionContext,
)

_NONCE_BYTES = 12
_KEY_BYTES = 32
_HKDF_LABEL = b"verifiable-exo-v1/private-input"


class VerifiableDecryptionError(ValueError):
    """Raised when a private task payload fails authenticated decryption."""


def _encode(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _decode(value: str) -> bytes:
    return base64.b64decode(value, validate=True)


def _authenticated_data(context: VerifiableEncryptionContext) -> bytes:
    serialized = json.dumps(
        context.model_dump(mode="json", by_alias=True),
        sort_keys=True,
        separators=(",", ":"),
    )
    return serialized.encode("utf-8")


def _derive_key(shared_secret: bytes, authenticated_data: bytes) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(),
        length=_KEY_BYTES,
        salt=None,
        info=_HKDF_LABEL + hashlib.sha256(authenticated_data).digest(),
    ).derive(shared_secret)


def generate_delivery_private_key() -> str:
    """Generate a base64-encoded raw X25519 provider delivery private key."""
    private_key = X25519PrivateKey.generate()
    return _encode(
        private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )


def delivery_public_key(private_key: str) -> str:
    """Return the base64-encoded public key for a raw delivery private key."""
    loaded = X25519PrivateKey.from_private_bytes(_decode(private_key))
    return _encode(
        loaded.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    )


def encrypt_private_payload(
    payload: bytes,
    recipient_public_key: str,
    context: VerifiableEncryptionContext,
) -> VerifiableEncryptedInput:
    """Encrypt a payload for one provider and authenticate its placement context."""
    recipient = X25519PublicKey.from_public_bytes(_decode(recipient_public_key))
    ephemeral_private_key = X25519PrivateKey.generate()
    shared_secret = ephemeral_private_key.exchange(recipient)
    authenticated_data = _authenticated_data(context)
    encryption_key = _derive_key(shared_secret, authenticated_data)
    nonce = os.urandom(_NONCE_BYTES)
    ciphertext = AESGCM(encryption_key).encrypt(
        nonce, payload, associated_data=authenticated_data
    )
    ephemeral_public_key = ephemeral_private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return VerifiableEncryptedInput(
        scheme="X25519-HKDF-SHA256-AES256GCM",
        ephemeral_public_key=_encode(ephemeral_public_key),
        nonce=_encode(nonce),
        ciphertext=_encode(ciphertext),
    )


def decrypt_private_payload(
    encrypted_input: VerifiableEncryptedInput,
    recipient_private_key: str,
    context: VerifiableEncryptionContext,
) -> bytes:
    """Decrypt a payload, rejecting a wrong key or modified placement context."""
    try:
        private_key = X25519PrivateKey.from_private_bytes(
            _decode(recipient_private_key)
        )
        ephemeral_public_key = X25519PublicKey.from_public_bytes(
            _decode(encrypted_input.ephemeral_public_key)
        )
        authenticated_data = _authenticated_data(context)
        shared_secret = private_key.exchange(ephemeral_public_key)
        encryption_key = _derive_key(shared_secret, authenticated_data)
        return AESGCM(encryption_key).decrypt(
            _decode(encrypted_input.nonce),
            _decode(encrypted_input.ciphertext),
            associated_data=authenticated_data,
        )
    except (InvalidTag, ValueError, binascii.Error) as error:
        raise VerifiableDecryptionError(
            "Private task payload authentication failed"
        ) from error
