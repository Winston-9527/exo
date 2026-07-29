"""Wire-safe types shared by the verifiable request control and data planes."""

from typing import Literal

from pydantic import Field

from exo.shared.types.common import ModelId, NodeId
from exo.utils.pydantic_ext import FrozenModel

VerifiableEncryptionScheme = Literal["X25519-HKDF-SHA256-AES256GCM"]


class VerifiableRecipient(FrozenModel):
    node_id: NodeId
    provider_id: str
    key_id: str


class VerifiableProviderIdentity(FrozenModel):
    node_id: NodeId
    provider_id: str
    key_id: str
    public_key: str


class VerifiableInputReceipt(FrozenModel):
    request_id: str
    instance_id: str
    placement_digest: str
    node_id: NodeId
    provider_id: str
    key_id: str
    device_rank: int = Field(ge=0)
    world_size: int = Field(gt=0)
    start_layer: int = Field(ge=0)
    end_layer: int = Field(gt=0)
    input_source: Literal["decrypted_envelope", "shape_only_dummy"]
    private_input_accessed: bool
    prompt_token_count: int = Field(gt=0)
    ciphertext_digest: str


class VerifiableAuditResponse(FrozenModel):
    request_id: str
    expected_ranks: int = Field(gt=0)
    receipts: list[VerifiableInputReceipt]


class VerifiableGenerationParams(FrozenModel):
    max_output_tokens: int = Field(gt=0)
    temperature: float = Field(ge=0.0)
    seed: int
    stream: bool = False


class VerifiableEncryptedInput(FrozenModel):
    scheme: VerifiableEncryptionScheme
    ephemeral_public_key: str
    nonce: str
    ciphertext: str


class VerifiableEncryptionContext(FrozenModel):
    """Public fields authenticated alongside an encrypted private payload."""

    protocol_version: Literal["verifiable-exo-v1"]
    request_id: str
    model: ModelId
    instance_id: str
    placement_digest: str
    recipient: VerifiableRecipient


class VerifiableTaskMetadata(FrozenModel):
    """Encrypted input and immutable placement binding carried by a global task."""

    protocol_version: Literal["verifiable-exo-v1"]
    request_id: str
    placement_digest: str
    recipient: VerifiableRecipient
    encrypted_input: VerifiableEncryptedInput


class VerifiableChatCompletionRequest(FrozenModel):
    """Placement-bound request whose private model input is always encrypted."""

    protocol_version: Literal["verifiable-exo-v1"]
    request_id: str
    model: ModelId
    instance_id: str
    placement_digest: str
    recipient: VerifiableRecipient
    generation: VerifiableGenerationParams
    encrypted_input: VerifiableEncryptedInput
