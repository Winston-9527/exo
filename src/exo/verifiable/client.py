"""Requester-side construction of placement-bound encrypted chat requests."""

from exo.shared.types.common import NodeId
from exo.shared.types.verifiable import (
    VerifiableChatCompletionRequest,
    VerifiableEncryptionContext,
    VerifiableGenerationParams,
    VerifiableProviderIdentity,
    VerifiableRecipient,
)
from exo.shared.types.worker.instances import Instance
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.verifiable.crypto import encrypt_private_payload
from exo.verifiable.identity import provider_id_from_public_key
from exo.verifiable.placement import placement_digest
from exo.verifiable.private_types import VerifiablePrivateTaskPayload


def build_verifiable_chat_request(
    *,
    instance: Instance,
    identity: VerifiableProviderIdentity,
    private_payload: VerifiablePrivateTaskPayload,
    generation: VerifiableGenerationParams,
    request_id: str,
) -> VerifiableChatCompletionRequest:
    """Encrypt a private input to the provider holding the first model shard."""
    first_shard_nodes: list[NodeId] = []
    for node_id, runner_id in instance.shard_assignments.node_to_runner.items():
        shard = instance.shard_assignments.runner_to_shard[runner_id]
        if not isinstance(shard, PipelineShardMetadata):
            raise ValueError("Verifiable requests require a pure pipeline placement")
        if shard.is_first_layer:
            first_shard_nodes.append(node_id)
    if len(first_shard_nodes) != 1:
        raise ValueError("Placement must contain exactly one first pipeline shard")
    if identity.node_id != first_shard_nodes[0]:
        raise ValueError("Provider identity does not belong to placement ingress")
    if identity.provider_id != provider_id_from_public_key(identity.public_key):
        raise ValueError("Provider identity fingerprint does not match its public key")

    recipient = VerifiableRecipient(
        node_id=identity.node_id,
        provider_id=identity.provider_id,
        key_id=identity.key_id,
    )
    digest = placement_digest(instance)
    context = VerifiableEncryptionContext(
        protocol_version="verifiable-exo-v1",
        request_id=request_id,
        model=instance.shard_assignments.model_id,
        instance_id=str(instance.instance_id),
        placement_digest=digest,
        recipient=recipient,
    )
    encrypted_input = encrypt_private_payload(
        private_payload.model_dump_json().encode("utf-8"),
        identity.public_key,
        context,
    )
    return VerifiableChatCompletionRequest(
        protocol_version=context.protocol_version,
        request_id=request_id,
        model=context.model,
        instance_id=context.instance_id,
        placement_digest=digest,
        recipient=recipient,
        generation=generation,
        encrypted_input=encrypted_input,
    )
