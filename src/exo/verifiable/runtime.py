"""Role-aware preparation of encrypted text-generation inputs."""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.verifiable import (
    VerifiableEncryptionContext,
    VerifiableInputReceipt,
)
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.verifiable.crypto import decrypt_private_payload, delivery_public_key
from exo.verifiable.identity import DELIVERY_KEY_ID, provider_id_from_public_key
from exo.verifiable.placement import placement_digest
from exo.verifiable.private_types import VerifiablePrivateTaskPayload


class VerifiableInputSource(str, Enum):
    DecryptedEnvelope = "decrypted_envelope"
    ShapeOnlyDummy = "shape_only_dummy"


@dataclass(frozen=True)
class PreparedVerifiableRankInput:
    source: VerifiableInputSource
    private_task_params: TextGenerationTaskParams | None


@dataclass(frozen=True)
class PreparedRankGenerationInput:
    source: VerifiableInputSource
    task_params: TextGenerationTaskParams
    prompt: str
    private_prompt_source_rank: int


def private_prompt_token_plan(
    *,
    local_token_ids: list[int],
    rank: int,
    source_rank: int,
    gathered_lengths: list[int],
) -> list[int]:
    """Return source tokens at ingress and equal-length dummy IDs elsewhere."""
    if not 0 <= source_rank < len(gathered_lengths):
        raise ValueError("Private prompt source rank is outside the distributed group")
    source_length = gathered_lengths[source_rank]
    if source_length <= 0:
        raise ValueError("Private prompt source rank produced no prompt tokens")

    for candidate_rank, length in enumerate(gathered_lengths):
        if candidate_rank != source_rank and length != 0:
            raise ValueError("A non-source rank produced private prompt token values")

    if rank == source_rank:
        if len(local_token_ids) != source_length:
            raise ValueError("Source prompt token length does not match collective")
        return local_token_ids
    if local_token_ids:
        raise ValueError("A non-source rank retained private prompt token values")
    return [0] * source_length


def build_verifiable_input_receipt(
    public_task_params: TextGenerationTaskParams,
    bound_instance: BoundInstance,
    source: VerifiableInputSource,
    *,
    prompt_token_count: int,
) -> VerifiableInputReceipt:
    metadata = public_task_params.verifiable
    if metadata is None:
        raise ValueError("Task is not a verifiable encrypted text-generation task")
    shard = bound_instance.bound_shard
    if not isinstance(shard, PipelineShardMetadata):
        raise ValueError("Verifiable receipts require a pure pipeline shard")
    ciphertext_digest = hashlib.sha256(
        metadata.encrypted_input.ciphertext.encode("ascii")
    ).hexdigest()
    return VerifiableInputReceipt(
        request_id=metadata.request_id,
        instance_id=str(bound_instance.instance.instance_id),
        placement_digest=metadata.placement_digest,
        node_id=bound_instance.bound_node_id,
        provider_id=metadata.recipient.provider_id,
        key_id=metadata.recipient.key_id,
        device_rank=shard.device_rank,
        world_size=shard.world_size,
        start_layer=shard.start_layer,
        end_layer=shard.end_layer,
        input_source=source.value,
        private_input_accessed=source is VerifiableInputSource.DecryptedEnvelope,
        prompt_token_count=prompt_token_count,
        ciphertext_digest=f"sha256:{ciphertext_digest}",
    )


def prepare_verifiable_rank_input(
    public_task_params: TextGenerationTaskParams,
    bound_instance: BoundInstance,
    delivery_private_key_loader: Callable[[], str],
) -> PreparedVerifiableRankInput:
    """Select the decrypting or shape-only path before accessing key material."""
    if public_task_params.verifiable is None:
        raise ValueError("Task is not a verifiable encrypted text-generation task")

    shard = bound_instance.bound_shard
    if not isinstance(shard, PipelineShardMetadata):
        raise ValueError("Verifiable tasks require a pure pipeline shard")

    metadata = public_task_params.verifiable
    if metadata.placement_digest != placement_digest(bound_instance.instance):
        raise ValueError("Verifiable task placement digest does not match local instance")

    if not shard.is_first_layer:
        return PreparedVerifiableRankInput(
            source=VerifiableInputSource.ShapeOnlyDummy,
            private_task_params=None,
        )

    if metadata.recipient.node_id != bound_instance.bound_node_id:
        raise ValueError("First pipeline shard does not match encrypted recipient")

    private_key = delivery_private_key_loader()
    local_provider_id = provider_id_from_public_key(delivery_public_key(private_key))
    if (
        metadata.recipient.provider_id != local_provider_id
        or metadata.recipient.key_id != DELIVERY_KEY_ID
    ):
        raise ValueError("Encrypted recipient does not match local provider identity")

    context = VerifiableEncryptionContext(
        protocol_version=metadata.protocol_version,
        request_id=metadata.request_id,
        model=public_task_params.model,
        instance_id=str(bound_instance.instance.instance_id),
        placement_digest=metadata.placement_digest,
        recipient=metadata.recipient,
    )
    plaintext = decrypt_private_payload(
        metadata.encrypted_input,
        private_key,
        context,
    )
    private_payload = VerifiablePrivateTaskPayload.model_validate_json(plaintext)
    private_task_params = public_task_params.model_copy(
        update={
            "input": private_payload.input,
            "instructions": private_payload.instructions,
        }
    )
    return PreparedVerifiableRankInput(
        source=VerifiableInputSource.DecryptedEnvelope,
        private_task_params=private_task_params,
    )


def prepare_rank_generation_input(
    public_task_params: TextGenerationTaskParams,
    bound_instance: BoundInstance,
    delivery_private_key_loader: Callable[[], str],
    prompt_renderer: Callable[[TextGenerationTaskParams], str],
) -> PreparedRankGenerationInput:
    """Prepare a private ingress prompt or a non-sensitive downstream placeholder."""
    prepared = prepare_verifiable_rank_input(
        public_task_params,
        bound_instance,
        delivery_private_key_loader,
    )
    if prepared.source is VerifiableInputSource.ShapeOnlyDummy:
        return PreparedRankGenerationInput(
            source=prepared.source,
            task_params=public_task_params,
            prompt="",
            private_prompt_source_rank=0,
        )

    private_task_params = prepared.private_task_params
    assert private_task_params is not None
    return PreparedRankGenerationInput(
        source=prepared.source,
        task_params=private_task_params,
        prompt=prompt_renderer(private_task_params),
        private_prompt_source_rank=0,
    )
