"""Requester-side construction of a placement-bound encrypted request."""

from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.text_generation import InputMessage, InputMessageContent
from exo.shared.types.verifiable import (
    VerifiableGenerationParams,
    VerifiableProviderIdentity,
    VerifiableRecipient,
)
from exo.shared.types.worker.instances import InstanceId, MlxRingInstance
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.verifiable.client import build_verifiable_chat_request
from exo.verifiable.crypto import (
    delivery_public_key,
    generate_delivery_private_key,
)
from exo.verifiable.identity import DELIVERY_KEY_ID, provider_id_from_public_key
from exo.verifiable.placement import placement_digest
from exo.verifiable.private_types import VerifiablePrivateTaskPayload


def _instance() -> MlxRingInstance:
    model = ModelCard(
        model_id=ModelId("mlx-community/Qwen3-0.6B-8bit"),
        storage_size=Memory.from_bytes(1),
        n_layers=28,
        hidden_size=1024,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxCuda],
    )
    rank_zero = RunnerId("runner-zero")
    rank_one = RunnerId("runner-one")
    return MlxRingInstance(
        instance_id=InstanceId("instance-client-test"),
        shard_assignments=ShardAssignments(
            model_id=model.model_id,
            runner_to_shard={
                rank_zero: PipelineShardMetadata(
                    model_card=model,
                    device_rank=0,
                    world_size=2,
                    start_layer=0,
                    end_layer=14,
                    n_layers=28,
                ),
                rank_one: PipelineShardMetadata(
                    model_card=model,
                    device_rank=1,
                    world_size=2,
                    start_layer=14,
                    end_layer=28,
                    n_layers=28,
                ),
            },
            node_to_runner={
                NodeId("node-ingress"): rank_zero,
                NodeId("node-downstream"): rank_one,
            },
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )


def test_requester_encrypts_private_payload_to_placement_ingress() -> None:
    private_key = generate_delivery_private_key()
    public_key = delivery_public_key(private_key)
    identity = VerifiableProviderIdentity(
        node_id=NodeId("node-ingress"),
        provider_id=provider_id_from_public_key(public_key),
        key_id=DELIVERY_KEY_ID,
        public_key=public_key,
    )

    request = build_verifiable_chat_request(
        instance=_instance(),
        identity=identity,
        private_payload=VerifiablePrivateTaskPayload(
            input=[
                InputMessage(
                    role="user", content=InputMessageContent("private client prompt")
                )
            ]
        ),
        generation=VerifiableGenerationParams(
            max_output_tokens=16,
            temperature=0.0,
            seed=42,
            logprobs=True,
            top_logprobs=5,
        ),
        request_id="request-client-test",
    )

    assert request.recipient.node_id == NodeId("node-ingress")
    assert request.instance_id == "instance-client-test"
    assert request.generation.logprobs is True
    assert request.generation.top_logprobs == 5
    assert "private client prompt" not in request.model_dump_json()


def test_placement_commitment_binds_recipient_provider_identity() -> None:
    instance = _instance()
    recipient = VerifiableRecipient(
        node_id=NodeId("node-ingress"),
        provider_id="sha256:" + "a" * 64,
        key_id=DELIVERY_KEY_ID,
    )
    substituted_provider = recipient.model_copy(
        update={"provider_id": "sha256:" + "b" * 64}
    )
    substituted_key = recipient.model_copy(update={"key_id": "delivery-key-v2"})

    committed = placement_digest(instance, recipient)

    assert committed != placement_digest(instance, substituted_provider)
    assert committed != placement_digest(instance, substituted_key)
