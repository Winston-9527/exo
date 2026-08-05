"""Rank-role tests for selective disclosure during task preparation."""

from typing import Literal

import pytest

from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.verifiable import (
    VerifiableEncryptedInput,
    VerifiableEncryptionContext,
    VerifiableRecipient,
    VerifiableTaskMetadata,
)
from exo.shared.types.worker.instances import BoundInstance, InstanceId, MlxRingInstance
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata, TensorShardMetadata
from exo.verifiable.crypto import (
    delivery_public_key,
    encrypt_private_payload,
    generate_delivery_private_key,
)
from exo.verifiable.identity import DELIVERY_KEY_ID, provider_id_from_public_key
from exo.verifiable.placement import placement_digest
from exo.verifiable.private_types import VerifiablePrivateTaskPayload
from exo.verifiable.runtime import (
    VerifiableInputSource,
    build_verifiable_input_receipt,
    prepare_rank_generation_input,
    prepare_verifiable_rank_input,
    private_prompt_token_plan,
)


def _downstream_bound_instance() -> BoundInstance:
    model = ModelCard(
        model_id=ModelId("mlx-community/Qwen3-0.6B-8bit"),
        storage_size=Memory.from_bytes(1),
        n_layers=28,
        hidden_size=1024,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxCuda],
    )
    ingress_node = NodeId("node-ingress")
    downstream_node = NodeId("node-downstream")
    ingress_runner = RunnerId("runner-ingress")
    downstream_runner = RunnerId("runner-downstream")
    assignments = ShardAssignments(
        model_id=model.model_id,
        runner_to_shard={
            ingress_runner: PipelineShardMetadata(
                model_card=model,
                device_rank=0,
                world_size=2,
                start_layer=0,
                end_layer=14,
                n_layers=28,
            ),
            downstream_runner: PipelineShardMetadata(
                model_card=model,
                device_rank=1,
                world_size=2,
                start_layer=14,
                end_layer=28,
                n_layers=28,
            ),
        },
        node_to_runner={
            ingress_node: ingress_runner,
            downstream_node: downstream_runner,
        },
    )
    return BoundInstance(
        instance=MlxRingInstance(
            instance_id=InstanceId("instance-1"),
            shard_assignments=assignments,
            hosts_by_node={},
            ephemeral_port=50000,
        ),
        bound_runner_id=downstream_runner,
        bound_node_id=downstream_node,
    )


def _ingress_bound_instance() -> BoundInstance:
    downstream = _downstream_bound_instance()
    ingress_node = NodeId("node-ingress")
    ingress_runner = downstream.instance.shard_assignments.node_to_runner[ingress_node]
    return BoundInstance(
        instance=downstream.instance,
        bound_runner_id=ingress_runner,
        bound_node_id=ingress_node,
    )


def _invalid_placement(
    case: Literal[
        "non_pipeline",
        "non_contiguous_rank",
        "inconsistent_world_size",
        "first_layer_not_rank_zero",
        "multiple_first_layers",
    ],
) -> BoundInstance:
    bound_instance = _downstream_bound_instance()
    assignments = bound_instance.instance.shard_assignments
    ingress_runner = assignments.node_to_runner[NodeId("node-ingress")]
    downstream_runner = assignments.node_to_runner[NodeId("node-downstream")]
    ingress = assignments.runner_to_shard[ingress_runner]
    downstream = assignments.runner_to_shard[downstream_runner]
    assert isinstance(ingress, PipelineShardMetadata)
    assert isinstance(downstream, PipelineShardMetadata)

    if case == "non_pipeline":
        invalid_ingress = TensorShardMetadata(
            model_card=ingress.model_card,
            device_rank=ingress.device_rank,
            world_size=ingress.world_size,
            start_layer=ingress.start_layer,
            end_layer=ingress.end_layer,
            n_layers=ingress.n_layers,
        )
        invalid_downstream = downstream
    elif case == "non_contiguous_rank":
        invalid_ingress = ingress
        invalid_downstream = downstream.model_copy(update={"device_rank": 2})
    elif case == "inconsistent_world_size":
        invalid_ingress = ingress
        invalid_downstream = downstream.model_copy(update={"world_size": 3})
    elif case == "first_layer_not_rank_zero":
        invalid_ingress = ingress.model_copy(update={"device_rank": 1})
        invalid_downstream = downstream.model_copy(update={"device_rank": 0})
    else:
        invalid_ingress = ingress
        invalid_downstream = downstream.model_copy(update={"start_layer": 0})

    invalid_assignments = assignments.model_copy(
        update={
            "runner_to_shard": {
                ingress_runner: invalid_ingress,
                downstream_runner: invalid_downstream,
            }
        }
    )
    invalid_instance = bound_instance.instance.model_copy(
        update={"shard_assignments": invalid_assignments}
    )
    return bound_instance.model_copy(update={"instance": invalid_instance})


def _task_for_bound_instance(
    bound_instance: BoundInstance,
) -> TextGenerationTaskParams:
    task_params = _encrypted_task_params()
    metadata = task_params.verifiable
    assert metadata is not None
    return task_params.model_copy(
        update={
            "verifiable": metadata.model_copy(
                update={
                    "placement_digest": placement_digest(
                        bound_instance.instance, metadata.recipient
                    )
                }
            )
        }
    )


def _encrypted_task_params() -> TextGenerationTaskParams:
    bound_instance = _downstream_bound_instance()
    recipient = VerifiableRecipient(
        node_id=NodeId("node-ingress"),
        provider_id="sha256:" + "b" * 64,
        key_id=DELIVERY_KEY_ID,
    )
    return TextGenerationTaskParams(
        model=ModelId("mlx-community/Qwen3-0.6B-8bit"),
        input=[],
        max_output_tokens=32,
        temperature=0.0,
        seed=42,
        verifiable=VerifiableTaskMetadata(
            protocol_version="verifiable-exo-v1",
            request_id="request-1",
            placement_digest=placement_digest(bound_instance.instance, recipient),
            recipient=recipient,
            encrypted_input=VerifiableEncryptedInput(
                scheme="X25519-HKDF-SHA256-AES256GCM",
                ephemeral_public_key="not-loaded-by-downstream",
                nonce="not-loaded-by-downstream",
                ciphertext="not-loaded-by-downstream",
            ),
        ),
    )


@pytest.mark.parametrize(
    ("case", "expected_error"),
    [
        ("non_pipeline", "pure pipeline"),
        ("non_contiguous_rank", "contiguous device ranks"),
        ("inconsistent_world_size", "common world size"),
        ("first_layer_not_rank_zero", "first layer must use device rank zero"),
        ("multiple_first_layers", "exactly one first layer"),
    ],
)
def test_runtime_rejects_invalid_private_prompt_placement(
    case: Literal[
        "non_pipeline",
        "non_contiguous_rank",
        "inconsistent_world_size",
        "first_layer_not_rank_zero",
        "multiple_first_layers",
    ],
    expected_error: str,
) -> None:
    bound_instance = _invalid_placement(case)
    task_params = _task_for_bound_instance(bound_instance)

    with pytest.raises(ValueError, match=expected_error):
        prepare_verifiable_rank_input(
            task_params,
            bound_instance,
            lambda: (_ for _ in ()).throw(AssertionError("key load attempted")),
        )


def test_downstream_rank_never_loads_delivery_key_or_private_input() -> None:
    """A non-first pipeline rank selects the dummy path before touching key material."""
    key_load_attempted = False

    def forbidden_key_loader() -> str:
        nonlocal key_load_attempted
        key_load_attempted = True
        raise AssertionError("downstream rank attempted to load a delivery key")

    prepared = prepare_verifiable_rank_input(
        _encrypted_task_params(),
        _downstream_bound_instance(),
        forbidden_key_loader,
    )

    assert prepared.source is VerifiableInputSource.ShapeOnlyDummy
    assert prepared.private_task_params is None
    assert not key_load_attempted


def test_first_pipeline_rank_decrypts_private_input() -> None:
    """Only the first shard reconstructs the real text-generation input."""
    bound_instance = _ingress_bound_instance()
    private_key = generate_delivery_private_key()
    public_key = delivery_public_key(private_key)
    recipient = VerifiableRecipient(
        node_id=bound_instance.bound_node_id,
        provider_id=provider_id_from_public_key(public_key),
        key_id=DELIVERY_KEY_ID,
    )
    context = VerifiableEncryptionContext(
        protocol_version="verifiable-exo-v1",
        request_id="request-1",
        model=ModelId("mlx-community/Qwen3-0.6B-8bit"),
        instance_id=str(bound_instance.instance.instance_id),
        placement_digest=placement_digest(bound_instance.instance, recipient),
        recipient=recipient,
    )
    private_payload = VerifiablePrivateTaskPayload(
        input=[
            InputMessage(
                role="user", content=InputMessageContent("secret prompt for rank zero")
            )
        ]
    )
    encrypted_input = encrypt_private_payload(
        private_payload.model_dump_json().encode("utf-8"),
        public_key,
        context,
    )
    public_task_params = TextGenerationTaskParams(
        model=context.model,
        input=[],
        max_output_tokens=32,
        temperature=0.0,
        seed=42,
        verifiable=VerifiableTaskMetadata(
            protocol_version=context.protocol_version,
            request_id=context.request_id,
            placement_digest=context.placement_digest,
            recipient=recipient,
            encrypted_input=encrypted_input,
        ),
    )

    prepared = prepare_verifiable_rank_input(
        public_task_params,
        bound_instance,
        lambda: private_key,
    )

    assert prepared.source is VerifiableInputSource.DecryptedEnvelope
    assert prepared.private_task_params is not None
    assert prepared.private_task_params.input == private_payload.input


def test_downstream_rank_never_renders_private_prompt() -> None:
    """Downstream prompt construction is shape-only and cannot touch plaintext."""
    prompt_render_attempted = False

    def forbidden_prompt_renderer(_: TextGenerationTaskParams) -> str:
        nonlocal prompt_render_attempted
        prompt_render_attempted = True
        raise AssertionError("downstream rank attempted to render a private prompt")

    prepared = prepare_rank_generation_input(
        _encrypted_task_params(),
        _downstream_bound_instance(),
        lambda: (_ for _ in ()).throw(AssertionError("key load attempted")),
        forbidden_prompt_renderer,
    )

    assert prepared.source is VerifiableInputSource.ShapeOnlyDummy
    assert prepared.task_params.input == []
    assert prepared.prompt == ""
    assert prepared.private_prompt_source_rank == 0
    assert not prompt_render_attempted


def test_ingress_rank_renders_decrypted_prompt() -> None:
    bound_instance = _ingress_bound_instance()
    private_key = generate_delivery_private_key()
    public_key = delivery_public_key(private_key)
    recipient = VerifiableRecipient(
        node_id=bound_instance.bound_node_id,
        provider_id=provider_id_from_public_key(public_key),
        key_id=DELIVERY_KEY_ID,
    )
    context = VerifiableEncryptionContext(
        protocol_version="verifiable-exo-v1",
        request_id="request-render",
        model=ModelId("mlx-community/Qwen3-0.6B-8bit"),
        instance_id=str(bound_instance.instance.instance_id),
        placement_digest=placement_digest(bound_instance.instance, recipient),
        recipient=recipient,
    )
    payload = VerifiablePrivateTaskPayload(
        input=[InputMessage(role="user", content=InputMessageContent("secret"))]
    )
    encrypted = encrypt_private_payload(
        payload.model_dump_json().encode(), public_key, context
    )
    public = _encrypted_task_params().model_copy(
        update={
            "verifiable": VerifiableTaskMetadata(
                protocol_version=context.protocol_version,
                request_id=context.request_id,
                placement_digest=context.placement_digest,
                recipient=recipient,
                encrypted_input=encrypted,
            )
        }
    )

    prepared = prepare_rank_generation_input(
        public,
        bound_instance,
        lambda: private_key,
        lambda params: f"rendered:{params.input[0].content}",
    )

    assert prepared.source is VerifiableInputSource.DecryptedEnvelope
    assert prepared.prompt == "rendered:secret"
    assert prepared.private_prompt_source_rank == 0


def test_downstream_token_plan_contains_only_equal_length_dummy_ids() -> None:
    planned = private_prompt_token_plan(
        local_token_ids=[],
        rank=1,
        source_rank=0,
        gathered_lengths=[17, 0],
    )

    assert planned == [0] * 17


def test_private_prompt_token_plan_rejects_non_source_token_values() -> None:
    try:
        private_prompt_token_plan(
            local_token_ids=[1234],
            rank=1,
            source_rank=0,
            gathered_lengths=[17, 1],
        )
    except ValueError as error:
        assert "non-source rank" in str(error)
    else:
        raise AssertionError("downstream real token values were accepted")


def test_ingress_rejects_provider_id_not_derived_from_local_key() -> None:
    bound_instance = _ingress_bound_instance()
    private_key = generate_delivery_private_key()
    public_key = delivery_public_key(private_key)
    forged_recipient = VerifiableRecipient(
        node_id=bound_instance.bound_node_id,
        provider_id="sha256:" + "f" * 64,
        key_id=DELIVERY_KEY_ID,
    )
    context = VerifiableEncryptionContext(
        protocol_version="verifiable-exo-v1",
        request_id="request-forged-provider",
        model=ModelId("mlx-community/Qwen3-0.6B-8bit"),
        instance_id=str(bound_instance.instance.instance_id),
        placement_digest=placement_digest(bound_instance.instance, forged_recipient),
        recipient=forged_recipient,
    )
    encrypted = encrypt_private_payload(
        VerifiablePrivateTaskPayload(
            input=[InputMessage(role="user", content=InputMessageContent("secret"))]
        )
        .model_dump_json()
        .encode(),
        public_key,
        context,
    )
    public = _encrypted_task_params().model_copy(
        update={
            "verifiable": VerifiableTaskMetadata(
                protocol_version=context.protocol_version,
                request_id=context.request_id,
                placement_digest=context.placement_digest,
                recipient=forged_recipient,
                encrypted_input=encrypted,
            )
        }
    )

    try:
        prepare_verifiable_rank_input(public, bound_instance, lambda: private_key)
    except ValueError as error:
        assert "provider identity" in str(error)
    else:
        raise AssertionError("forged provider identity was accepted")


def test_execution_receipt_contains_shape_metadata_but_no_private_values() -> None:
    task_params = _encrypted_task_params()
    bound_instance = _downstream_bound_instance()
    reporting_private_key = generate_delivery_private_key()
    reporting_public_key = delivery_public_key(reporting_private_key)

    receipt = build_verifiable_input_receipt(
        task_params,
        bound_instance,
        VerifiableInputSource.ShapeOnlyDummy,
        execution_id=CommandId("execution-runtime-test"),
        delivery_private_key_loader=lambda: reporting_private_key,
        prompt_token_count=23,
    )

    assert receipt.execution_id == CommandId("execution-runtime-test")
    assert receipt.device_rank == 1
    assert receipt.reporting_provider_id == provider_id_from_public_key(
        reporting_public_key
    )
    assert receipt.reporting_key_id == DELIVERY_KEY_ID
    assert receipt.recipient_provider_id != receipt.reporting_provider_id
    assert receipt.recipient_key_id == DELIVERY_KEY_ID
    assert receipt.prompt_token_count == 23
    assert receipt.private_input_accessed is False
    assert receipt.ciphertext_digest.startswith("sha256:")
    serialized = receipt.model_dump_json()
    assert "not-loaded-by-downstream" not in serialized
    assert "input_ids" not in serialized


def test_ingress_receipt_reporting_identity_matches_envelope_recipient() -> None:
    bound_instance = _ingress_bound_instance()
    reporting_private_key = generate_delivery_private_key()
    reporting_public_key = delivery_public_key(reporting_private_key)
    provider_id = provider_id_from_public_key(reporting_public_key)
    recipient = VerifiableRecipient(
        node_id=bound_instance.bound_node_id,
        provider_id=provider_id,
        key_id=DELIVERY_KEY_ID,
    )
    task_params = _encrypted_task_params()
    metadata = task_params.verifiable
    assert metadata is not None
    task_params = task_params.model_copy(
        update={
            "verifiable": metadata.model_copy(
                update={
                    "recipient": recipient,
                    "placement_digest": placement_digest(
                        bound_instance.instance, recipient
                    ),
                }
            )
        }
    )

    receipt = build_verifiable_input_receipt(
        task_params,
        bound_instance,
        VerifiableInputSource.DecryptedEnvelope,
        execution_id=CommandId("execution-ingress-test"),
        delivery_private_key_loader=lambda: reporting_private_key,
        prompt_token_count=23,
    )

    assert receipt.reporting_provider_id == provider_id
    assert receipt.reporting_key_id == DELIVERY_KEY_ID
    assert receipt.recipient_provider_id == provider_id
    assert receipt.recipient_key_id == DELIVERY_KEY_ID
