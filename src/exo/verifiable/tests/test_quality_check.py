"""Requester-side deterministic quality comparison through public EXO APIs."""

import hashlib

import httpx
import pytest
from pydantic import JsonValue

from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.chunks import TokenChunk
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.events import ChunkGenerated, TaskCreated
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.tasks import TaskId
from exo.shared.types.tasks import TextGeneration as TextGenerationTask
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.verifiable import (
    VerifiableAuditResponse,
    VerifiableChatCompletionRequest,
    VerifiableInputReceipt,
    VerifiableProviderIdentity,
)
from exo.shared.types.worker.instances import InstanceId, MlxRingInstance
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.verifiable.crypto import delivery_public_key, generate_delivery_private_key
from exo.verifiable.identity import DELIVERY_KEY_ID, provider_id_from_public_key
from exo.verifiable.quality_check import (
    QualityCheckConfig,
    QualityCheckError,
    _task_instance_for_command,  # pyright: ignore[reportPrivateUsage]
    run_quality_check,
)

MODEL = ModelId("mlx-community/Qwen3-0.6B-8bit")
PROMPT = "Keep this requester prompt out of the JSON quality report."


def _instance() -> MlxRingInstance:
    model = ModelCard(
        model_id=MODEL,
        storage_size=Memory.from_bytes(1),
        n_layers=28,
        hidden_size=1024,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxCuda],
    )
    ingress_runner = RunnerId("runner-ingress")
    downstream_runner = RunnerId("runner-downstream")
    return MlxRingInstance(
        instance_id=InstanceId("quality-instance"),
        shard_assignments=ShardAssignments(
            model_id=MODEL,
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
                NodeId("node-ingress"): ingress_runner,
                NodeId("node-downstream"): downstream_runner,
            },
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )


def _chat_response(command_id: str) -> dict[str, object]:
    return {
        "id": command_id,
        "object": "chat.completion",
        "created": 1,
        "model": str(MODEL),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "same answer"},
                "finish_reason": "stop",
            }
        ],
    }


def _events(
    instance_id: InstanceId, *, baseline_instance_id: InstanceId | None = None
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for command_id in ("baseline-command", "verifiable-command"):
        task_id = TaskId(f"task-{command_id}")
        created = TaskCreated(
            task_id=task_id,
            task=TextGenerationTask(
                task_id=task_id,
                instance_id=(
                    baseline_instance_id
                    if command_id == "baseline-command"
                    and baseline_instance_id is not None
                    else instance_id
                ),
                command_id=CommandId(command_id),
                task_params=TextGenerationTaskParams(model=MODEL, input=[]),
            ),
        )
        # Distributed event forwarding can surface the same TaskCreated event
        # through the master and both workers. Every copy must bind to the same
        # instance, but duplicate copies are not ambiguous by themselves.
        events.extend([created.model_dump(mode="json")] * 3)
        for text, token_id, is_final in (
            ("same ", 101, False),
            ("answer", 202, True),
        ):
            event = ChunkGenerated(
                command_id=CommandId(command_id),
                chunk=TokenChunk(
                    model=MODEL,
                    text=text,
                    token_id=token_id,
                    usage=None,
                    finish_reason="stop" if is_final else None,
                ),
            )
            events.append(event.model_dump(mode="json"))
    return events


def test_task_instance_binding_deduplicates_only_consistent_event_copies() -> None:
    command_id = "replicated-command"
    expected_instance = InstanceId("expected-instance")
    replicated_event: JsonValue = {
        "TaskCreated": {
            "task": {
                "TextGeneration": {
                    "command_id": command_id,
                    "instance_id": str(expected_instance),
                }
            }
        }
    }
    replicated_events: JsonValue = [
        replicated_event,
        replicated_event,
        replicated_event,
    ]

    assert (
        _task_instance_for_command(replicated_events, command_id) == expected_instance
    )

    conflicting_event: JsonValue = {
        "TaskCreated": {
            "task": {
                "TextGeneration": {
                    "command_id": command_id,
                    "instance_id": "conflicting-instance",
                }
            }
        }
    }
    conflicting_events: JsonValue = [replicated_event, conflicting_event]
    with pytest.raises(QualityCheckError, match="exactly one unique"):
        _task_instance_for_command(
            conflicting_events,
            command_id,
        )


@pytest.mark.parametrize(
    "tampered_binding",
    [
        None,
        "request_id",
        "instance_id",
        "placement_digest",
        "ciphertext_digest",
        "world_size",
        "duplicate_rank",
        "node_id",
        "provider_id",
        "ingress_reporting_provider_id",
        "recipient_provider_id",
        "recipient_key_id",
        "receipt_execution_id",
        "audit_execution_id",
        "reporting_fingerprint",
        "reporting_key_id",
        "baseline_instance",
    ],
)
def test_quality_check_rejects_unbound_audit_receipts_without_reporting_prompt(
    tampered_binding: str | None,
) -> None:
    instance = _instance()
    private_key = generate_delivery_private_key()
    public_key = delivery_public_key(private_key)
    provider_id = provider_id_from_public_key(public_key)
    downstream_provider_id = provider_id_from_public_key(
        delivery_public_key(generate_delivery_private_key())
    )
    identity = VerifiableProviderIdentity(
        node_id=NodeId("node-ingress"),
        provider_id=provider_id,
        key_id=DELIVERY_KEY_ID,
        public_key=public_key,
    )
    encrypted_wire_bodies: list[str] = []
    encrypted_requests: list[VerifiableChatCompletionRequest] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/state":
            assert request.url.host == "control.test"
            state = State(
                instances={instance.instance_id: instance},
                node_backends={
                    NodeId("node-ingress"): [Backend.MlxCuda],
                    NodeId("node-downstream"): [Backend.MlxMetal],
                },
            )
            return httpx.Response(
                200, json=state.model_dump(mode="json", by_alias=True)
            )
        if request.method == "GET" and request.url.path == "/v1/verifiable/identity":
            assert request.url.host == "ingress.test"
            return httpx.Response(200, json=identity.model_dump(mode="json"))
        if request.method == "POST" and request.url.path == "/v1/chat/completions":
            assert request.url.host == "control.test"
            assert PROMPT in request.content.decode()
            return httpx.Response(200, json=_chat_response("baseline-command"))
        if (
            request.method == "POST"
            and request.url.path == "/v1/verifiable/chat/completions"
        ):
            assert request.url.host == "ingress.test"
            encrypted_wire_bodies.append(request.content.decode())
            encrypted_requests.append(
                VerifiableChatCompletionRequest.model_validate_json(request.content)
            )
            return httpx.Response(200, json=_chat_response("verifiable-command"))
        if request.method == "GET" and request.url.path == "/events":
            assert request.url.host == "control.test"
            return httpx.Response(
                200,
                json=_events(
                    instance.instance_id,
                    baseline_instance_id=(
                        InstanceId("unexpected-baseline-instance")
                        if tampered_binding == "baseline_instance"
                        else None
                    ),
                ),
            )
        if (
            request.method == "GET"
            and request.url.path == "/v1/verifiable/audit/quality-request"
        ):
            assert request.url.host == "control.test"
            if not encrypted_requests:
                return httpx.Response(404)
            encrypted_request = encrypted_requests[-1]
            ciphertext_digest = (
                "sha256:"
                + hashlib.sha256(
                    encrypted_request.encrypted_input.ciphertext.encode("ascii")
                ).hexdigest()
            )
            receipts = [
                VerifiableInputReceipt(
                    request_id="quality-request",
                    execution_id=CommandId("verifiable-command"),
                    instance_id=str(instance.instance_id),
                    placement_digest=encrypted_request.placement_digest,
                    node_id=NodeId("node-ingress"),
                    reporting_provider_id=(
                        downstream_provider_id
                        if tampered_binding == "ingress_reporting_provider_id"
                        else provider_id
                    ),
                    reporting_key_id=DELIVERY_KEY_ID,
                    recipient_provider_id=provider_id,
                    recipient_key_id=(
                        "unknown-key"
                        if tampered_binding == "recipient_key_id"
                        else DELIVERY_KEY_ID
                    ),
                    device_rank=0,
                    world_size=2,
                    start_layer=0,
                    end_layer=14,
                    input_source="decrypted_envelope",
                    private_input_accessed=True,
                    prompt_token_count=12,
                    ciphertext_digest=ciphertext_digest,
                ),
                VerifiableInputReceipt(
                    request_id=(
                        "replayed-other-request"
                        if tampered_binding == "request_id"
                        else "quality-request"
                    ),
                    execution_id=(
                        CommandId("other-execution")
                        if tampered_binding == "receipt_execution_id"
                        else CommandId("verifiable-command")
                    ),
                    instance_id=(
                        "other-instance"
                        if tampered_binding == "instance_id"
                        else str(instance.instance_id)
                    ),
                    placement_digest=(
                        "sha256:" + "0" * 64
                        if tampered_binding == "placement_digest"
                        else encrypted_request.placement_digest
                    ),
                    node_id=NodeId(
                        "node-ingress"
                        if tampered_binding == "node_id"
                        else "node-downstream"
                    ),
                    reporting_provider_id=(
                        provider_id
                        if tampered_binding == "provider_id"
                        else (
                            "not-a-provider-fingerprint"
                            if tampered_binding == "reporting_fingerprint"
                            else downstream_provider_id
                        )
                    ),
                    reporting_key_id=(
                        "unknown-key"
                        if tampered_binding == "reporting_key_id"
                        else DELIVERY_KEY_ID
                    ),
                    recipient_provider_id=(
                        "sha256:" + "0" * 64
                        if tampered_binding == "recipient_provider_id"
                        else provider_id
                    ),
                    recipient_key_id=DELIVERY_KEY_ID,
                    device_rank=0 if tampered_binding == "duplicate_rank" else 1,
                    world_size=3 if tampered_binding == "world_size" else 2,
                    start_layer=14,
                    end_layer=28,
                    input_source="shape_only_dummy",
                    private_input_accessed=False,
                    prompt_token_count=12,
                    ciphertext_digest=(
                        "sha256:" + "0" * 64
                        if tampered_binding == "ciphertext_digest"
                        else ciphertext_digest
                    ),
                ),
            ]
            audit = VerifiableAuditResponse(
                request_id="quality-request",
                execution_id=(
                    CommandId("other-execution")
                    if tampered_binding == "audit_execution_id"
                    else CommandId("verifiable-command")
                ),
                expected_ranks=2,
                receipts=receipts,
            )
            return httpx.Response(200, json=audit.model_dump(mode="json"))
        return httpx.Response(404)

    transport = httpx.MockTransport(handle)
    with httpx.Client(transport=transport, base_url="http://control.test") as client:
        report = run_quality_check(
            client,
            QualityCheckConfig(
                model=MODEL,
                prompt=PROMPT,
                request_id="quality-request",
                instance_id=instance.instance_id,
                ingress_url="http://ingress.test:52415",
                max_output_tokens=16,
                seed=42,
                audit_timeout_seconds=0.0,
            ),
        )

    serialized = report.model_dump_json()
    expected_pass = tampered_binding is None
    assert report.passed is expected_pass
    assert report.comparison.token_ids_exact is True
    assert report.comparison.final_text_exact is True
    assert report.placement.same_instance is (tampered_binding != "baseline_instance")
    assert report.placement.baseline_instance_id == (
        InstanceId("unexpected-baseline-instance")
        if tampered_binding == "baseline_instance"
        else instance.instance_id
    )
    assert report.placement.verifiable_instance_id == instance.instance_id
    assert report.placement.standard_api_exact_instance_binding is False
    assert report.placement.unique_model_instance_required is True
    assert report.audit.execution_id == (
        CommandId("other-execution")
        if tampered_binding == "audit_execution_id"
        else CommandId("verifiable-command")
    )
    assert report.audit.complete is (tampered_binding != "duplicate_rank")
    assert report.audit.bindings_valid is (
        tampered_binding is None or tampered_binding == "baseline_instance"
    )
    assert report.audit.private_input_nodes == [NodeId("node-ingress")]
    assert PROMPT not in serialized
    assert PROMPT not in "".join(encrypted_wire_bodies)
    assert report.prompt.sha256.startswith("sha256:")


def test_quality_check_rejects_ambiguous_standard_placement() -> None:
    selected = _instance()
    other = selected.model_copy(update={"instance_id": InstanceId("other-instance")})
    post_count = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal post_count
        if request.method == "GET" and request.url.path == "/state":
            state = State(
                instances={
                    selected.instance_id: selected,
                    other.instance_id: other,
                }
            )
            return httpx.Response(
                200, json=state.model_dump(mode="json", by_alias=True)
            )
        if request.method == "POST":
            post_count += 1
        return httpx.Response(404)

    with (
        httpx.Client(
            transport=httpx.MockTransport(handle), base_url="http://exo.test"
        ) as client,
        pytest.raises(QualityCheckError, match="exactly one matching"),
    ):
        run_quality_check(
            client,
            QualityCheckConfig(
                model=MODEL,
                prompt=PROMPT,
                instance_id=selected.instance_id,
            ),
        )

    assert post_count == 0


def test_quality_check_rejects_replayed_request_id_before_transmitting_prompt() -> None:
    post_count = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal post_count
        if (
            request.method == "GET"
            and request.url.path == "/v1/verifiable/audit/replayed-request"
        ):
            existing = VerifiableAuditResponse(
                request_id="replayed-request",
                execution_id=CommandId("old-execution"),
                expected_ranks=2,
                receipts=[],
            )
            return httpx.Response(200, json=existing.model_dump(mode="json"))
        if request.method == "POST":
            post_count += 1
        return httpx.Response(404)

    with (
        httpx.Client(
            transport=httpx.MockTransport(handle), base_url="http://exo.test"
        ) as client,
        pytest.raises(QualityCheckError, match="already has audit state"),
    ):
        run_quality_check(
            client,
            QualityCheckConfig(
                model=MODEL,
                prompt=PROMPT,
                request_id="replayed-request",
            ),
        )

    assert post_count == 0
