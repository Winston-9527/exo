"""Deterministic baseline-versus-verifiable quality check CLI.

Run with ``python -m exo.verifiable.quality_check``. The JSON report contains
only hashes and lengths for requester/model text; the plaintext prompt is never
included.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections.abc import Sequence
from typing import cast
from uuid import uuid4

import httpx
from pydantic import Field, JsonValue, TypeAdapter

from exo.shared.types.common import CommandId, ModelId, NodeId
from exo.shared.types.state import State
from exo.shared.types.text_generation import InputMessage, InputMessageContent
from exo.shared.types.verifiable import (
    VerifiableAuditResponse,
    VerifiableChatCompletionRequest,
    VerifiableGenerationParams,
    VerifiableInputReceipt,
    VerifiableProviderIdentity,
)
from exo.shared.types.worker.instances import Instance, InstanceId
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.utils.pydantic_ext import FrozenModel
from exo.verifiable.client import build_verifiable_chat_request
from exo.verifiable.identity import DELIVERY_KEY_ID
from exo.verifiable.private_types import VerifiablePrivateTaskPayload

DEFAULT_MODEL = ModelId("mlx-community/Qwen3-0.6B-8bit")
JSON_VALUE_ADAPTER: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)
PROVIDER_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


class QualityCheckError(RuntimeError):
    """The public EXO API did not provide evidence required for comparison."""


class QualityCheckConfig(FrozenModel):
    model: ModelId = DEFAULT_MODEL
    prompt: str
    request_id: str = Field(default_factory=lambda: f"quality-{uuid4()}")
    instance_id: InstanceId | None = None
    expected_instance: Instance | None = None
    ingress_url: str | None = None
    max_output_tokens: int = Field(default=32, gt=0)
    seed: int = 42
    events_timeout_seconds: float = Field(default=15.0, ge=0.0)
    events_poll_interval_seconds: float = Field(default=0.2, gt=0.0)
    audit_timeout_seconds: float = Field(default=15.0, ge=0.0)
    audit_poll_interval_seconds: float = Field(default=0.2, gt=0.0)


class ContentFingerprint(FrozenModel):
    sha256: str
    utf8_bytes: int = Field(ge=0)


class GenerationEvidence(FrozenModel):
    command_id: str
    token_count: int = Field(ge=0)
    token_ids_sha256: str
    final_text: ContentFingerprint


class QualityComparison(FrozenModel):
    token_ids_exact: bool
    final_text_exact: bool
    first_token_mismatch_index: int | None = Field(default=None, ge=0)


class PlacementEvidence(FrozenModel):
    standard_api_exact_instance_binding: bool = False
    unique_model_instance_required: bool = True
    binding_method: str = "unique-model-instance-plus-task-created-event"
    baseline_instance_id: InstanceId
    verifiable_instance_id: InstanceId
    same_instance: bool


class AuditEvidence(FrozenModel):
    execution_id: CommandId
    expected_ranks: int = Field(ge=0)
    receipt_count: int = Field(ge=0)
    complete: bool
    bindings_valid: bool
    binding_errors: list[str]
    ingress_only_private_access: bool
    private_input_nodes: list[NodeId]
    shape_only_nodes: list[NodeId]
    receipts: list[VerifiableInputReceipt]


class QualityCheckReport(FrozenModel):
    protocol_version: str = "verifiable-exo-quality-v1"
    model: ModelId
    instance_id: InstanceId
    ingress_node_id: NodeId
    provider_id: str
    request_id: str
    seed: int
    temperature: float = 0.0
    max_output_tokens: int = Field(gt=0)
    prompt: ContentFingerprint
    baseline: GenerationEvidence
    verifiable: GenerationEvidence
    comparison: QualityComparison
    placement: PlacementEvidence
    audit: AuditEvidence
    passed: bool


def _sha256(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _content_fingerprint(value: str) -> ContentFingerprint:
    encoded = value.encode("utf-8")
    return ContentFingerprint(sha256=_sha256(encoded), utf8_bytes=len(encoded))


def _select_instance(state: State, config: QualityCheckConfig) -> Instance:
    matches = [
        instance
        for instance in state.instances.values()
        if instance.shard_assignments.model_id == config.model
    ]
    if len(matches) != 1:
        raise QualityCheckError(
            "Quality comparison requires exactly one matching model instance"
        )
    instance = matches[0]
    if config.instance_id is not None and instance.instance_id != config.instance_id:
        raise QualityCheckError("The requested instance does not exist for this model")
    if config.expected_instance is not None and instance != config.expected_instance:
        raise QualityCheckError("Live placement does not match the frozen instance")
    return instance


def _ingress_node(instance: Instance) -> NodeId:
    ingress_nodes: list[NodeId] = []
    for node_id, runner_id in instance.shard_assignments.node_to_runner.items():
        shard = instance.shard_assignments.runner_to_shard[runner_id]
        if not isinstance(shard, PipelineShardMetadata):
            raise QualityCheckError(
                "Quality comparison requires pure pipeline sharding"
            )
        if shard.is_first_layer:
            ingress_nodes.append(node_id)
    if len(ingress_nodes) != 1:
        raise QualityCheckError("Pipeline placement must have exactly one ingress")
    return ingress_nodes[0]


def _pipeline_world_size(instance: Instance) -> int:
    shards = list(instance.shard_assignments.runner_to_shard.values())
    if not shards or not all(
        isinstance(shard, PipelineShardMetadata) for shard in shards
    ):
        raise QualityCheckError("Quality comparison requires pure pipeline sharding")
    pipeline_shards = cast(list[PipelineShardMetadata], shards)
    world_sizes = {shard.world_size for shard in pipeline_shards}
    if len(world_sizes) != 1:
        raise QualityCheckError("Pipeline placement has inconsistent world sizes")
    world_size = next(iter(world_sizes))
    ranks = {shard.device_rank for shard in pipeline_shards}
    if len(pipeline_shards) != world_size or ranks != set(range(world_size)):
        raise QualityCheckError("Pipeline placement does not contain every unique rank")
    first_shards = [shard for shard in pipeline_shards if shard.is_first_layer]
    if len(first_shards) != 1 or first_shards[0].device_rank != 0:
        raise QualityCheckError("Pipeline ingress must be the unique rank zero shard")
    return world_size


def _decode_json(response: httpx.Response) -> JsonValue:
    return JSON_VALUE_ADAPTER.validate_json(response.content)


def _response_json(response: httpx.Response) -> dict[str, JsonValue]:
    response.raise_for_status()
    payload = _decode_json(response)
    if not isinstance(payload, dict):
        raise QualityCheckError("EXO returned a non-object response")
    return payload


def _ingress_endpoint(config: QualityCheckConfig, path: str) -> str:
    if config.ingress_url is None:
        return path
    return f"{config.ingress_url.rstrip('/')}{path}"


def _chat_result(payload: dict[str, JsonValue]) -> tuple[str, str]:
    command_id = payload.get("id")
    choices = payload.get("choices")
    if not isinstance(command_id, str) or not isinstance(choices, list) or not choices:
        raise QualityCheckError("Chat response is missing its command id or choice")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise QualityCheckError("Chat response choice has an invalid shape")
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise QualityCheckError("Chat response is missing its assistant message")
    reasoning = message.get("reasoning_content", "")
    content = message.get("content", "")
    if reasoning is None:
        reasoning = ""
    if content is None:
        content = ""
    if not isinstance(reasoning, str) or not isinstance(content, str):
        raise QualityCheckError("Quality check supports textual chat responses only")
    # Preserve the two OpenAI response channels while comparing their text.
    final_text = json.dumps(
        {"reasoning_content": reasoning, "content": content},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return command_id, final_text


def _field(
    payload: dict[str, JsonValue], snake_name: str, camel_name: str
) -> JsonValue:
    return payload.get(snake_name, payload.get(camel_name))


def _token_ids_for_command(events: JsonValue, command_id: str) -> list[int]:
    if not isinstance(events, list):
        raise QualityCheckError("EXO /events did not return a list")
    token_ids: list[int] = []
    terminal_chunk_count = 0
    saw_non_token_generation_chunk = False
    last_unique_token_was_terminal = False
    seen_chunk_events: dict[str, str] = {}
    for untyped_event in events:
        if not isinstance(untyped_event, dict):
            continue
        event = untyped_event.get("ChunkGenerated")
        if not isinstance(event, dict):
            continue
        if _field(event, "command_id", "commandId") != command_id:
            continue
        chunk_container = event.get("chunk")
        if not isinstance(chunk_container, dict):
            continue
        canonical_event = json.dumps(
            event,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        event_id = event.get("event_id")
        event_identity = (
            f"event-id:{event_id}" if isinstance(event_id, str) else canonical_event
        )
        previous_event = seen_chunk_events.get(event_identity)
        if previous_event is not None:
            if previous_event != canonical_event:
                raise QualityCheckError(
                    "A token event id was reused for a conflicting payload"
                )
            continue
        seen_chunk_events[event_identity] = canonical_event
        chunk = chunk_container.get("TokenChunk")
        if not isinstance(chunk, dict):
            if any(
                isinstance(chunk_container.get(chunk_type), dict)
                for chunk_type in ("ErrorChunk", "ToolCallChunk")
            ):
                saw_non_token_generation_chunk = True
            continue
        token_id = _field(chunk, "token_id", "tokenId")
        if not isinstance(token_id, int) or isinstance(token_id, bool):
            raise QualityCheckError("TokenChunk contained an invalid token id")
        token_ids.append(token_id)
        last_unique_token_was_terminal = (
            _field(chunk, "finish_reason", "finishReason") is not None
        )
        if last_unique_token_was_terminal:
            terminal_chunk_count += 1
    if not token_ids:
        raise QualityCheckError(
            "No TokenChunk events were found for a completed request"
        )
    if saw_non_token_generation_chunk:
        raise QualityCheckError(
            "Quality comparison requires token output without error/tool chunks"
        )
    if terminal_chunk_count != 1:
        raise QualityCheckError(
            "A completed request must have exactly one unique terminal TokenChunk"
        )
    if not last_unique_token_was_terminal:
        raise QualityCheckError("Token output continued after its terminal chunk")
    _require_completed_task(events, command_id)
    return token_ids


def _task_ids_for_command(events: JsonValue, command_id: str) -> set[str]:
    if not isinstance(events, list):
        raise QualityCheckError("EXO /events did not return a list")
    task_ids: set[str] = set()
    for untyped_event in events:
        if not isinstance(untyped_event, dict):
            continue
        created = untyped_event.get("TaskCreated")
        if not isinstance(created, dict):
            continue
        task_container = created.get("task")
        if not isinstance(task_container, dict):
            continue
        task = task_container.get("TextGeneration")
        if not isinstance(task, dict):
            continue
        if _field(task, "command_id", "commandId") != command_id:
            continue
        task_id = _field(created, "task_id", "taskId")
        if isinstance(task_id, str):
            task_ids.add(task_id)
    return task_ids


def _require_completed_task(events: JsonValue, command_id: str) -> None:
    if not isinstance(events, list):
        raise QualityCheckError("EXO /events did not return a list")
    task_ids = _task_ids_for_command(events, command_id)
    if len(task_ids) != 1:
        raise QualityCheckError(
            "A command must bind to exactly one unique text-generation task"
        )
    task_id = next(iter(task_ids))
    statuses: set[str] = set()
    for untyped_event in events:
        if not isinstance(untyped_event, dict):
            continue
        updated = untyped_event.get("TaskStatusUpdated")
        if not isinstance(updated, dict):
            continue
        if _field(updated, "task_id", "taskId") != task_id:
            continue
        status = _field(updated, "task_status", "taskStatus")
        if isinstance(status, str):
            statuses.add(status)
    failed_statuses = statuses.intersection({"Failed", "TimedOut", "Cancelled"})
    if failed_statuses:
        raise QualityCheckError("The text-generation task terminated unsuccessfully")
    if "Complete" not in statuses:
        raise QualityCheckError(
            "The text-generation task has no complete terminal status"
        )


def _task_instance_for_command(events: JsonValue, command_id: str) -> InstanceId:
    if not isinstance(events, list):
        raise QualityCheckError("EXO /events did not return a list")
    matching_instances: set[InstanceId] = set()
    for untyped_event in events:
        if not isinstance(untyped_event, dict):
            continue
        event = untyped_event.get("TaskCreated")
        if not isinstance(event, dict):
            continue
        task_container = event.get("task")
        if not isinstance(task_container, dict):
            continue
        task = task_container.get("TextGeneration")
        if not isinstance(task, dict):
            continue
        if _field(task, "command_id", "commandId") != command_id:
            continue
        instance_id = _field(task, "instance_id", "instanceId")
        if isinstance(instance_id, str):
            matching_instances.add(InstanceId(instance_id))
    if len(matching_instances) != 1:
        raise QualityCheckError(
            "A completed request must have exactly one unique TaskCreated instance binding"
        )
    return next(iter(matching_instances))


def _token_ids_digest(token_ids: list[int]) -> str:
    canonical = json.dumps(token_ids, separators=(",", ":")).encode("ascii")
    return _sha256(canonical)


def _first_mismatch(left: list[int], right: list[int]) -> int | None:
    for index, (left_token, right_token) in enumerate(zip(left, right, strict=False)):
        if left_token != right_token:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _audit_evidence(
    audit: VerifiableAuditResponse,
    *,
    instance: Instance,
    encrypted_request: VerifiableChatCompletionRequest,
    identity: VerifiableProviderIdentity,
    ingress_node_id: NodeId,
    expected_world_size: int,
    verifiable_execution_id: CommandId,
) -> AuditEvidence:
    expected_by_rank: dict[int, tuple[NodeId, PipelineShardMetadata]] = {}
    binding_errors: list[str] = []
    for node_id, runner_id in instance.shard_assignments.node_to_runner.items():
        shard = instance.shard_assignments.runner_to_shard[runner_id]
        if not isinstance(shard, PipelineShardMetadata):
            binding_errors.append("placement_not_pure_pipeline")
            continue
        if shard.device_rank in expected_by_rank:
            binding_errors.append("placement_duplicate_rank")
            continue
        expected_by_rank[shard.device_rank] = (node_id, shard)

    if len(expected_by_rank) != expected_world_size or set(expected_by_rank) != set(
        range(expected_world_size)
    ):
        binding_errors.append("placement_rank_world_size_mismatch")
    expected_ciphertext_digest = _sha256(
        encrypted_request.encrypted_input.ciphertext.encode("ascii")
    )
    if audit.request_id != encrypted_request.request_id:
        binding_errors.append("audit_request_id_mismatch")
    if audit.execution_id != verifiable_execution_id:
        binding_errors.append("audit_execution_id_mismatch")
    if audit.expected_ranks != expected_world_size:
        binding_errors.append("audit_world_size_mismatch")
    if (
        encrypted_request.recipient.node_id != ingress_node_id
        or identity.node_id != ingress_node_id
        or identity.provider_id != encrypted_request.recipient.provider_id
        or identity.key_id != encrypted_request.recipient.key_id
    ):
        binding_errors.append("recipient_provider_node_mismatch")

    seen_ranks: set[int] = set()
    prompt_token_counts: set[int] = set()
    reporting_provider_nodes: dict[str, NodeId] = {}
    node_reporting_identities: dict[NodeId, tuple[str, str]] = {}
    for receipt in audit.receipts:
        rank = receipt.device_rank
        if rank in seen_ranks:
            binding_errors.append("duplicate_receipt_rank")
        seen_ranks.add(rank)
        prompt_token_counts.add(receipt.prompt_token_count)

        if receipt.request_id != encrypted_request.request_id:
            binding_errors.append("receipt_request_id_mismatch")
        if (
            receipt.execution_id != audit.execution_id
            or receipt.execution_id != verifiable_execution_id
        ):
            binding_errors.append("receipt_execution_id_mismatch")
        if receipt.instance_id != str(instance.instance_id):
            binding_errors.append("receipt_instance_id_mismatch")
        if receipt.placement_digest != encrypted_request.placement_digest:
            binding_errors.append("receipt_placement_digest_mismatch")
        if receipt.ciphertext_digest != expected_ciphertext_digest:
            binding_errors.append("receipt_ciphertext_digest_mismatch")
        if receipt.world_size != expected_world_size:
            binding_errors.append("receipt_world_size_mismatch")
        if (
            receipt.recipient_provider_id != encrypted_request.recipient.provider_id
            or receipt.recipient_key_id != encrypted_request.recipient.key_id
        ):
            binding_errors.append("receipt_recipient_identity_mismatch")
        if PROVIDER_ID_PATTERN.fullmatch(receipt.reporting_provider_id) is None:
            binding_errors.append("receipt_reporting_fingerprint_invalid")
        if receipt.reporting_key_id != DELIVERY_KEY_ID:
            binding_errors.append("receipt_reporting_key_id_invalid")

        previous_node = reporting_provider_nodes.setdefault(
            receipt.reporting_provider_id, receipt.node_id
        )
        if previous_node != receipt.node_id:
            binding_errors.append("reporting_provider_reused_across_nodes")
        reporting_identity = (
            receipt.reporting_provider_id,
            receipt.reporting_key_id,
        )
        previous_identity = node_reporting_identities.setdefault(
            receipt.node_id, reporting_identity
        )
        if previous_identity != reporting_identity:
            binding_errors.append("node_reported_multiple_provider_identities")

        # Recipient identity describes the envelope on every rank. Reporting
        # identity describes the provider that produced this particular receipt.
        if receipt.device_rank == 0 and (
            receipt.reporting_provider_id != encrypted_request.recipient.provider_id
            or receipt.reporting_key_id != encrypted_request.recipient.key_id
        ):
            binding_errors.append("ingress_reporting_identity_mismatch")
        if (
            receipt.device_rank != 0
            and receipt.reporting_provider_id == encrypted_request.recipient.provider_id
        ):
            binding_errors.append("downstream_impersonates_ingress")

        expected_rank = expected_by_rank.get(rank)
        if expected_rank is None:
            binding_errors.append("unexpected_receipt_rank")
            continue
        expected_node_id, expected_shard = expected_rank
        if receipt.node_id != expected_node_id:
            binding_errors.append("receipt_rank_node_mismatch")
        if (
            receipt.start_layer != expected_shard.start_layer
            or receipt.end_layer != expected_shard.end_layer
        ):
            binding_errors.append("receipt_rank_layers_mismatch")

    if len(prompt_token_counts) > 1:
        binding_errors.append("receipt_prompt_length_mismatch")

    ranks = {receipt.device_rank for receipt in audit.receipts}
    private_nodes = sorted(
        {
            receipt.node_id
            for receipt in audit.receipts
            if receipt.private_input_accessed
        }
    )
    dummy_nodes = sorted(
        {
            receipt.node_id
            for receipt in audit.receipts
            if receipt.input_source == "shape_only_dummy"
        }
    )
    complete = len(audit.receipts) == expected_world_size and ranks == set(
        range(expected_world_size)
    )
    ingress_only = private_nodes == [ingress_node_id] and all(
        (
            receipt.node_id == ingress_node_id
            and receipt.private_input_accessed
            and receipt.input_source == "decrypted_envelope"
        )
        or (
            receipt.node_id != ingress_node_id
            and not receipt.private_input_accessed
            and receipt.input_source == "shape_only_dummy"
        )
        for receipt in audit.receipts
    )
    return AuditEvidence(
        execution_id=audit.execution_id,
        expected_ranks=expected_world_size,
        receipt_count=len(audit.receipts),
        complete=complete,
        bindings_valid=not binding_errors,
        binding_errors=sorted(set(binding_errors)),
        ingress_only_private_access=ingress_only,
        private_input_nodes=private_nodes,
        shape_only_nodes=dummy_nodes,
        receipts=audit.receipts,
    )


def _fetch_audit(
    client: httpx.Client,
    config: QualityCheckConfig,
    *,
    expected_ranks: int,
    execution_id: CommandId,
) -> VerifiableAuditResponse:
    deadline = time.monotonic() + config.audit_timeout_seconds
    latest: VerifiableAuditResponse | None = None
    while True:
        response = client.get(f"/v1/verifiable/audit/{config.request_id}")
        if response.status_code == 200:
            latest = VerifiableAuditResponse.model_validate(_decode_json(response))
            unique_ranks = {receipt.device_rank for receipt in latest.receipts}
            if len(unique_ranks) >= expected_ranks:
                return latest
        elif response.status_code != 404:
            response.raise_for_status()

        if time.monotonic() >= deadline:
            if latest is not None:
                return latest
            return VerifiableAuditResponse(
                request_id=config.request_id,
                execution_id=execution_id,
                expected_ranks=expected_ranks,
                receipts=[],
            )
        time.sleep(config.audit_poll_interval_seconds)


def _fetch_events(
    client: httpx.Client,
    config: QualityCheckConfig,
    *,
    command_ids: Sequence[str],
) -> JsonValue:
    deadline = time.monotonic() + config.events_timeout_seconds
    latest_error: QualityCheckError | None = None
    while True:
        response = client.get("/events")
        response.raise_for_status()
        events = _decode_json(response)
        if not isinstance(events, list):
            raise QualityCheckError("EXO /events did not return a list")
        try:
            for command_id in command_ids:
                _token_ids_for_command(events, command_id)
                _task_instance_for_command(events, command_id)
        except QualityCheckError as error:
            latest_error = error
        else:
            return events

        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            raise latest_error
        time.sleep(min(config.events_poll_interval_seconds, remaining_seconds))


def _assert_request_id_fresh(client: httpx.Client, request_id: str) -> None:
    response = client.get(f"/v1/verifiable/audit/{request_id}")
    if response.status_code == 404:
        return
    if response.status_code == 200:
        raise QualityCheckError("The request id already has audit state")
    response.raise_for_status()


def run_quality_check(
    client: httpx.Client,
    config: QualityCheckConfig,
) -> QualityCheckReport:
    """Compare standard and encrypted EXO requests using public API evidence."""
    _assert_request_id_fresh(client, config.request_id)
    state_response = client.get("/state")
    state_response.raise_for_status()
    # State is strict; JSON-mode validation preserves its enum/datetime wire conversions.
    state = State.model_validate_json(state_response.content)
    instance = _select_instance(state, config)
    ingress_node_id = _ingress_node(instance)
    expected_ranks = _pipeline_world_size(instance)

    identity_endpoint = _ingress_endpoint(config, "/v1/verifiable/identity")
    identity = VerifiableProviderIdentity.model_validate(
        _response_json(client.get(identity_endpoint))
    )
    if identity.node_id != ingress_node_id:
        raise QualityCheckError(
            "Identity endpoint does not belong to placement ingress"
        )

    baseline_response = _response_json(
        client.post(
            "/v1/chat/completions",
            json={
                "model": str(config.model),
                "messages": [{"role": "user", "content": config.prompt}],
                "max_tokens": config.max_output_tokens,
                "temperature": 0.0,
                "seed": config.seed,
                "stream": False,
                # Match the verifiable path, which disables prompt-history
                # processors because downstream ranks hold dummy prompt IDs.
                "repetition_penalty": 1.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
            },
        )
    )
    baseline_command_id, baseline_text = _chat_result(baseline_response)

    encrypted_request = build_verifiable_chat_request(
        instance=instance,
        identity=identity,
        private_payload=VerifiablePrivateTaskPayload(
            input=[
                InputMessage(role="user", content=InputMessageContent(config.prompt))
            ]
        ),
        generation=VerifiableGenerationParams(
            max_output_tokens=config.max_output_tokens,
            temperature=0.0,
            seed=config.seed,
            stream=False,
        ),
        request_id=config.request_id,
    )
    verifiable_response = _response_json(
        client.post(
            _ingress_endpoint(config, "/v1/verifiable/chat/completions"),
            json=encrypted_request.model_dump(mode="json", by_alias=True),
        )
    )
    verifiable_command_id, verifiable_text = _chat_result(verifiable_response)

    events = _fetch_events(
        client,
        config,
        command_ids=(baseline_command_id, verifiable_command_id),
    )
    baseline_tokens = _token_ids_for_command(events, baseline_command_id)
    verifiable_tokens = _token_ids_for_command(events, verifiable_command_id)
    baseline_instance_id = _task_instance_for_command(events, baseline_command_id)
    verifiable_instance_id = _task_instance_for_command(events, verifiable_command_id)
    placement = PlacementEvidence(
        baseline_instance_id=baseline_instance_id,
        verifiable_instance_id=verifiable_instance_id,
        same_instance=(
            baseline_instance_id == instance.instance_id
            and verifiable_instance_id == instance.instance_id
        ),
    )

    first_mismatch = _first_mismatch(baseline_tokens, verifiable_tokens)
    comparison = QualityComparison(
        token_ids_exact=first_mismatch is None,
        final_text_exact=baseline_text == verifiable_text,
        first_token_mismatch_index=first_mismatch,
    )
    raw_audit = _fetch_audit(
        client,
        config,
        expected_ranks=expected_ranks,
        execution_id=CommandId(verifiable_command_id),
    )
    audit = _audit_evidence(
        raw_audit,
        instance=instance,
        encrypted_request=encrypted_request,
        identity=identity,
        ingress_node_id=ingress_node_id,
        expected_world_size=expected_ranks,
        verifiable_execution_id=CommandId(verifiable_command_id),
    )

    postflight_state_response = client.get("/state")
    postflight_state_response.raise_for_status()
    postflight_state = State.model_validate_json(postflight_state_response.content)
    postflight_instance = _select_instance(postflight_state, config)
    if postflight_instance != instance:
        raise QualityCheckError(
            "Live placement changed while quality evidence was collected"
        )

    baseline_evidence = GenerationEvidence(
        command_id=baseline_command_id,
        token_count=len(baseline_tokens),
        token_ids_sha256=_token_ids_digest(baseline_tokens),
        final_text=_content_fingerprint(baseline_text),
    )
    verifiable_evidence = GenerationEvidence(
        command_id=verifiable_command_id,
        token_count=len(verifiable_tokens),
        token_ids_sha256=_token_ids_digest(verifiable_tokens),
        final_text=_content_fingerprint(verifiable_text),
    )
    passed = (
        comparison.token_ids_exact
        and comparison.final_text_exact
        and placement.same_instance
        and audit.complete
        and audit.bindings_valid
        and audit.ingress_only_private_access
    )
    return QualityCheckReport(
        model=config.model,
        instance_id=instance.instance_id,
        ingress_node_id=ingress_node_id,
        provider_id=identity.provider_id,
        request_id=config.request_id,
        seed=config.seed,
        max_output_tokens=config.max_output_tokens,
        prompt=_content_fingerprint(config.prompt),
        baseline=baseline_evidence,
        verifiable=verifiable_evidence,
        comparison=comparison,
        placement=placement,
        audit=audit,
        passed=passed,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare deterministic standard and encrypted EXO inference without "
            "printing the requester prompt"
        )
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:52415")
    parser.add_argument(
        "--ingress-url",
        help=(
            "API base URL used for ingress identity and encrypted POST; "
            "defaults to --base-url"
        ),
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--instance-id")
    prompt_source = parser.add_mutually_exclusive_group()
    prompt_source.add_argument(
        "--prompt",
        help="Prompt text (prefer stdin or --prompt-file to avoid process listings)",
    )
    prompt_source.add_argument("--prompt-file")
    parser.add_argument("--request-id")
    parser.add_argument("--max-output-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--events-timeout", type=float, default=15.0)
    parser.add_argument("--audit-timeout", type=float, default=15.0)
    return parser


def _read_prompt(args: argparse.Namespace) -> str:
    prompt = cast(str | None, args.prompt)
    prompt_file = cast(str | None, args.prompt_file)
    if prompt is not None:
        return prompt
    if prompt_file is not None:
        with open(prompt_file, encoding="utf-8") as file:
            return file.read()
    return sys.stdin.read()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    prompt = _read_prompt(args)
    request_id = cast(str | None, args.request_id) or f"quality-{uuid4()}"
    instance_arg = cast(str | None, args.instance_id)
    ingress_url = cast(str | None, args.ingress_url)
    config = QualityCheckConfig(
        model=ModelId(cast(str, args.model)),
        prompt=prompt,
        request_id=request_id,
        instance_id=InstanceId(instance_arg) if instance_arg is not None else None,
        ingress_url=ingress_url,
        max_output_tokens=cast(int, args.max_output_tokens),
        seed=cast(int, args.seed),
        events_timeout_seconds=cast(float, args.events_timeout),
        audit_timeout_seconds=cast(float, args.audit_timeout),
    )
    try:
        with httpx.Client(
            base_url=cast(str, args.base_url),
            timeout=cast(float, args.timeout),
            follow_redirects=True,
        ) as client:
            report = run_quality_check(client, config)
    except Exception as error:
        # Avoid echoing server bodies: an upstream error could contain private input.
        failure: dict[str, JsonValue] = {
            "protocol_version": "verifiable-exo-quality-v1",
            "passed": False,
            "error_type": type(error).__name__,
        }
        if isinstance(error, QualityCheckError):
            failure["constraint"] = str(error)
        print(
            json.dumps(
                failure,
                separators=(",", ":"),
            )
        )
        return 2

    print(report.model_dump_json(indent=2))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
