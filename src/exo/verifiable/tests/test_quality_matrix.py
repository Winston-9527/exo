"""Public-contract tests for the two-provider verifiable quality matrix."""

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import cast

import httpx
import pytest
from pydantic import JsonValue, TypeAdapter

import exo.verifiable.quality_matrix as quality_matrix
from exo.api.types import PlacementPreview
from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.common import CommandId, ModelId, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.verifiable import (
    VerifiableAuditResponse,
    VerifiableChatCompletionRequest,
    VerifiableInputReceipt,
    VerifiableRecipient,
)
from exo.shared.types.worker.instances import InstanceId, InstanceMeta, MlxRingInstance
from exo.shared.types.worker.runners import (
    RunnerFailed,
    RunnerId,
    RunnerLoading,
    RunnerReady,
    ShardAssignments,
)
from exo.shared.types.worker.shards import PipelineShardMetadata, Sharding
from exo.verifiable.crypto import delivery_public_key, generate_delivery_private_key
from exo.verifiable.identity import DELIVERY_KEY_ID, provider_id_from_public_key
from exo.verifiable.placement import placement_digest
from exo.verifiable.quality_matrix import (
    MAX_OUTPUT_TOKENS,
    NEGATIVE_CONTROLS,
    PROMPT_CLASSES,
    AttemptJournal,
    CleanAuditCellEvidence,
    LoadedPrompt,
    MatrixCell,
    MatrixEnvironment,
    MatrixNodeEndpoint,
    MatrixPromptFile,
    NegativeCellEvidence,
    PromptClass,
    PromptEvidence,
    QualityMatrixConfig,
    TokenizerEvidence,
    _receipts_match_frozen_placement,  # pyright: ignore[reportPrivateUsage]
    build_matrix_cells,
    execute_matrix_cell,
    inspect_matrix_environment,
    load_prompt_set,
    prepare_matrix_environment,
)

JSON_OBJECT_ADAPTER = TypeAdapter(dict[str, JsonValue])


def test_matrix_contains_the_locked_q_a_and_n_planes() -> None:
    cells = build_matrix_cells()

    assert len(cells) == 49
    assert Counter(cell.plane for cell in cells) == {
        "quality": 30,
        "clean_audit": 15,
        "negative": 4,
    }

    quality_coordinates = {
        (cell.prompt_class, cell.max_output_tokens, cell.repeat_index)
        for cell in cells
        if cell.plane == "quality"
    }
    assert quality_coordinates == {
        (prompt_class, max_output_tokens, repeat_index)
        for prompt_class in PROMPT_CLASSES
        for max_output_tokens in MAX_OUTPUT_TOKENS
        for repeat_index in range(3)
    }

    clean_audit_coordinates = {
        (cell.prompt_class, cell.repeat_index)
        for cell in cells
        if cell.plane == "clean_audit"
    }
    assert clean_audit_coordinates == {
        (prompt_class, repeat_index)
        for prompt_class in PROMPT_CLASSES
        for repeat_index in range(3)
    }

    assert {cell.negative_control for cell in cells if cell.plane == "negative"} == set(
        NEGATIVE_CONTROLS
    )
    assert len({cell.cell_id for cell in cells}) == len(cells)


def test_prompt_set_loads_all_five_classes_without_serializing_plaintext(
    tmp_path: Path,
) -> None:
    prompt_files: list[MatrixPromptFile] = []
    secrets: list[str] = []
    for index, prompt_class in enumerate(PROMPT_CLASSES):
        secret = f"private prompt {prompt_class} {index}"
        path = tmp_path / f"{prompt_class}.txt"
        path.write_text(secret, encoding="utf-8")
        prompt_files.append(
            MatrixPromptFile(
                prompt_class=prompt_class,
                path=str(path),
                target_prompt_tokens=(4096 if prompt_class == "prefill_4096" else None),
            )
        )
        secrets.append(secret)

    config = QualityMatrixConfig(
        base_url="http://control.test",
        nodes=[
            MatrixNodeEndpoint(label="mini1", api_url="http://mini.test"),
            MatrixNodeEndpoint(label="rtx3090", api_url="http://rtx.test"),
        ],
        instance_id="quality-instance",
        output_directory=str(tmp_path / "results"),
        run_id="matrix-run-001",
        prompts=prompt_files,
    )

    prompt_set = load_prompt_set(config)
    serialized_evidence = json.dumps(
        [item.evidence.model_dump(mode="json") for item in prompt_set.values()]
    )

    assert set(prompt_set) == set(PROMPT_CLASSES)
    assert all(item.plaintext for item in prompt_set.values())
    assert all(secret not in serialized_evidence for secret in secrets)
    assert all(str(tmp_path) not in serialized_evidence for _ in [0])


def test_attempt_journal_preserves_failures_and_uses_fresh_retry_ids(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "attempts.jsonl"
    journal = AttemptJournal(journal_path)
    cell = build_matrix_cells()[0]
    prompt_evidence = load_prompt_set(
        _config_with_prompts(tmp_path, output_directory=tmp_path / "results")
    )["ascii_short"].evidence

    first = journal.start("matrix-run-001", cell, prompt_evidence)
    journal.finish(
        first,
        status="infra_fail",
        failure_class="transport_timeout",
    )
    second = journal.start("matrix-run-001", cell, prompt_evidence)
    journal.finish(second, status="pass")

    resumed = AttemptJournal(journal_path)
    records = resumed.records()

    assert second.attempt_index == first.attempt_index + 1
    assert second.attempt_id != first.attempt_id
    assert second.request_id != first.request_id
    assert [record.status for record in records if record.event == "finished"] == [
        "infra_fail",
        "pass",
    ]
    assert resumed.terminal_cell_ids() == {cell.cell_id}
    assert "private prompt" not in journal_path.read_text(encoding="utf-8")


def test_new_run_rejects_stale_journal_before_environment_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_directory = tmp_path / "results"
    config = _config_with_prompts(tmp_path, output_directory=output_directory)
    stale_journal = AttemptJournal(output_directory / "attempts.jsonl")
    stale_journal.start("another-run", build_matrix_cells()[0], None)
    environment_touched = False

    def prepare(
        client: httpx.Client,
        received: QualityMatrixConfig,
    ) -> MatrixEnvironment:
        nonlocal environment_touched
        del client, received
        environment_touched = True
        return _matrix_environment()

    monkeypatch.setattr(quality_matrix, "prepare_matrix_environment", prepare)

    def execute_cell(
        client: httpx.Client,
        received: QualityMatrixConfig,
        received_environment: MatrixEnvironment,
        cell: MatrixCell,
        prompt: LoadedPrompt | None,
        request_id: str,
    ) -> quality_matrix.CellExecutionResult:
        del client, received, received_environment, cell, prompt, request_id
        return quality_matrix.CellExecutionResult(status="pass")

    with (
        _no_network_client(config) as client,
        pytest.raises(ValueError, match="artifacts already exist"),
    ):
        quality_matrix.run_quality_matrix(
            client,
            config,
            execute_cell=execute_cell,
        )

    assert environment_touched is False


def test_attempt_journal_rejects_foreign_unknown_and_unstarted_records(
    tmp_path: Path,
) -> None:
    run_id = "matrix-run-001"
    cells = build_matrix_cells()

    foreign = AttemptJournal(tmp_path / "foreign.jsonl")
    foreign.start("another-run", cells[0], None)
    with pytest.raises(ValueError, match="another run"):
        foreign.validate(run_id=run_id, cells=cells)

    unknown = AttemptJournal(tmp_path / "unknown.jsonl")
    unknown.start(
        run_id,
        MatrixCell(cell_id="unknown-cell", plane="quality"),
        None,
    )
    with pytest.raises(ValueError, match="unknown matrix cell"):
        unknown.validate(run_id=run_id, cells=cells)

    unfinished = AttemptJournal(tmp_path / "unstarted-finish.jsonl")
    finished_record = quality_matrix.AttemptRecord(
        run_id=run_id,
        cell_id=cells[0].cell_id,
        attempt_id="attempt-without-start",
        attempt_index=0,
        request_id="request-without-start",
        event="finished",
        timestamp="2026-07-30T00:00:00+00:00",
        status="pass",
    )
    unfinished.path.write_text(
        finished_record.model_dump_json() + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="before its start"):
        unfinished.validate(run_id=run_id, cells=cells)

    with pytest.raises(ValueError, match="outside the locked matrix run"):
        quality_matrix._summarize(  # pyright: ignore[reportPrivateUsage]
            run_id,
            cells,
            unknown.records(),
            formal_execution=True,
        )


def test_environment_binds_the_two_live_identities_to_the_committed_pipeline(
    tmp_path: Path,
) -> None:
    instance = _two_node_instance()
    identities = {
        NodeId("node-mini"): _identity(NodeId("node-mini")),
        NodeId("node-rtx"): _identity(NodeId("node-rtx")),
    }
    state = State(
        instances={instance.instance_id: instance},
        node_backends={
            NodeId("node-mini"): [Backend.MlxMetal],
            NodeId("node-rtx"): [Backend.MlxCuda],
        },
    )

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/state":
            return httpx.Response(
                200, json=state.model_dump(mode="json", by_alias=True)
            )
        node_id = {
            "mini.test": NodeId("node-mini"),
            "rtx.test": NodeId("node-rtx"),
        }[request.url.host]
        if request.url.path == "/node_id":
            return httpx.Response(200, json=str(node_id))
        if request.url.path == "/v1/verifiable/identity":
            return httpx.Response(
                200, json=identities[node_id].model_dump(mode="json", by_alias=True)
            )
        return httpx.Response(404)

    config = _config_with_prompts(tmp_path, output_directory=tmp_path / "results")
    with httpx.Client(
        transport=httpx.MockTransport(handle), base_url=config.base_url
    ) as client:
        environment = inspect_matrix_environment(client, config)

    assert environment.world_size == 2
    assert environment.ingress_node_id == NodeId("node-mini")
    assert environment.ingress_url == "http://mini.test"
    assert environment.downstream_node_id == NodeId("node-rtx")
    assert environment.downstream_url == "http://rtx.test"
    assert environment.identities == identities


def test_exact_preview_is_posted_and_live_rank_zero_selects_rtx_ingress(
    tmp_path: Path,
) -> None:
    instance = _two_node_instance(rtx_ingress=True)
    preview = PlacementPreview(
        model_id=instance.shard_assignments.model_id,
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        instance=instance,
    )
    preview_path = tmp_path / "exact-preview.json"
    preview_path.write_text(preview.model_dump_json(), encoding="utf-8")
    identities = {
        NodeId("node-mini"): _identity(NodeId("node-mini")),
        NodeId("node-rtx"): _identity(NodeId("node-rtx")),
    }
    created = False
    state_reads_after_creation = 0
    posted_instances: list[JsonValue] = []

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal created, state_reads_after_creation
        if request.method == "GET" and request.url.path == "/state":
            if created:
                state_reads_after_creation += 1
            runner_status = (
                RunnerReady()
                if state_reads_after_creation >= 3
                else RunnerLoading(layers_loaded=1, total_layers=28)
            )
            state = State(
                instances={instance.instance_id: instance} if created else {},
                runners={
                    runner_id: runner_status
                    for runner_id in instance.shard_assignments.runner_to_shard
                }
                if created
                else {},
                node_backends={
                    NodeId("node-mini"): [Backend.MlxMetal],
                    NodeId("node-rtx"): [Backend.MlxCuda],
                },
            )
            return httpx.Response(200, json=state.model_dump(mode="json"))
        if request.method == "POST" and request.url.path == "/instance":
            body = JSON_OBJECT_ADAPTER.validate_json(request.content)
            posted_instances.append(body["instance"])
            created = True
            return httpx.Response(200, json={"message": "accepted"})
        node_id = {
            "mini.test": NodeId("node-mini"),
            "rtx.test": NodeId("node-rtx"),
        }[request.url.host]
        if request.url.path == "/node_id":
            return httpx.Response(200, json=str(node_id))
        if request.url.path == "/v1/verifiable/identity":
            return httpx.Response(200, json=identities[node_id].model_dump(mode="json"))
        return httpx.Response(404)

    config = _config_with_prompts(
        tmp_path, output_directory=tmp_path / "results"
    ).model_copy(
        update={
            "instance_id": None,
            "placement_preview_file": str(preview_path),
        }
    )
    with httpx.Client(
        transport=httpx.MockTransport(handle), base_url=config.base_url
    ) as client:
        environment = prepare_matrix_environment(client, config)

    assert posted_instances == [instance.model_dump(mode="json", by_alias=True)]
    assert state_reads_after_creation >= 3
    assert environment.ingress_node_id == NodeId("node-rtx")
    assert environment.ingress_url == "http://rtx.test"
    assert environment.downstream_node_id == NodeId("node-mini")


def test_existing_instance_runner_failure_aborts_preflight_immediately(
    tmp_path: Path,
) -> None:
    instance = _two_node_instance()
    identities = {
        NodeId("node-mini"): _identity(NodeId("node-mini")),
        NodeId("node-rtx"): _identity(NodeId("node-rtx")),
    }
    state_reads = 0
    failed_runner = next(iter(instance.shard_assignments.runner_to_shard))

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal state_reads
        if request.method == "GET" and request.url.path == "/state":
            state_reads += 1
            state = State(
                instances={instance.instance_id: instance},
                runners={
                    runner_id: (
                        RunnerFailed(error_message="private diagnostic", diagnostics=[])
                        if runner_id == failed_runner
                        else RunnerReady()
                    )
                    for runner_id in instance.shard_assignments.runner_to_shard
                },
                node_backends={
                    NodeId("node-mini"): [Backend.MlxMetal],
                    NodeId("node-rtx"): [Backend.MlxCuda],
                },
            )
            return httpx.Response(200, json=state.model_dump(mode="json"))
        node_id = {
            "mini.test": NodeId("node-mini"),
            "rtx.test": NodeId("node-rtx"),
        }[request.url.host]
        if request.url.path == "/node_id":
            return httpx.Response(200, json=str(node_id))
        if request.url.path == "/v1/verifiable/identity":
            return httpx.Response(200, json=identities[node_id].model_dump(mode="json"))
        return httpx.Response(404)

    config = _config_with_prompts(
        tmp_path, output_directory=tmp_path / "results"
    ).model_copy(
        update={
            "placement_timeout_seconds": 10.0,
            "poll_interval_seconds": 0.001,
        }
    )
    with (
        httpx.Client(
            transport=httpx.MockTransport(handle), base_url=config.base_url
        ) as client,
        pytest.raises(RuntimeError, match="runner failed"),
    ):
        prepare_matrix_environment(client, config)

    assert state_reads <= 3


def test_execute_matrix_cell_clean_audit_uses_only_an_encrypted_canary() -> None:
    canary = "private clean-audit canary 7f8f09b3"
    request_id = "matrix-clean-audit-request"
    command_id = CommandId("clean-audit-command")
    environment = _matrix_environment()
    config = _matrix_config_without_private_files().model_copy(
        update={"event_timeout_seconds": 0.1}
    )
    cell = next(
        cell
        for cell in build_matrix_cells()
        if cell.plane == "clean_audit" and cell.prompt_class == "ascii_short"
    )
    loaded_prompt = _loaded_prompt("ascii_short", canary)
    encrypted_requests: list[VerifiableChatCompletionRequest] = []
    encrypted_wire_bodies: list[str] = []
    events_request_count = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal events_request_count
        if (
            request.method == "GET"
            and request.url.path == f"/v1/verifiable/audit/{request_id}"
            and not encrypted_requests
        ):
            return httpx.Response(404)
        if (
            request.method == "POST"
            and request.url.path == "/v1/verifiable/chat/completions"
        ):
            assert request.url.host == "mini.test"
            wire_body = request.content.decode("utf-8")
            encrypted_wire_bodies.append(wire_body)
            encrypted_requests.append(
                VerifiableChatCompletionRequest.model_validate_json(request.content)
            )
            return httpx.Response(
                200,
                json={
                    "id": str(command_id),
                    "choices": [
                        {"message": {"role": "assistant", "content": "safe output"}}
                    ],
                },
            )
        if (
            request.method == "GET"
            and request.url.path == f"/v1/verifiable/audit/{request_id}"
        ):
            encrypted_request = encrypted_requests[-1]
            return httpx.Response(
                200,
                json=_audit_for_request(
                    environment,
                    encrypted_request,
                    command_id,
                    prompt_token_count=11,
                ).model_dump(mode="json", by_alias=True),
            )
        if request.method == "GET" and request.url.path == "/events":
            events_request_count += 1
            encrypted_request = encrypted_requests[-1]
            audit = _audit_for_request(
                environment,
                encrypted_request,
                command_id,
                prompt_token_count=11,
            )
            events = _clean_audit_events(
                command_id,
                request_id,
                environment.instance.instance_id,
                complete=events_request_count > 1,
            )
            events.extend(_raw_receipt_events(audit))
            return httpx.Response(200, json=events)
        return httpx.Response(404)

    with httpx.Client(
        transport=httpx.MockTransport(handle),
        base_url=config.base_url,
    ) as client:
        result = execute_matrix_cell(
            client,
            config,
            environment,
            cell,
            loaded_prompt,
            request_id,
        )

    assert result.status == "pass"
    assert result.failure_class is None
    assert isinstance(result.evidence, CleanAuditCellEvidence)
    assert result.evidence.complete is True
    assert result.evidence.bindings_valid is True
    assert result.evidence.binding_errors == []
    assert result.evidence.ingress_only_private_access is True
    assert result.evidence.task_input_empty is True
    assert result.evidence.task_envelope_bound is True
    assert result.evidence.canary_absent_from_public_events is True
    assert len(encrypted_requests) == 1
    assert encrypted_requests[0].recipient.node_id == environment.ingress_node_id
    assert canary not in encrypted_wire_bodies[0]
    assert canary not in result.model_dump_json()
    assert events_request_count == 2


def test_clean_audit_never_passes_without_terminal_task_evidence() -> None:
    canary = "private incomplete clean-audit canary"
    request_id = "matrix-clean-audit-incomplete-events"
    command_id = CommandId("clean-audit-incomplete-command")
    environment = _matrix_environment()
    config = _matrix_config_without_private_files()
    cell = next(
        cell
        for cell in build_matrix_cells()
        if cell.plane == "clean_audit" and cell.prompt_class == "ascii_short"
    )
    loaded_prompt = _loaded_prompt("ascii_short", canary)
    encrypted_requests: list[VerifiableChatCompletionRequest] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if (
            request.method == "GET"
            and request.url.path == f"/v1/verifiable/audit/{request_id}"
            and not encrypted_requests
        ):
            return httpx.Response(404)
        if (
            request.method == "POST"
            and request.url.path == "/v1/verifiable/chat/completions"
        ):
            encrypted_requests.append(
                VerifiableChatCompletionRequest.model_validate_json(request.content)
            )
            return httpx.Response(
                200,
                json={
                    "id": str(command_id),
                    "choices": [
                        {"message": {"role": "assistant", "content": "safe output"}}
                    ],
                },
            )
        if (
            request.method == "GET"
            and request.url.path == f"/v1/verifiable/audit/{request_id}"
        ):
            audit = _audit_for_request(
                environment,
                encrypted_requests[-1],
                command_id,
                prompt_token_count=11,
            )
            return httpx.Response(200, json=audit.model_dump(mode="json"))
        if request.method == "GET" and request.url.path == "/events":
            return httpx.Response(
                200,
                json=_clean_audit_events(
                    command_id,
                    request_id,
                    environment.instance.instance_id,
                    complete=False,
                ),
            )
        return httpx.Response(404)

    with (
        httpx.Client(
            transport=httpx.MockTransport(handle), base_url=config.base_url
        ) as client,
        pytest.raises(TimeoutError, match="complete task output evidence"),
    ):
        execute_matrix_cell(
            client,
            config,
            environment,
            cell,
            loaded_prompt,
            request_id,
        )


def test_quality_cell_passes_the_frozen_instance_to_quality_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _matrix_environment()
    config = _matrix_config_without_private_files()
    cell = next(cell for cell in build_matrix_cells() if cell.plane == "quality")
    loaded_prompt = _loaded_prompt("ascii_short", "synthetic quality prompt")
    captured_expected_instances: list[object] = []

    class StopAfterConfigCaptureError(RuntimeError):
        pass

    def capture_config(
        client: httpx.Client,
        quality_config: quality_matrix.QualityCheckConfig,
    ) -> None:
        del client
        captured_expected_instances.append(quality_config.expected_instance)
        raise StopAfterConfigCaptureError

    monkeypatch.setattr(quality_matrix, "run_quality_check", capture_config)
    with (
        _no_network_client(config) as client,
        pytest.raises(StopAfterConfigCaptureError),
    ):
        execute_matrix_cell(
            client,
            config,
            environment,
            cell,
            loaded_prompt,
            "quality-frozen-instance-request",
        )

    assert captured_expected_instances == [environment.instance]


def test_quality_receipts_must_match_the_frozen_placement_digest() -> None:
    environment = _matrix_environment()
    ingress_identity = environment.identities[environment.ingress_node_id]
    recipient = VerifiableRecipient(
        node_id=ingress_identity.node_id,
        provider_id=ingress_identity.provider_id,
        key_id=ingress_identity.key_id,
    )
    frozen_digest = placement_digest(environment.instance, recipient)
    receipts: list[VerifiableInputReceipt] = []
    for (
        node_id,
        runner_id,
    ) in environment.instance.shard_assignments.node_to_runner.items():
        shard = environment.instance.shard_assignments.runner_to_shard[runner_id]
        assert isinstance(shard, PipelineShardMetadata)
        reporting_identity = environment.identities[node_id]
        is_ingress = node_id == environment.ingress_node_id
        receipts.append(
            VerifiableInputReceipt(
                request_id="frozen-placement-request",
                execution_id=CommandId("frozen-placement-command"),
                instance_id=str(environment.instance.instance_id),
                placement_digest=frozen_digest,
                node_id=node_id,
                reporting_provider_id=reporting_identity.provider_id,
                reporting_key_id=reporting_identity.key_id,
                recipient_provider_id=recipient.provider_id,
                recipient_key_id=recipient.key_id,
                device_rank=shard.device_rank,
                world_size=shard.world_size,
                start_layer=shard.start_layer,
                end_layer=shard.end_layer,
                input_source=(
                    "decrypted_envelope" if is_ingress else "shape_only_dummy"
                ),
                private_input_accessed=is_ingress,
                prompt_token_count=11,
                ciphertext_digest="sha256:" + "a" * 64,
            )
        )
    audit = VerifiableAuditResponse(
        request_id="frozen-placement-request",
        execution_id=CommandId("frozen-placement-command"),
        expected_ranks=2,
        receipts=receipts,
    )

    assert _receipts_match_frozen_placement(audit, environment) is True

    tampered_receipts = [
        receipt.model_copy(update={"placement_digest": "sha256:" + "0" * 64})
        if receipt.device_rank == 1
        else receipt
        for receipt in receipts
    ]
    assert (
        _receipts_match_frozen_placement(
            audit.model_copy(update={"receipts": tampered_receipts}),
            environment,
        )
        is False
    )


@pytest.mark.parametrize("leaked_value", ["full_plaintext", "unique_nonce"])
def test_clean_audit_detects_canary_fragments_in_public_event(
    leaked_value: str,
) -> None:
    """Leak detection covers decoded multiline text and its unique nonce."""
    nonce = "7f8f09b3-6e7c-4eb4-90af-c921fb35ecad"
    canary = f"private base prompt\n[matrix-canary:matrix-run-001:{nonce}]"
    request_id = "matrix-clean-audit-leaked-request"
    command_id = CommandId("clean-audit-leaked-command")
    environment = _matrix_environment()
    config = _matrix_config_without_private_files()
    cell = next(
        cell
        for cell in build_matrix_cells()
        if cell.plane == "clean_audit" and cell.prompt_class == "ascii_short"
    )
    loaded_prompt = _loaded_prompt(
        "ascii_short",
        canary,
        leak_sentinels=(nonce,),
    )
    encrypted_requests: list[VerifiableChatCompletionRequest] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if (
            request.method == "GET"
            and request.url.path == f"/v1/verifiable/audit/{request_id}"
            and not encrypted_requests
        ):
            return httpx.Response(404)
        if (
            request.method == "POST"
            and request.url.path == "/v1/verifiable/chat/completions"
        ):
            encrypted_requests.append(
                VerifiableChatCompletionRequest.model_validate_json(request.content)
            )
            return httpx.Response(
                200,
                json={
                    "id": str(command_id),
                    "choices": [
                        {"message": {"role": "assistant", "content": "safe output"}}
                    ],
                },
            )
        if (
            request.method == "GET"
            and request.url.path == f"/v1/verifiable/audit/{request_id}"
        ):
            return httpx.Response(
                200,
                json=_audit_for_request(
                    environment,
                    encrypted_requests[-1],
                    command_id,
                    prompt_token_count=11,
                ).model_dump(mode="json", by_alias=True),
            )
        if request.method == "GET" and request.url.path == "/events":
            audit = _audit_for_request(
                environment,
                encrypted_requests[-1],
                command_id,
                prompt_token_count=11,
            )
            events = _clean_audit_events(
                command_id,
                request_id,
                environment.instance.instance_id,
            )
            events.extend(_raw_receipt_events(audit))
            events.append(
                {
                    "TestEvent": {
                        "event_id": "leak-event",
                        "value": canary if leaked_value == "full_plaintext" else nonce,
                    }
                }
            )
            return httpx.Response(200, json=events)
        return httpx.Response(404)

    with httpx.Client(
        transport=httpx.MockTransport(handle),
        base_url=config.base_url,
    ) as client:
        result = execute_matrix_cell(
            client,
            config,
            environment,
            cell,
            loaded_prompt,
            request_id,
        )

    assert result.status == "scientific_fail"
    assert result.failure_class == "privacy_violation"
    assert isinstance(result.evidence, CleanAuditCellEvidence)
    assert result.evidence.canary_absent_from_public_events is False


@pytest.mark.parametrize(
    ("conflicting_event_id", "expected_error"),
    [
        ("raw-conflict-rank-0", "raw_receipt_duplicate_rank"),
        ("raw-receipt-rank-0", "raw_receipt_event_id_conflict"),
    ],
)
def test_clean_audit_rejects_conflicting_raw_receipt_hidden_by_aggregate(
    conflicting_event_id: str,
    expected_error: str,
) -> None:
    """A state projection must not hide a second receipt for the same rank."""
    request_id = "matrix-clean-audit-conflicting-raw-receipt"
    command_id = CommandId("clean-audit-conflicting-raw-command")
    environment = _matrix_environment()
    config = _matrix_config_without_private_files()
    cell = next(
        cell
        for cell in build_matrix_cells()
        if cell.plane == "clean_audit" and cell.prompt_class == "ascii_short"
    )
    loaded_prompt = _loaded_prompt("ascii_short", "private clean audit canary")
    encrypted_requests: list[VerifiableChatCompletionRequest] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if (
            request.method == "GET"
            and request.url.path == f"/v1/verifiable/audit/{request_id}"
            and not encrypted_requests
        ):
            return httpx.Response(404)
        if (
            request.method == "POST"
            and request.url.path == "/v1/verifiable/chat/completions"
        ):
            encrypted_requests.append(
                VerifiableChatCompletionRequest.model_validate_json(request.content)
            )
            return httpx.Response(
                200,
                json={
                    "id": str(command_id),
                    "choices": [
                        {"message": {"role": "assistant", "content": "safe output"}}
                    ],
                },
            )
        if (
            request.method == "GET"
            and request.url.path == f"/v1/verifiable/audit/{request_id}"
        ):
            audit = _audit_for_request(
                environment,
                encrypted_requests[-1],
                command_id,
                prompt_token_count=11,
            )
            return httpx.Response(
                200,
                json=audit.model_dump(mode="json", by_alias=True),
            )
        if request.method == "GET" and request.url.path == "/events":
            audit = _audit_for_request(
                environment,
                encrypted_requests[-1],
                command_id,
                prompt_token_count=11,
            )
            events = _clean_audit_events(
                command_id,
                request_id,
                environment.instance.instance_id,
            )
            events.extend(_raw_receipt_events(audit))
            conflicting_receipt = audit.receipts[0].model_copy(
                update={"prompt_token_count": 12}
            )
            events.append(
                {
                    "VerifiableInputPrepared": {
                        "event_id": conflicting_event_id,
                        "receipt": conflicting_receipt.model_dump(
                            mode="json", by_alias=True
                        ),
                    }
                }
            )
            return httpx.Response(200, json=events)
        return httpx.Response(404)

    with httpx.Client(
        transport=httpx.MockTransport(handle),
        base_url=config.base_url,
    ) as client:
        result = execute_matrix_cell(
            client,
            config,
            environment,
            cell,
            loaded_prompt,
            request_id,
        )

    assert result.status == "scientific_fail"
    assert result.failure_class == "audit_binding_invalid"
    assert isinstance(result.evidence, CleanAuditCellEvidence)
    assert expected_error in result.evidence.binding_errors


def test_execute_matrix_cell_non_ingress_rejection_creates_no_task_or_audit() -> None:
    request_id = "matrix-negative-non-ingress"
    environment = _matrix_environment()
    config = _matrix_config_without_private_files()
    cell = next(
        cell
        for cell in build_matrix_cells()
        if cell.negative_control == "non_ingress_submission"
    )
    encrypted_requests: list[VerifiableChatCompletionRequest] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if (
            request.method == "GET"
            and request.url.path == f"/v1/verifiable/audit/{request_id}"
        ):
            return httpx.Response(404)
        if (
            request.method == "POST"
            and request.url.path == "/v1/verifiable/chat/completions"
        ):
            assert request.url.host == "rtx.test"
            assert b"negative-control" not in request.content
            encrypted_requests.append(
                VerifiableChatCompletionRequest.model_validate_json(request.content)
            )
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": (
                            "Verifiable requests must be submitted to the placement "
                            "ingress node API"
                        ),
                        "type": "Bad Request",
                        "param": None,
                        "code": 400,
                    }
                },
            )
        if request.method == "GET" and request.url.path == "/events":
            return httpx.Response(200, json=[])
        return httpx.Response(404)

    with httpx.Client(
        transport=httpx.MockTransport(handle),
        base_url=config.base_url,
    ) as client:
        result = execute_matrix_cell(
            client,
            config,
            environment,
            cell,
            None,
            request_id,
        )

    assert result.status == "expected_reject"
    assert result.failure_class is None
    assert isinstance(result.evidence, NegativeCellEvidence)
    assert result.evidence.http_status == 400
    assert result.evidence.rejected is True
    assert result.evidence.protocol_rejection_valid is True
    assert result.evidence.task_created is False
    assert result.evidence.receipt_event_created is False
    assert result.evidence.audit_created is False
    assert len(encrypted_requests) == 1
    assert encrypted_requests[0].recipient.node_id == environment.ingress_node_id
    assert "negative-control" not in result.model_dump_json()


@pytest.mark.parametrize(
    ("status_code", "message", "error_type", "error_code"),
    [
        (500, "internal failure", "Internal Server Error", 500),
        (404, "Not Found", "Not Found", 404),
        (400, "generic rejection", "Bad Request", 400),
        (
            400,
            "Verifiable requests must be submitted to the placement ingress node API",
            "Bad Request",
            500,
        ),
    ],
)
def test_negative_control_rejects_unexpected_http_errors(
    status_code: int,
    message: str,
    error_type: str,
    error_code: int,
) -> None:
    request_id = "matrix-negative-invalid-http-evidence"
    environment = _matrix_environment()
    config = _matrix_config_without_private_files()
    cell = next(
        cell
        for cell in build_matrix_cells()
        if cell.negative_control == "non_ingress_submission"
    )
    submitted = False

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal submitted
        if (
            request.method == "GET"
            and request.url.path == f"/v1/verifiable/audit/{request_id}"
        ):
            return httpx.Response(404)
        if request.method == "GET" and request.url.path == "/events":
            return httpx.Response(200, json=[])
        if (
            request.method == "POST"
            and request.url.path == "/v1/verifiable/chat/completions"
        ):
            submitted = True
            return httpx.Response(
                status_code,
                json={
                    "error": {
                        "message": message,
                        "type": error_type,
                        "param": None,
                        "code": error_code,
                    }
                },
            )
        return httpx.Response(404)

    with httpx.Client(
        transport=httpx.MockTransport(handle),
        base_url=config.base_url,
    ) as client:
        result = execute_matrix_cell(
            client,
            config,
            environment,
            cell,
            None,
            request_id,
        )

    assert submitted is True
    assert result.status == "unexpected_accept"
    assert result.failure_class == "unexpected_acceptance"
    assert isinstance(result.evidence, NegativeCellEvidence)
    assert result.evidence.protocol_rejection_valid is False


def test_negative_control_observes_delayed_task_or_audit_side_effect() -> None:
    """Absence is observed for a bounded window instead of sampled once."""
    request_id = "matrix-negative-delayed-side-effect"
    command_id = CommandId("negative-delayed-command")
    environment = _matrix_environment()
    config = _matrix_config_without_private_files().model_copy(
        update={
            "event_timeout_seconds": 0.05,
            "audit_timeout_seconds": 0.05,
            "poll_interval_seconds": 0.001,
        }
    )
    cell = next(
        cell
        for cell in build_matrix_cells()
        if cell.negative_control == "non_ingress_submission"
    )
    submitted = False
    events_reads = 0
    audit_reads_after_submission = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal submitted, events_reads, audit_reads_after_submission
        if (
            request.method == "POST"
            and request.url.path == "/v1/verifiable/chat/completions"
        ):
            submitted = True
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": (
                            "Verifiable requests must be submitted to the placement "
                            "ingress node API"
                        ),
                        "type": "Bad Request",
                        "param": None,
                        "code": 400,
                    }
                },
            )
        if (
            request.method == "GET"
            and request.url.path == f"/v1/verifiable/audit/{request_id}"
        ):
            if submitted:
                audit_reads_after_submission += 1
            return httpx.Response(
                200 if audit_reads_after_submission >= 3 else 404,
                json={} if audit_reads_after_submission >= 3 else None,
            )
        if request.method == "GET" and request.url.path == "/events":
            events_reads += 1
            if events_reads < 3:
                return httpx.Response(200, json=[])
            return httpx.Response(
                200,
                json=_clean_audit_events(
                    command_id,
                    request_id,
                    environment.instance.instance_id,
                ),
            )
        return httpx.Response(404)

    with httpx.Client(
        transport=httpx.MockTransport(handle),
        base_url=config.base_url,
    ) as client:
        result = execute_matrix_cell(
            client,
            config,
            environment,
            cell,
            None,
            request_id,
        )

    assert events_reads >= 3
    assert audit_reads_after_submission >= 3
    assert result.status == "unexpected_accept"
    assert isinstance(result.evidence, NegativeCellEvidence)
    assert result.evidence.task_created is True
    assert result.evidence.audit_created is True


def test_cross_epoch_operation_resumes_existing_epoch_and_restores_primary(
    tmp_path: Path,
) -> None:
    """A resumed N4 run must reuse epoch 2 and leave exact epoch 1 ready."""
    environment = _matrix_environment()
    primary_instance = environment.instance
    assert isinstance(primary_instance, MlxRingInstance)
    replay_instance = primary_instance.model_copy(
        update={
            "instance_id": InstanceId("quality-instance-replay"),
            "ephemeral_port": 50001,
        }
    )
    replay_preview = PlacementPreview(
        model_id=replay_instance.shard_assignments.model_id,
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        instance=replay_instance,
    )
    replay_preview_path = tmp_path / "replay-preview.json"
    replay_preview_path.write_text(replay_preview.model_dump_json(), encoding="utf-8")
    config = _matrix_config_without_private_files().model_copy(
        update={
            "replay_placement_preview_file": str(replay_preview_path),
            "allow_placement_epoch_transition": True,
        }
    )
    live_instances = {replay_instance.instance_id: replay_instance}
    posted_instance_ids: list[InstanceId] = []
    deleted_instance_ids: list[InstanceId] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/state":
            state = State(
                instances=live_instances,
                runners={
                    runner_id: RunnerReady()
                    for runner_id in primary_instance.shard_assignments.runner_to_shard
                },
                node_backends={
                    NodeId("node-mini"): [Backend.MlxMetal],
                    NodeId("node-rtx"): [Backend.MlxCuda],
                },
            )
            return httpx.Response(200, json=state.model_dump(mode="json"))
        if request.method == "DELETE" and request.url.path.startswith("/instance/"):
            instance_id = InstanceId(request.url.path.rsplit("/", 1)[1])
            deleted_instance_ids.append(instance_id)
            live_instances.pop(instance_id, None)
            return httpx.Response(200, json={"message": "deleted"})
        if request.method == "POST" and request.url.path == "/instance":
            body = JSON_OBJECT_ADAPTER.validate_json(request.content)
            created = MlxRingInstance.model_validate(body["instance"])
            posted_instance_ids.append(created.instance_id)
            live_instances[created.instance_id] = created
            return httpx.Response(200, json={"message": "accepted"})
        node_id = {
            "mini.test": NodeId("node-mini"),
            "rtx.test": NodeId("node-rtx"),
        }.get(request.url.host)
        if node_id is not None and request.url.path == "/node_id":
            return httpx.Response(200, json=str(node_id))
        if node_id is not None and request.url.path == "/v1/verifiable/identity":
            return httpx.Response(
                200,
                json=environment.identities[node_id].model_dump(mode="json"),
            )
        return httpx.Response(404)

    operation_calls = 0

    def operation() -> str:
        nonlocal operation_calls
        operation_calls += 1
        assert live_instances == {replay_instance.instance_id: replay_instance}
        return "operation-result"

    with httpx.Client(
        transport=httpx.MockTransport(handle), base_url=config.base_url
    ) as client:
        result = quality_matrix.run_cross_epoch_operation_with_primary_restore(
            client,
            config,
            environment,
            operation,
        )

    assert result == "operation-result"
    assert operation_calls == 1
    assert posted_instance_ids == [primary_instance.instance_id]
    assert deleted_instance_ids == [replay_instance.instance_id]
    assert live_instances == {primary_instance.instance_id: primary_instance}


@pytest.mark.parametrize("restore_failure", [False, True])
def test_cross_epoch_operation_restores_primary_without_swallowing_failure(
    tmp_path: Path,
    restore_failure: bool,
) -> None:
    """The operation's original exception survives a successful restoration."""
    environment = _matrix_environment()
    primary_instance = environment.instance
    assert isinstance(primary_instance, MlxRingInstance)
    replay_instance = primary_instance.model_copy(
        update={
            "instance_id": InstanceId("quality-instance-replay"),
            "ephemeral_port": 50001,
        }
    )
    replay_preview = PlacementPreview(
        model_id=replay_instance.shard_assignments.model_id,
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        instance=replay_instance,
    )
    replay_preview_path = tmp_path / "replay-preview.json"
    replay_preview_path.write_text(replay_preview.model_dump_json(), encoding="utf-8")
    config = _matrix_config_without_private_files().model_copy(
        update={
            "replay_placement_preview_file": str(replay_preview_path),
            "allow_placement_epoch_transition": True,
        }
    )
    live_instances = {primary_instance.instance_id: primary_instance}
    posted_instance_ids: list[InstanceId] = []
    deleted_instance_ids: list[InstanceId] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/state":
            state = State(
                instances=live_instances,
                runners={
                    runner_id: RunnerReady()
                    for runner_id in primary_instance.shard_assignments.runner_to_shard
                },
                node_backends={
                    NodeId("node-mini"): [Backend.MlxMetal],
                    NodeId("node-rtx"): [Backend.MlxCuda],
                },
            )
            return httpx.Response(200, json=state.model_dump(mode="json"))
        if request.method == "DELETE" and request.url.path.startswith("/instance/"):
            instance_id = InstanceId(request.url.path.rsplit("/", 1)[1])
            deleted_instance_ids.append(instance_id)
            live_instances.pop(instance_id, None)
            return httpx.Response(200, json={"message": "deleted"})
        if request.method == "POST" and request.url.path == "/instance":
            body = JSON_OBJECT_ADAPTER.validate_json(request.content)
            created = MlxRingInstance.model_validate(body["instance"])
            posted_instance_ids.append(created.instance_id)
            if restore_failure and created.instance_id == primary_instance.instance_id:
                return httpx.Response(503, json={"error": "restore failed"})
            live_instances[created.instance_id] = created
            return httpx.Response(200, json={"message": "accepted"})
        node_id = {
            "mini.test": NodeId("node-mini"),
            "rtx.test": NodeId("node-rtx"),
        }.get(request.url.host)
        if node_id is not None and request.url.path == "/node_id":
            return httpx.Response(200, json=str(node_id))
        if node_id is not None and request.url.path == "/v1/verifiable/identity":
            return httpx.Response(
                200,
                json=environment.identities[node_id].model_dump(mode="json"),
            )
        return httpx.Response(404)

    class OperationMarkerError(RuntimeError):
        pass

    marker = OperationMarkerError("operation failed")

    def operation() -> None:
        assert live_instances == {replay_instance.instance_id: replay_instance}
        raise marker

    with httpx.Client(
        transport=httpx.MockTransport(handle), base_url=config.base_url
    ) as client:
        if restore_failure:
            with pytest.raises(ExceptionGroup) as raised_group:
                quality_matrix.run_cross_epoch_operation_with_primary_restore(
                    client,
                    config,
                    environment,
                    operation,
                )
            assert marker in raised_group.value.exceptions
            assert any(
                isinstance(error, httpx.HTTPStatusError)
                for error in raised_group.value.exceptions
            )
        else:
            with pytest.raises(OperationMarkerError) as raised:
                quality_matrix.run_cross_epoch_operation_with_primary_restore(
                    client,
                    config,
                    environment,
                    operation,
                )
            assert raised.value is marker

    assert posted_instance_ids == [
        replay_instance.instance_id,
        primary_instance.instance_id,
    ]
    assert deleted_instance_ids == [
        primary_instance.instance_id,
        replay_instance.instance_id,
    ]
    if restore_failure:
        assert live_instances == {}
    else:
        assert live_instances == {primary_instance.instance_id: primary_instance}


@pytest.mark.parametrize("restore_failure", [False, True])
def test_cross_epoch_negative_control_observes_rejection_before_restoring_primary(
    tmp_path: Path,
    restore_failure: bool,
) -> None:
    """N4's POST, protocol check, and absence observation all run in epoch 2."""
    request_id = "matrix-negative-cross-epoch"
    environment = _matrix_environment()
    primary_instance = environment.instance
    assert isinstance(primary_instance, MlxRingInstance)
    replay_instance = primary_instance.model_copy(
        update={
            "instance_id": InstanceId("quality-instance-replay"),
            "ephemeral_port": 50001,
        }
    )
    replay_preview = PlacementPreview(
        model_id=replay_instance.shard_assignments.model_id,
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        instance=replay_instance,
    )
    replay_preview_path = tmp_path / "replay-preview.json"
    replay_preview_path.write_text(replay_preview.model_dump_json(), encoding="utf-8")
    config = _matrix_config_without_private_files().model_copy(
        update={
            "replay_placement_preview_file": str(replay_preview_path),
            "allow_placement_epoch_transition": True,
        }
    )
    cell = next(
        cell
        for cell in build_matrix_cells()
        if cell.negative_control == "cross_epoch_replay"
    )
    live_instances = {primary_instance.instance_id: primary_instance}
    posted_instance_ids: list[InstanceId] = []
    deleted_instance_ids: list[InstanceId] = []
    stale_envelope_submitted = False
    observed_after_submission = False

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal stale_envelope_submitted, observed_after_submission
        if request.method == "GET" and request.url.path == "/state":
            state = State(
                instances=live_instances,
                runners={
                    runner_id: RunnerReady()
                    for runner_id in primary_instance.shard_assignments.runner_to_shard
                },
                node_backends={
                    NodeId("node-mini"): [Backend.MlxMetal],
                    NodeId("node-rtx"): [Backend.MlxCuda],
                },
            )
            return httpx.Response(200, json=state.model_dump(mode="json"))
        if request.method == "DELETE" and request.url.path.startswith("/instance/"):
            instance_id = InstanceId(request.url.path.rsplit("/", 1)[1])
            deleted_instance_ids.append(instance_id)
            live_instances.pop(instance_id, None)
            return httpx.Response(200, json={"message": "deleted"})
        if request.method == "POST" and request.url.path == "/instance":
            body = JSON_OBJECT_ADAPTER.validate_json(request.content)
            created = MlxRingInstance.model_validate(body["instance"])
            posted_instance_ids.append(created.instance_id)
            if restore_failure and created.instance_id == primary_instance.instance_id:
                return httpx.Response(503, json={"error": "restore failed"})
            live_instances[created.instance_id] = created
            return httpx.Response(200, json={"message": "accepted"})
        if (
            request.method == "POST"
            and request.url.path == "/v1/verifiable/chat/completions"
        ):
            assert request.url.host == "mini.test"
            assert live_instances == {replay_instance.instance_id: replay_instance}
            stale_envelope_submitted = True
            return httpx.Response(
                404,
                json={
                    "error": {
                        "message": (
                            f"Instance {primary_instance.instance_id} was not found"
                        ),
                        "type": "Not Found",
                        "param": None,
                        "code": 404,
                    }
                },
            )
        if request.method == "GET" and request.url.path == "/events":
            assert stale_envelope_submitted is True
            assert live_instances == {replay_instance.instance_id: replay_instance}
            observed_after_submission = True
            return httpx.Response(200, json=[])
        if (
            request.method == "GET"
            and request.url.path == f"/v1/verifiable/audit/{request_id}"
        ):
            if stale_envelope_submitted:
                assert live_instances == {replay_instance.instance_id: replay_instance}
                observed_after_submission = True
            return httpx.Response(404)
        node_id = {
            "mini.test": NodeId("node-mini"),
            "rtx.test": NodeId("node-rtx"),
        }.get(request.url.host)
        if node_id is not None and request.url.path == "/node_id":
            return httpx.Response(200, json=str(node_id))
        if node_id is not None and request.url.path == "/v1/verifiable/identity":
            return httpx.Response(
                200,
                json=environment.identities[node_id].model_dump(mode="json"),
            )
        return httpx.Response(404)

    result: quality_matrix.CellExecutionResult | None = None
    with httpx.Client(
        transport=httpx.MockTransport(handle), base_url=config.base_url
    ) as client:
        if restore_failure:
            with pytest.raises(httpx.HTTPStatusError):
                execute_matrix_cell(
                    client,
                    config,
                    environment,
                    cell,
                    None,
                    request_id,
                )
        else:
            result = execute_matrix_cell(
                client,
                config,
                environment,
                cell,
                None,
                request_id,
            )

    assert stale_envelope_submitted is True
    assert observed_after_submission is True
    assert posted_instance_ids == [
        replay_instance.instance_id,
        primary_instance.instance_id,
    ]
    assert deleted_instance_ids == [
        primary_instance.instance_id,
        replay_instance.instance_id,
    ]
    if restore_failure:
        assert result is None
        assert live_instances == {}
        return

    assert result is not None
    assert result.status == "expected_reject"
    assert result.failure_class is None
    assert isinstance(result.evidence, NegativeCellEvidence)
    assert result.evidence.control == "cross_epoch_replay"
    assert result.evidence.protocol_rejection_valid is True
    assert result.evidence.task_created is False
    assert result.evidence.receipt_event_created is False
    assert result.evidence.audit_created is False
    assert live_instances == {primary_instance.instance_id: primary_instance}


def test_token_preflight_uses_bound_qwen_fallback_without_jinja(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer_directory = tmp_path / "tokenizer"
    tokenizer_directory.mkdir()
    (tokenizer_directory / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "chat_template": (
                    "<|im_start|>{{ message['role'] }}<|im_end|>"
                    "{% if add_generation_prompt %}<|im_start|>assistant{% endif %}"
                )
            }
        ),
        encoding="utf-8",
    )
    config = _config_with_prompts(
        tmp_path, output_directory=tmp_path / "results"
    ).model_copy(update={"tokenizer_directory": str(tokenizer_directory)})
    loaded = load_prompt_set(config)
    rendered_prompts: list[str] = []

    class FakeTokenizer:
        def apply_chat_template(self, *args: object, **kwargs: object) -> object:
            del args, kwargs
            raise ImportError("jinja is intentionally absent")

        def encode(self, text: str, *, add_special_tokens: bool) -> object:
            assert add_special_tokens is False
            assert text.startswith("<|im_start|>user\n")
            assert text.endswith("<|im_end|>\n<|im_start|>assistant\n")
            rendered_prompts.append(text)
            for prompt_class, target in quality_matrix.PROMPT_TOKEN_TARGETS.items():
                if prompt_class in text:
                    return list(range(target))
            raise AssertionError("prompt class marker is missing")

    class FakeFactory:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> FakeTokenizer:
            del args, kwargs
            return FakeTokenizer()

    class FakeTransformers:
        AutoTokenizer = FakeFactory

    def fake_import_module(name: str) -> object:
        assert name == "transformers"
        return FakeTransformers

    monkeypatch.setattr(
        quality_matrix.importlib,
        "import_module",
        fake_import_module,
    )

    validated, tokenizer_evidence = quality_matrix.validate_prompt_token_counts(
        config, loaded
    )

    assert len(rendered_prompts) == 5
    assert validated["prefill_4096"].evidence.actual_prompt_tokens == 4096
    assert validated["prefill_4608"].evidence.actual_prompt_tokens == 4608
    assert tokenizer_evidence.file_count == 1


def test_token_preflight_extracts_input_ids_from_batch_encoding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer_directory = tmp_path / "tokenizer"
    tokenizer_directory.mkdir()
    (tokenizer_directory / "tokenizer.json").write_text("{}", encoding="utf-8")
    config = _config_with_prompts(
        tmp_path, output_directory=tmp_path / "results"
    ).model_copy(update={"tokenizer_directory": str(tokenizer_directory)})
    loaded = load_prompt_set(config)

    class FakeTokenizer:
        def apply_chat_template(
            self,
            conversation: list[dict[str, str]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
        ) -> object:
            assert tokenize is True
            assert add_generation_prompt is True
            plaintext = conversation[0]["content"]
            prompt_class = cast(
                PromptClass,
                next(
                    candidate for candidate in PROMPT_CLASSES if candidate in plaintext
                ),
            )
            target = quality_matrix.PROMPT_TOKEN_TARGETS[prompt_class]
            return {
                "input_ids": list(range(target)),
                "attention_mask": [1] * target,
            }

        def encode(self, text: str, *, add_special_tokens: bool) -> object:
            del text, add_special_tokens
            raise AssertionError("fallback tokenizer must not be used")

    def load_tokenizer(directory: Path) -> FakeTokenizer:
        del directory
        return FakeTokenizer()

    monkeypatch.setattr(quality_matrix, "_load_chat_tokenizer", load_tokenizer)

    validated, _ = quality_matrix.validate_prompt_token_counts(config, loaded)

    assert {
        prompt_class: prompt.evidence.actual_prompt_tokens
        for prompt_class, prompt in validated.items()
    } == quality_matrix.PROMPT_TOKEN_TARGETS


@pytest.mark.parametrize(
    "prompt_class",
    ["ascii_short", "zh_unicode", "code_json"],
)
def test_token_preflight_rejects_short_prompt_outside_locked_length(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt_class: PromptClass,
) -> None:
    tokenizer_directory = tmp_path / "tokenizer"
    tokenizer_directory.mkdir()
    (tokenizer_directory / "tokenizer.json").write_text("{}", encoding="utf-8")
    config = _config_with_prompts(
        tmp_path, output_directory=tmp_path / "results"
    ).model_copy(update={"tokenizer_directory": str(tokenizer_directory)})
    loaded = load_prompt_set(config)

    class FakeTokenizer:
        def apply_chat_template(
            self,
            conversation: list[dict[str, str]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
        ) -> object:
            assert tokenize is True
            assert add_generation_prompt is True
            plaintext = conversation[0]["content"]
            matched_class = cast(
                PromptClass,
                next(
                    candidate for candidate in PROMPT_CLASSES if candidate in plaintext
                ),
            )
            target = quality_matrix.PROMPT_TOKEN_TARGETS[matched_class]
            if matched_class == prompt_class:
                target += 1
            return list(range(target))

        def encode(self, text: str, *, add_special_tokens: bool) -> object:
            del text, add_special_tokens
            raise AssertionError("fallback tokenizer must not be used")

    def load_tokenizer(directory: Path) -> FakeTokenizer:
        del directory
        return FakeTokenizer()

    monkeypatch.setattr(quality_matrix, "_load_chat_tokenizer", load_tokenizer)

    with pytest.raises(ValueError, match="rendered to"):
        quality_matrix.validate_prompt_token_counts(config, loaded)


def test_run_quality_matrix_writes_complete_redacted_manifest_and_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public runner emits the locked matrix without copying private inputs."""
    output_directory = tmp_path / "results"
    config = _config_with_prompts(tmp_path, output_directory=output_directory)
    environment = _matrix_environment()
    _stub_environment_inspection(monkeypatch, environment)
    _stub_tokenizer_validation(monkeypatch)
    executed: list[str] = []
    private_prompts_by_plane: dict[str, list[str]] = {
        "quality": [],
        "clean_audit": [],
    }

    def execute_cell(
        client: httpx.Client,
        received: QualityMatrixConfig,
        received_environment: MatrixEnvironment,
        cell: MatrixCell,
        prompt: LoadedPrompt | None,
        request_id: str,
    ):
        del client, received, request_id
        assert received_environment is environment
        executed.append(cell.cell_id)
        if prompt is not None:
            private_prompts_by_plane[cell.plane].append(prompt.plaintext)
        return quality_matrix.CellExecutionResult(
            status="expected_reject" if cell.plane == "negative" else "pass"
        )

    with _no_network_client(config) as client:
        returned_summary = quality_matrix.run_quality_matrix(
            client,
            config,
            execute_cell=execute_cell,
        )

    manifest_path = output_directory / "manifest.json"
    attempts_path = output_directory / "attempts.jsonl"
    summary_path = output_directory / "summary.json"
    manifest = JSON_OBJECT_ADAPTER.validate_json(manifest_path.read_bytes())
    summary = JSON_OBJECT_ADAPTER.validate_json(summary_path.read_bytes())
    artifacts = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (manifest_path, attempts_path, summary_path)
    )

    assert len(executed) == 49
    assert set(executed) == {cell.cell_id for cell in build_matrix_cells()}
    manifest_cells = cast(list[dict[str, JsonValue]], manifest["cells"])
    status_counts = cast(dict[str, JsonValue], summary["status_counts"])
    assert len(manifest_cells) == 49
    assert {cast(str, cell["cell_id"]) for cell in manifest_cells} == set(executed)
    assert summary["total_cells"] == 49
    assert summary["terminal_cells"] == 49
    assert status_counts["expected_reject"] == 4
    assert status_counts["pass"] == 45
    assert summary["passed"] is True
    assert manifest["formal_execution"] is False
    assert summary["formal_execution"] is False
    assert summary["formal_passed"] is False
    assert returned_summary.model_dump(mode="json") == summary
    quality_plaintexts = set(private_prompts_by_plane["quality"])
    clean_plaintexts = private_prompts_by_plane["clean_audit"]
    assert len(clean_plaintexts) == 15
    assert len(set(clean_plaintexts)) == 15
    assert quality_plaintexts.isdisjoint(clean_plaintexts)

    for prompt_class in PROMPT_CLASSES:
        private_text = f"private prompt for {prompt_class}"
        assert private_text not in artifacts
    assert str(tmp_path) not in artifacts
    assert '"plaintext"' not in artifacts
    assert '"path"' not in artifacts


def test_run_quality_matrix_resume_retries_only_infrastructure_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resume preserves the failed attempt and allocates a fresh request id."""
    output_directory = tmp_path / "results"
    config = _config_with_prompts(tmp_path, output_directory=output_directory)
    environment = _matrix_environment()
    _stub_environment_inspection(monkeypatch, environment)
    _stub_tokenizer_validation(monkeypatch)
    retry_cell_id = build_matrix_cells()[0].cell_id
    cross_epoch_cell_id = next(
        cell.cell_id
        for cell in build_matrix_cells()
        if cell.negative_control == "cross_epoch_replay"
    )
    first_run_cells: list[str] = []

    def first_execute(
        client: httpx.Client,
        received: QualityMatrixConfig,
        received_environment: MatrixEnvironment,
        cell: MatrixCell,
        prompt: LoadedPrompt | None,
        request_id: str,
    ):
        del client, received, received_environment, prompt, request_id
        first_run_cells.append(cell.cell_id)
        if cell.cell_id == retry_cell_id:
            return quality_matrix.CellExecutionResult(
                status="infra_fail",
                failure_class="transport_timeout",
            )
        return quality_matrix.CellExecutionResult(
            status="expected_reject" if cell.plane == "negative" else "pass"
        )

    with _no_network_client(config) as client:
        first_summary = quality_matrix.run_quality_matrix(
            client,
            config,
            execute_cell=first_execute,
        )

    resumed_cells: list[str] = []

    def resumed_execute(
        client: httpx.Client,
        received: QualityMatrixConfig,
        received_environment: MatrixEnvironment,
        cell: MatrixCell,
        prompt: LoadedPrompt | None,
        request_id: str,
    ):
        del client, received, received_environment, prompt, request_id
        resumed_cells.append(cell.cell_id)
        return quality_matrix.CellExecutionResult(
            status="expected_reject" if cell.plane == "negative" else "pass"
        )

    resumed_config = config.model_copy(update={"resume": True})
    with _no_network_client(resumed_config) as client:
        resumed_summary = quality_matrix.run_quality_matrix(
            client,
            resumed_config,
            execute_cell=resumed_execute,
        )

    records = AttemptJournal(output_directory / "attempts.jsonl").records()
    retried_records = [record for record in records if record.cell_id == retry_cell_id]
    starts = [record for record in retried_records if record.event == "started"]
    finishes = [record for record in retried_records if record.event == "finished"]

    assert len(first_run_cells) == 48
    assert cross_epoch_cell_id not in first_run_cells
    assert first_summary.status_counts["infra_fail"] == 1
    assert first_summary.passed is False
    assert resumed_cells == [retry_cell_id, cross_epoch_cell_id]
    assert [record.attempt_index for record in starts] == [0, 1]
    assert starts[0].attempt_id != starts[1].attempt_id
    assert starts[0].request_id != starts[1].request_id
    assert [record.status for record in finishes] == ["infra_fail", "pass"]
    assert resumed_summary.terminal_cells == 49
    assert resumed_summary.status_counts["expected_reject"] == 4
    assert resumed_summary.status_counts["pass"] == 45
    assert resumed_summary.attempt_count == 50
    assert resumed_summary.attempt_status_counts["infra_fail"] == 1
    assert resumed_summary.attempt_status_counts["pass"] == 45
    assert resumed_summary.retried_cells == [retry_cell_id]
    assert resumed_summary.passed is True


def test_run_quality_matrix_never_retries_protocol_evidence_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_directory = tmp_path / "results"
    config = _config_with_prompts(tmp_path, output_directory=output_directory)
    environment = _matrix_environment()
    _stub_environment_inspection(monkeypatch, environment)
    _stub_tokenizer_validation(monkeypatch)
    failed_cell_id = build_matrix_cells()[0].cell_id
    cross_epoch_cell_id = next(
        cell.cell_id
        for cell in build_matrix_cells()
        if cell.negative_control == "cross_epoch_replay"
    )
    first_run_cells: list[str] = []

    def first_execute(
        client: httpx.Client,
        received: QualityMatrixConfig,
        received_environment: MatrixEnvironment,
        cell: MatrixCell,
        prompt: LoadedPrompt | None,
        request_id: str,
    ):
        del client, received, received_environment, prompt, request_id
        first_run_cells.append(cell.cell_id)
        if cell.cell_id == failed_cell_id:
            raise quality_matrix.QualityCheckError("multiple terminal chunks")
        return quality_matrix.CellExecutionResult(
            status="expected_reject" if cell.plane == "negative" else "pass"
        )

    with _no_network_client(config) as client:
        first_summary = quality_matrix.run_quality_matrix(
            client,
            config,
            execute_cell=first_execute,
        )

    resumed_cells: list[str] = []

    def resumed_execute(
        client: httpx.Client,
        received: QualityMatrixConfig,
        received_environment: MatrixEnvironment,
        cell: MatrixCell,
        prompt: LoadedPrompt | None,
        request_id: str,
    ):
        del client, received, received_environment, prompt, request_id
        resumed_cells.append(cell.cell_id)
        return quality_matrix.CellExecutionResult(status="pass")

    with _no_network_client(config.model_copy(update={"resume": True})) as client:
        resumed_summary = quality_matrix.run_quality_matrix(
            client,
            config.model_copy(update={"resume": True}),
            execute_cell=resumed_execute,
        )

    assert first_summary.status_counts["scientific_fail"] == 1
    assert cross_epoch_cell_id not in first_run_cells
    assert first_summary.attempt_status_counts["scientific_fail"] == 1
    assert resumed_cells == []
    assert resumed_summary.status_counts["scientific_fail"] == 1
    assert resumed_summary.passed is False


def test_resume_stops_before_external_work_when_an_attempt_has_unknown_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_directory = tmp_path / "results"
    initial_config = _config_with_prompts(tmp_path, output_directory=output_directory)
    environment = _matrix_environment()
    _stub_environment_inspection(monkeypatch, environment)
    _stub_tokenizer_validation(monkeypatch)

    def execute_cell(
        client: httpx.Client,
        received: QualityMatrixConfig,
        received_environment: MatrixEnvironment,
        cell: MatrixCell,
        prompt: LoadedPrompt | None,
        request_id: str,
    ) -> quality_matrix.CellExecutionResult:
        del client, received, received_environment, prompt, request_id
        if cell.cell_id == build_matrix_cells()[0].cell_id:
            return quality_matrix.CellExecutionResult(
                status="infra_fail",
                failure_class="transport_timeout",
            )
        return quality_matrix.CellExecutionResult(
            status="expected_reject" if cell.plane == "negative" else "pass"
        )

    with _no_network_client(initial_config) as client:
        quality_matrix.run_quality_matrix(
            client,
            initial_config,
            execute_cell=execute_cell,
        )

    config = initial_config.model_copy(update={"resume": True})
    cell = build_matrix_cells()[0]
    journal = AttemptJournal(output_directory / "attempts.jsonl")
    journal.start(config.run_id, cell, None)
    environment_touched = False

    def inspect(
        client: httpx.Client,
        received: QualityMatrixConfig,
    ) -> MatrixEnvironment:
        nonlocal environment_touched
        del client, received
        environment_touched = True
        return _matrix_environment()

    monkeypatch.setattr(quality_matrix, "prepare_matrix_environment", inspect)

    with (
        _no_network_client(config) as client,
        pytest.raises(ValueError, match="unknown outcome"),
    ):
        quality_matrix.run_quality_matrix(
            client,
            config,
            execute_cell=execute_cell,
        )

    assert environment_touched is False


def test_run_quality_matrix_requires_resume_and_an_identical_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing evidence cannot be silently overwritten or mixed across configs."""
    output_directory = tmp_path / "results"
    replay_preview = tmp_path / "replay-preview.json"
    replay_preview.write_text('{"epoch":2}', encoding="utf-8")
    config = _config_with_prompts(
        tmp_path, output_directory=output_directory
    ).model_copy(
        update={
            "replay_placement_preview_file": str(replay_preview),
            "allow_placement_epoch_transition": True,
        }
    )
    environment = _matrix_environment()
    _stub_environment_inspection(monkeypatch, environment)
    _stub_tokenizer_validation(monkeypatch)

    def execute_cell(
        client: httpx.Client,
        received: QualityMatrixConfig,
        received_environment: MatrixEnvironment,
        cell: MatrixCell,
        prompt: LoadedPrompt | None,
        request_id: str,
    ):
        del client, received, received_environment, prompt, request_id
        return quality_matrix.CellExecutionResult(
            status="expected_reject" if cell.plane == "negative" else "pass"
        )

    with _no_network_client(config) as client:
        quality_matrix.run_quality_matrix(client, config, execute_cell=execute_cell)

    with (
        _no_network_client(config) as client,
        pytest.raises(ValueError, match="resume"),
    ):
        quality_matrix.run_quality_matrix(client, config, execute_cell=execute_cell)

    environment_touched = False

    def prepare(
        client: httpx.Client,
        received: QualityMatrixConfig,
    ) -> MatrixEnvironment:
        nonlocal environment_touched
        del client, received
        environment_touched = True
        return environment

    monkeypatch.setattr(quality_matrix, "prepare_matrix_environment", prepare)
    mismatched = config.model_copy(update={"resume": True, "run_id": "different-run"})
    with (
        _no_network_client(mismatched) as client,
        pytest.raises(ValueError, match="manifest"),
    ):
        quality_matrix.run_quality_matrix(client, mismatched, execute_cell=execute_cell)
    assert environment_touched is False

    _stub_environment_inspection(monkeypatch, environment)

    replay_preview.write_text('{"epoch":3}', encoding="utf-8")
    with (
        _no_network_client(config.model_copy(update={"resume": True})) as client,
        pytest.raises(ValueError, match="manifest"),
    ):
        quality_matrix.run_quality_matrix(
            client,
            config.model_copy(update={"resume": True}),
            execute_cell=execute_cell,
        )


def _config_with_prompts(
    tmp_path: Path, *, output_directory: Path
) -> QualityMatrixConfig:
    prompt_files: list[MatrixPromptFile] = []
    for prompt_class in PROMPT_CLASSES:
        path = tmp_path / f"fixture-{prompt_class}.txt"
        path.write_text(f"private prompt for {prompt_class}", encoding="utf-8")
        prompt_files.append(MatrixPromptFile(prompt_class=prompt_class, path=str(path)))
    return QualityMatrixConfig(
        base_url="http://control.test",
        nodes=[
            MatrixNodeEndpoint(label="mini1", api_url="http://mini.test"),
            MatrixNodeEndpoint(label="rtx3090", api_url="http://rtx.test"),
        ],
        instance_id="quality-instance",
        output_directory=str(output_directory),
        run_id="matrix-run-001",
        prompts=prompt_files,
    )


def _two_node_instance(*, rtx_ingress: bool = False) -> MlxRingInstance:
    model = ModelCard(
        model_id=ModelId("mlx-community/Qwen3-0.6B-8bit"),
        storage_size=Memory.from_bytes(698351616),
        n_layers=28,
        hidden_size=1024,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal, Backend.MlxCuda],
    )
    mini_runner = RunnerId("runner-mini")
    rtx_runner = RunnerId("runner-rtx")
    mini_shard = PipelineShardMetadata(
        model_card=model,
        device_rank=1 if rtx_ingress else 0,
        world_size=2,
        start_layer=22 if rtx_ingress else 0,
        end_layer=28 if rtx_ingress else 14,
        n_layers=28,
    )
    rtx_shard = PipelineShardMetadata(
        model_card=model,
        device_rank=0 if rtx_ingress else 1,
        world_size=2,
        start_layer=0 if rtx_ingress else 14,
        end_layer=22 if rtx_ingress else 28,
        n_layers=28,
    )
    return MlxRingInstance(
        instance_id=InstanceId("quality-instance"),
        shard_assignments=ShardAssignments(
            model_id=model.model_id,
            runner_to_shard={
                mini_runner: mini_shard,
                rtx_runner: rtx_shard,
            },
            node_to_runner={
                NodeId("node-mini"): mini_runner,
                NodeId("node-rtx"): rtx_runner,
            },
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )


def _identity(node_id: NodeId):
    public_key = delivery_public_key(generate_delivery_private_key())
    from exo.shared.types.verifiable import VerifiableProviderIdentity

    return VerifiableProviderIdentity(
        node_id=node_id,
        provider_id=provider_id_from_public_key(public_key),
        key_id=DELIVERY_KEY_ID,
        public_key=public_key,
    )


def _matrix_environment() -> MatrixEnvironment:
    instance = _two_node_instance()
    ingress_node = NodeId("node-mini")
    downstream_node = NodeId("node-rtx")
    return MatrixEnvironment(
        instance=instance,
        world_size=2,
        ingress_node_id=ingress_node,
        ingress_url="http://mini.test",
        downstream_node_id=downstream_node,
        downstream_url="http://rtx.test",
        identities={
            ingress_node: _identity(ingress_node),
            downstream_node: _identity(downstream_node),
        },
    )


def _matrix_config_without_private_files() -> QualityMatrixConfig:
    return QualityMatrixConfig(
        base_url="http://control.test",
        nodes=[
            MatrixNodeEndpoint(label="mini1", api_url="http://mini.test"),
            MatrixNodeEndpoint(label="rtx3090", api_url="http://rtx.test"),
        ],
        instance_id="quality-instance",
        output_directory="unused-results",
        run_id="matrix-public-seam-test",
        prompts=[
            MatrixPromptFile(prompt_class=prompt_class, path=f"unused-{prompt_class}")
            for prompt_class in PROMPT_CLASSES
        ],
        audit_timeout_seconds=0.0,
        event_timeout_seconds=0.0,
        poll_interval_seconds=0.001,
    )


def _loaded_prompt(
    prompt_class: PromptClass,
    plaintext: str,
    *,
    leak_sentinels: tuple[str, ...] = (),
) -> LoadedPrompt:
    encoded = plaintext.encode("utf-8")
    return LoadedPrompt(
        plaintext=plaintext,
        evidence=PromptEvidence(
            prompt_class=prompt_class,
            sha256="sha256:" + hashlib.sha256(encoded).hexdigest(),
            utf8_bytes=len(encoded),
            actual_prompt_tokens=11,
        ),
        leak_sentinels=leak_sentinels,
    )


def _audit_for_request(
    environment: MatrixEnvironment,
    request: VerifiableChatCompletionRequest,
    command_id: CommandId,
    *,
    prompt_token_count: int,
) -> VerifiableAuditResponse:
    ciphertext_digest = (
        "sha256:"
        + hashlib.sha256(request.encrypted_input.ciphertext.encode("utf-8")).hexdigest()
    )
    receipts: list[VerifiableInputReceipt] = []
    for (
        node_id,
        runner_id,
    ) in environment.instance.shard_assignments.node_to_runner.items():
        shard = environment.instance.shard_assignments.runner_to_shard[runner_id]
        assert isinstance(shard, PipelineShardMetadata)
        identity = environment.identities[node_id]
        is_ingress = node_id == environment.ingress_node_id
        receipts.append(
            VerifiableInputReceipt(
                request_id=request.request_id,
                execution_id=command_id,
                instance_id=str(environment.instance.instance_id),
                placement_digest=request.placement_digest,
                node_id=node_id,
                reporting_provider_id=identity.provider_id,
                reporting_key_id=identity.key_id,
                recipient_provider_id=request.recipient.provider_id,
                recipient_key_id=request.recipient.key_id,
                device_rank=shard.device_rank,
                world_size=shard.world_size,
                start_layer=shard.start_layer,
                end_layer=shard.end_layer,
                input_source=(
                    "decrypted_envelope" if is_ingress else "shape_only_dummy"
                ),
                private_input_accessed=is_ingress,
                prompt_token_count=prompt_token_count,
                ciphertext_digest=ciphertext_digest,
            )
        )
    return VerifiableAuditResponse(
        request_id=request.request_id,
        execution_id=command_id,
        expected_ranks=environment.world_size,
        receipts=receipts,
    )


def _clean_audit_events(
    command_id: CommandId,
    request_id: str,
    instance_id: InstanceId,
    *,
    complete: bool = True,
) -> list[dict[str, object]]:
    task_id = f"task-{command_id}"
    events: list[dict[str, object]] = [
        {
            "TaskCreated": {
                "task_id": task_id,
                "task": {
                    "TextGeneration": {
                        "task_id": task_id,
                        "command_id": str(command_id),
                        "instance_id": str(instance_id),
                        "task_params": {
                            "input": [],
                            "verifiable": {"request_id": request_id},
                        },
                    }
                },
            }
        },
        {
            "ChunkGenerated": {
                "command_id": str(command_id),
                "chunk": {
                    "TokenChunk": {
                        "token_id": 123,
                        "finish_reason": "stop" if complete else None,
                    }
                },
            }
        },
    ]
    if complete:
        events.append(
            {
                "TaskStatusUpdated": {
                    "task_id": task_id,
                    "task_status": "Complete",
                }
            }
        )
    return events


def _raw_receipt_events(
    audit: VerifiableAuditResponse,
) -> list[dict[str, object]]:
    return [
        {
            "VerifiableInputPrepared": {
                "event_id": f"raw-receipt-rank-{receipt.device_rank}",
                "receipt": receipt.model_dump(mode="json", by_alias=True),
            }
        }
        for receipt in audit.receipts
    ]


def _stub_environment_inspection(
    monkeypatch: pytest.MonkeyPatch,
    environment: MatrixEnvironment,
) -> None:
    def inspect(
        client: httpx.Client,
        config: QualityMatrixConfig,
    ) -> MatrixEnvironment:
        del client, config
        return environment

    def wait_for_instance(
        client: httpx.Client,
        expected_instance: object,
        *,
        timeout_seconds: float,
        poll_interval_seconds: float,
    ) -> None:
        del client, expected_instance, timeout_seconds, poll_interval_seconds

    monkeypatch.setattr(quality_matrix, "inspect_matrix_environment", inspect)
    monkeypatch.setattr(quality_matrix, "_wait_for_instance", wait_for_instance)


def _stub_tokenizer_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    def validate(
        config: QualityMatrixConfig,
        loaded_prompts: dict[PromptClass, LoadedPrompt],
    ) -> tuple[
        dict[PromptClass, LoadedPrompt],
        TokenizerEvidence,
    ]:
        del config
        validated: dict[PromptClass, LoadedPrompt] = {}
        for prompt_class, loaded in loaded_prompts.items():
            token_target = quality_matrix.PROMPT_TOKEN_TARGETS[prompt_class]
            validated[prompt_class] = LoadedPrompt(
                plaintext=loaded.plaintext,
                evidence=loaded.evidence.model_copy(
                    update={
                        "target_prompt_tokens": token_target,
                        "actual_prompt_tokens": token_target,
                    }
                ),
            )
        return validated, TokenizerEvidence(
            metadata_sha256="sha256:" + "a" * 64,
            file_count=1,
        )

    monkeypatch.setattr(quality_matrix, "validate_prompt_token_counts", validate)


def _no_network_client(config: QualityMatrixConfig) -> httpx.Client:
    def reject_unexpected_http(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected HTTP request to {request.url.path}")

    return httpx.Client(
        transport=httpx.MockTransport(reject_unexpected_http),
        base_url=config.base_url,
    )
