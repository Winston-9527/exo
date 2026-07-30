"""Reproducible two-provider quality and access-audit matrix runner."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from pathlib import Path
from typing import Literal, Protocol, TypeVar, cast
from uuid import uuid4

import httpx
from pydantic import Field, JsonValue, TypeAdapter, ValidationError, model_validator

from exo.api.types import PlacementPreview
from exo.shared.types.backends import Backend
from exo.shared.types.common import CommandId, ModelId, NodeId
from exo.shared.types.state import State
from exo.shared.types.text_generation import InputMessage, InputMessageContent
from exo.shared.types.verifiable import (
    VerifiableAuditResponse,
    VerifiableChatCompletionRequest,
    VerifiableGenerationParams,
    VerifiableInputReceipt,
    VerifiableProviderIdentity,
    VerifiableRecipient,
)
from exo.shared.types.worker.instances import Instance, InstanceId
from exo.shared.types.worker.runners import RunnerFailed, RunnerReady
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.utils.pydantic_ext import FrozenModel
from exo.verifiable.client import build_verifiable_chat_request
from exo.verifiable.identity import provider_id_from_public_key
from exo.verifiable.placement import placement_digest
from exo.verifiable.private_types import VerifiablePrivateTaskPayload
from exo.verifiable.quality_check import (
    QualityCheckConfig,
    QualityCheckError,
    QualityCheckReport,
    run_quality_check,
)

PromptClass = Literal[
    "ascii_short",
    "zh_unicode",
    "code_json",
    "prefill_4096",
    "prefill_4608",
]
MatrixPlane = Literal["quality", "clean_audit", "negative"]
NegativeControl = Literal[
    "non_ingress_submission",
    "placement_digest_tamper",
    "recipient_identity_tamper",
    "cross_epoch_replay",
]
AttemptStatus = Literal[
    "pass",
    "scientific_fail",
    "infra_fail",
    "expected_reject",
    "unexpected_accept",
]
FailureClass = Literal[
    "transport_connect",
    "transport_timeout",
    "http_error",
    "protocol_decode",
    "quality_mismatch",
    "audit_incomplete",
    "audit_binding_invalid",
    "privacy_violation",
    "unexpected_acceptance",
    "protocol_evidence_invalid",
    "preflight_failed",
    "harness_internal",
]

PROMPT_CLASSES: tuple[PromptClass, ...] = (
    "ascii_short",
    "zh_unicode",
    "code_json",
    "prefill_4096",
    "prefill_4608",
)
MAX_OUTPUT_TOKENS = (16, 64)
REPEAT_COUNT = 3
NEGATIVE_CONTROLS: tuple[NegativeControl, ...] = (
    "non_ingress_submission",
    "placement_digest_tamper",
    "recipient_identity_tamper",
    "cross_epoch_replay",
)
JSON_VALUE_ADAPTER: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)
PROTOCOL_VERSION = "verifiable-exo-matrix-v1"
LOCKED_MODEL = "mlx-community/Qwen3-0.6B-8bit"
PROMPT_TOKEN_TARGETS: dict[PromptClass, int] = {
    "ascii_short": 32,
    "zh_unicode": 64,
    "code_json": 256,
    "prefill_4096": 4096,
    "prefill_4608": 4608,
}
OperationResult = TypeVar("OperationResult")


class ChatTokenizer(Protocol):
    """Small boundary used to count exactly what the model chat template emits."""

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> object: ...

    def encode(self, text: str, *, add_special_tokens: bool) -> object: ...


class ChatTokenizerFactory(Protocol):
    """Typed view of the locally installed Transformers tokenizer factory."""

    def from_pretrained(
        self,
        pretrained_model_name_or_path: Path,
        *,
        local_files_only: bool,
        trust_remote_code: bool,
    ) -> ChatTokenizer: ...


class TransformersModule(Protocol):
    AutoTokenizer: ChatTokenizerFactory


class MatrixCell(FrozenModel):
    """One immutable coordinate in the locked two-provider matrix."""

    cell_id: str
    plane: MatrixPlane
    prompt_class: PromptClass | None = None
    max_output_tokens: int | None = Field(default=None, gt=0)
    repeat_index: int | None = Field(default=None, ge=0)
    negative_control: NegativeControl | None = None


class MatrixNodeEndpoint(FrozenModel):
    """A provider label and the API URL served by its EXO node."""

    label: str = Field(min_length=1)
    api_url: str = Field(min_length=1)


class RuntimeNodeProvenance(FrozenModel):
    """Redacted process, source, hardware, and model identity for one EXO node."""

    label: Literal["mini1", "rtx3090"]
    node_id: NodeId
    provider_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    key_id: str = Field(min_length=1)
    host_name: str = Field(min_length=1)
    operating_system: str = Field(min_length=1)
    architecture: str = Field(min_length=1)
    accelerator: str = Field(min_length=1)
    gpu_driver_version: str | None = None
    backends: list[Backend]
    process_id: int = Field(gt=0)
    process_argv: list[str] = Field(min_length=1)
    network_namespace: str = Field(min_length=1)
    api_port: int = Field(gt=0, lt=65536)
    zenoh_port: int = Field(gt=0, lt=65536)
    discovery_port: int = Field(gt=0, lt=65536)
    exo_home: str = Field(min_length=1)
    source_branch: str = Field(min_length=1)
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_tree_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source_dirty: bool
    python_version: str = Field(min_length=1)
    packages: dict[str, str]
    model_id: str = Field(min_length=1)
    model_files_sha256: dict[str, str]
    tokenizer_metadata_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    tokenizer_metadata_file_count: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_formal_node(self) -> "RuntimeNodeProvenance":
        if self.source_dirty:
            raise ValueError("Formal node source trees must be clean")
        required_packages = {"exo", "mlx", "mlx-lm"}
        if not required_packages.issubset(self.packages):
            raise ValueError("Runtime provenance is missing required packages")
        required_model_files = {
            "config.json",
            "model.safetensors",
            "tokenizer.json",
        }
        if not required_model_files.issubset(self.model_files_sha256):
            raise ValueError("Runtime provenance is missing required model hashes")
        if any(
            not value.startswith("sha256:") or len(value) != 71
            for value in self.model_files_sha256.values()
        ):
            raise ValueError("Runtime provenance contains an invalid model hash")
        if self.label == "mini1":
            if (
                Backend.MlxMetal not in self.backends
                or self.gpu_driver_version is not None
            ):
                raise ValueError("mini1 provenance must describe MLX Metal")
        elif Backend.MlxCuda not in self.backends or not self.gpu_driver_version:
            raise ValueError("RTX3090 provenance must describe MLX CUDA and its driver")
        return self


class RuntimeProvenance(FrozenModel):
    captured_at: str = Field(min_length=1)
    nodes: list[RuntimeNodeProvenance]

    @model_validator(mode="after")
    def validate_two_node_runtime(self) -> "RuntimeProvenance":
        if {node.label for node in self.nodes} != {"mini1", "rtx3090"}:
            raise ValueError("Runtime provenance must contain mini1 and RTX3090")
        if len(self.nodes) != 2 or len({node.node_id for node in self.nodes}) != 2:
            raise ValueError("Runtime provenance must contain two unique EXO nodes")
        if len({node.network_namespace for node in self.nodes}) != 1:
            raise ValueError("Runtime nodes must share one isolated namespace")
        if (
            len({node.source_commit for node in self.nodes}) != 1
            or len({node.source_tree_sha256 for node in self.nodes}) != 1
        ):
            raise ValueError("Runtime nodes must execute the same clean source tree")
        if (
            len(
                {
                    json.dumps(node.model_files_sha256, sort_keys=True)
                    for node in self.nodes
                }
            )
            != 1
        ):
            raise ValueError("Runtime nodes must use byte-identical model artifacts")
        if (
            len(
                {
                    (node.tokenizer_metadata_sha256, node.tokenizer_metadata_file_count)
                    for node in self.nodes
                }
            )
            != 1
        ):
            raise ValueError("Runtime nodes must use byte-identical tokenizer metadata")
        for package_name in ("exo", "mlx", "mlx-lm"):
            if len({node.packages[package_name] for node in self.nodes}) != 1:
                raise ValueError("Runtime nodes must share core package versions")
        if len({node.exo_home for node in self.nodes}) != 2:
            raise ValueError("Runtime nodes must use distinct isolated EXO homes")
        return self


class MatrixPromptFile(FrozenModel):
    """Private prompt input; its filesystem path is never copied to results."""

    prompt_class: PromptClass
    path: str = Field(min_length=1)
    target_prompt_tokens: int | None = Field(default=None, gt=0)


class QualityMatrixConfig(FrozenModel):
    """Locked two-provider matrix configuration."""

    base_url: str = Field(min_length=1)
    nodes: list[MatrixNodeEndpoint]
    instance_id: str | None = Field(default=None, min_length=1)
    placement_preview_file: str | None = Field(default=None, min_length=1)
    replay_placement_preview_file: str | None = Field(default=None, min_length=1)
    runtime_provenance_file: str | None = Field(default=None, min_length=1)
    allow_placement_epoch_transition: bool = False
    tokenizer_directory: str | None = Field(default=None, min_length=1)
    output_directory: str = Field(min_length=1)
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    prompts: list[MatrixPromptFile]
    model: str = LOCKED_MODEL
    seed: int = 42
    request_timeout_seconds: float = Field(default=300.0, gt=0.0)
    audit_timeout_seconds: float = Field(default=30.0, ge=0.0)
    event_timeout_seconds: float = Field(default=30.0, ge=0.0)
    placement_timeout_seconds: float = Field(default=180.0, gt=0.0)
    poll_interval_seconds: float = Field(default=0.2, gt=0.0)
    resume: bool = False

    @model_validator(mode="after")
    def validate_locked_inputs(self) -> "QualityMatrixConfig":
        if self.model != LOCKED_MODEL:
            raise ValueError("The formal matrix is locked to Qwen3-0.6B-8bit")
        if len(self.nodes) != 2:
            raise ValueError("The quality matrix requires exactly two node APIs")
        if len({node.label for node in self.nodes}) != 2:
            raise ValueError("Matrix node labels must be unique")
        if {node.label.lower() for node in self.nodes} != {"mini1", "rtx3090"}:
            raise ValueError("The formal matrix requires mini1 and RTX3090")
        if len({node.api_url.rstrip("/") for node in self.nodes}) != 2:
            raise ValueError("Matrix node API URLs must be unique")
        if (
            self.replay_placement_preview_file is not None
            and not self.allow_placement_epoch_transition
        ):
            raise ValueError(
                "A replay placement preview requires explicit epoch-transition consent"
            )
        prompt_classes = [prompt.prompt_class for prompt in self.prompts]
        if len(prompt_classes) != len(PROMPT_CLASSES) or set(prompt_classes) != set(
            PROMPT_CLASSES
        ):
            raise ValueError(
                "The quality matrix requires each locked prompt class once"
            )
        return self


class MatrixEnvironment(FrozenModel):
    """Live two-node placement and provider identities used by every cell."""

    instance: Instance
    world_size: int = Field(gt=0)
    ingress_node_id: NodeId
    ingress_url: str
    downstream_node_id: NodeId
    downstream_url: str
    identities: dict[NodeId, VerifiableProviderIdentity]
    backends: dict[NodeId, list[Backend]] = Field(default_factory=dict)


class PromptEvidence(FrozenModel):
    """Non-reversible prompt metadata safe for a public result artifact."""

    prompt_class: PromptClass
    sha256: str
    utf8_bytes: int = Field(ge=0)
    target_prompt_tokens: int | None = Field(default=None, gt=0)
    actual_prompt_tokens: int | None = Field(default=None, gt=0)


class TokenizerEvidence(FrozenModel):
    """Public fingerprint of the private tokenizer metadata directory."""

    metadata_sha256: str
    file_count: int = Field(gt=0)


class NodeManifest(FrozenModel):
    label: str
    api_url: str
    node_id: NodeId
    provider_id: str
    key_id: str
    public_key_sha256: str
    backends: list[Backend]
    device_rank: int = Field(ge=0)
    world_size: int = Field(gt=0)
    start_layer: int = Field(ge=0)
    end_layer: int = Field(gt=0)


class MatrixManifest(FrozenModel):
    protocol_version: Literal["verifiable-exo-matrix-v1"] = PROTOCOL_VERSION
    run_id: str
    created_at: str
    formal_execution: bool
    model: str
    instance_id: str
    placement_digest: str
    placement_preview_sha256: str | None
    replay_placement_preview_sha256: str | None
    runtime_provenance_sha256: str | None
    seed: int
    temperature: float = 0.0
    request_timeout_seconds: float
    event_timeout_seconds: float
    audit_timeout_seconds: float
    placement_timeout_seconds: float
    poll_interval_seconds: float
    quality_output_tokens: list[int]
    clean_audit_output_tokens: int
    branch: str | None
    commit: str | None
    source_dirty: bool | None
    source_sha256: str
    lockfile_sha256: str | None
    python_version: str
    platform: str
    tokenizer: TokenizerEvidence
    prompts: list[PromptEvidence]
    nodes: list[NodeManifest]
    cells: list[MatrixCell]
    reproducibility_sha256: str


class RawReceiptEvidence(FrozenModel):
    """Independent receipt evidence recovered from the append-only event log."""

    event_ids: list[str]
    receipt_count: int = Field(ge=0)
    matches_aggregate: bool
    binding_errors: list[str]


class QualityCellEvidence(FrozenModel):
    kind: Literal["quality"] = "quality"
    report: QualityCheckReport
    all_receipts_live_identity_bound: bool
    all_receipts_frozen_placement_bound: bool
    preflight_prompt_tokens: int | None = Field(default=None, gt=0)
    receipt_prompt_tokens: int | None = Field(default=None, gt=0)
    prompt_token_count_matches: bool
    raw_receipts: RawReceiptEvidence
    receipts: list[VerifiableInputReceipt]


class CleanAuditCellEvidence(FrozenModel):
    kind: Literal["clean_audit"] = "clean_audit"
    execution_id: CommandId
    output_sha256: str
    output_utf8_bytes: int = Field(ge=0)
    ciphertext_sha256: str
    prompt_token_count: int | None = Field(default=None, gt=0)
    receipt_count: int = Field(ge=0)
    complete: bool
    bindings_valid: bool
    binding_errors: list[str]
    ingress_only_private_access: bool
    task_input_empty: bool
    task_envelope_bound: bool
    canary_absent_from_public_events: bool
    raw_receipts: RawReceiptEvidence
    receipts: list[VerifiableInputReceipt]


class NegativeCellEvidence(FrozenModel):
    kind: Literal["negative"] = "negative"
    control: NegativeControl
    http_status: int
    rejected: bool
    protocol_rejection_valid: bool
    task_created: bool
    receipt_event_created: bool
    audit_created: bool


CellEvidence = QualityCellEvidence | CleanAuditCellEvidence | NegativeCellEvidence


class CellExecutionResult(FrozenModel):
    status: AttemptStatus
    failure_class: FailureClass | None = None
    evidence: CellEvidence | None = None


class MatrixSummary(FrozenModel):
    protocol_version: Literal["verifiable-exo-matrix-v1"] = PROTOCOL_VERSION
    run_id: str
    updated_at: str
    formal_execution: bool
    total_cells: int = Field(ge=0)
    terminal_cells: int = Field(ge=0)
    missing_cells: int = Field(ge=0)
    status_counts: dict[str, int]
    attempt_count: int = Field(ge=0)
    attempt_status_counts: dict[str, int]
    retried_cells: list[str]
    passed: bool
    formal_passed: bool


@dataclass(frozen=True)
class LoadedPrompt:
    """A private in-memory prompt paired with its public fingerprint."""

    plaintext: str
    evidence: PromptEvidence
    leak_sentinels: tuple[str, ...] = ()


class AttemptHandle(FrozenModel):
    """Fresh identifiers allocated before a cell performs any external work."""

    run_id: str
    cell_id: str
    attempt_id: str
    attempt_index: int = Field(ge=0)
    request_id: str
    started_at: str


class AttemptRecord(FrozenModel):
    """One append-only state transition in the public attempt journal."""

    protocol_version: Literal["verifiable-exo-matrix-v1"] = PROTOCOL_VERSION
    run_id: str
    cell_id: str
    attempt_id: str
    attempt_index: int = Field(ge=0)
    request_id: str
    event: Literal["started", "finished"]
    timestamp: str
    prompt: PromptEvidence | None = None
    status: AttemptStatus | None = None
    failure_class: FailureClass | None = None
    duration_seconds: float | None = Field(default=None, ge=0.0)
    evidence: CellEvidence | None = None


class AttemptJournal:
    """Durable append-only attempt ledger used for crash-safe resume."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def records(self) -> list[AttemptRecord]:
        if not self.path.exists():
            return []
        records: list[AttemptRecord] = []
        with self.path.open(encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    records.append(AttemptRecord.model_validate_json(line))
        return records

    def validate(
        self,
        *,
        run_id: str,
        cells: Sequence[MatrixCell],
    ) -> list[AttemptRecord]:
        """Validate that this journal is one well-formed run of the locked plan."""
        records = self.records()
        allowed_cell_ids = {cell.cell_id for cell in cells}
        attempts: dict[str, AttemptRecord] = {}
        finished_attempt_ids: set[str] = set()
        finished_statuses: dict[str, AttemptStatus] = {}
        request_ids: set[str] = set()
        started_per_cell: Counter[str] = Counter()
        latest_attempt_by_cell: dict[str, AttemptRecord] = {}

        for record in records:
            if record.run_id != run_id:
                raise ValueError("Attempt journal belongs to another run")
            if record.cell_id not in allowed_cell_ids:
                raise ValueError("Attempt journal contains an unknown matrix cell")

            if record.event == "started":
                if record.attempt_id in attempts:
                    raise ValueError("Attempt journal contains a duplicate start")
                if record.request_id in request_ids:
                    raise ValueError("Attempt journal reuses a request id")
                expected_index = started_per_cell[record.cell_id]
                if record.attempt_index != expected_index or expected_index >= 2:
                    raise ValueError("Attempt journal has an invalid retry sequence")
                if expected_index:
                    previous = latest_attempt_by_cell[record.cell_id]
                    if finished_statuses.get(previous.attempt_id) != "infra_fail":
                        raise ValueError(
                            "Attempt journal retries a non-infrastructure outcome"
                        )
                if any(
                    value is not None
                    for value in (
                        record.status,
                        record.failure_class,
                        record.duration_seconds,
                        record.evidence,
                    )
                ):
                    raise ValueError("Attempt start contains terminal evidence")
                attempts[record.attempt_id] = record
                latest_attempt_by_cell[record.cell_id] = record
                request_ids.add(record.request_id)
                started_per_cell[record.cell_id] += 1
                continue

            started = attempts.get(record.attempt_id)
            if started is None:
                raise ValueError("Attempt journal finishes an attempt before its start")
            if record.attempt_id in finished_attempt_ids:
                raise ValueError("Attempt journal contains a duplicate finish")
            if (
                record.cell_id != started.cell_id
                or record.attempt_index != started.attempt_index
                or record.request_id != started.request_id
            ):
                raise ValueError("Attempt finish does not match its start")
            if record.prompt is not None or record.status is None:
                raise ValueError("Attempt finish has an invalid terminal transition")
            finished_attempt_ids.add(record.attempt_id)
            finished_statuses[record.attempt_id] = record.status

        return records

    def start(
        self,
        run_id: str,
        cell: MatrixCell,
        prompt: PromptEvidence | None,
    ) -> AttemptHandle:
        previous_indices = [
            record.attempt_index
            for record in self.records()
            if record.cell_id == cell.cell_id
        ]
        attempt_index = max(previous_indices, default=-1) + 1
        attempt_id = str(uuid4())
        request_id = f"matrix-{run_id}-{cell.plane}-{uuid4()}"
        started_at = _utc_now()
        handle = AttemptHandle(
            run_id=run_id,
            cell_id=cell.cell_id,
            attempt_id=attempt_id,
            attempt_index=attempt_index,
            request_id=request_id,
            started_at=started_at,
        )
        self._append(
            AttemptRecord(
                run_id=run_id,
                cell_id=cell.cell_id,
                attempt_id=attempt_id,
                attempt_index=attempt_index,
                request_id=request_id,
                event="started",
                timestamp=started_at,
                prompt=prompt,
            )
        )
        return handle

    def finish(
        self,
        handle: AttemptHandle,
        *,
        status: AttemptStatus,
        failure_class: FailureClass | None = None,
        duration_seconds: float | None = None,
        evidence: CellEvidence | None = None,
    ) -> None:
        self._append(
            AttemptRecord(
                run_id=handle.run_id,
                cell_id=handle.cell_id,
                attempt_id=handle.attempt_id,
                attempt_index=handle.attempt_index,
                request_id=handle.request_id,
                event="finished",
                timestamp=_utc_now(),
                status=status,
                failure_class=failure_class,
                duration_seconds=duration_seconds,
                evidence=evidence,
            )
        )

    def terminal_cell_ids(self) -> set[str]:
        return {
            record.cell_id
            for record in self.records()
            if record.event == "finished" and record.status != "infra_fail"
        }

    def orphaned_attempt_ids(self) -> list[str]:
        return _orphaned_attempt_ids(self.records())

    def _append(self, record: AttemptRecord) -> None:
        serialized = record.model_dump_json() + "\n"
        file_descriptor = os.open(
            self.path,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT,
            0o600,
        )
        with os.fdopen(file_descriptor, "a", encoding="utf-8") as file:
            _ = file.write(serialized)
            file.flush()
            os.fsync(file.fileno())


def _orphaned_attempt_ids(records: Sequence[AttemptRecord]) -> list[str]:
    started = {record.attempt_id for record in records if record.event == "started"}
    finished = {record.attempt_id for record in records if record.event == "finished"}
    return sorted(started - finished)


def load_prompt_set(config: QualityMatrixConfig) -> dict[PromptClass, LoadedPrompt]:
    """Read the five private inputs without exposing their paths or text."""
    loaded: dict[PromptClass, LoadedPrompt] = {}
    for prompt_file in config.prompts:
        plaintext = Path(prompt_file.path).read_text(encoding="utf-8")
        if not plaintext:
            raise ValueError(f"Prompt class {prompt_file.prompt_class} is empty")
        encoded = plaintext.encode("utf-8")
        loaded[prompt_file.prompt_class] = LoadedPrompt(
            plaintext=plaintext,
            evidence=PromptEvidence(
                prompt_class=prompt_file.prompt_class,
                sha256=f"sha256:{hashlib.sha256(encoded).hexdigest()}",
                utf8_bytes=len(encoded),
                target_prompt_tokens=prompt_file.target_prompt_tokens,
            ),
        )
    return loaded


def validate_prompt_token_counts(
    config: QualityMatrixConfig,
    loaded_prompts: dict[PromptClass, LoadedPrompt],
) -> tuple[dict[PromptClass, LoadedPrompt], TokenizerEvidence]:
    """Count the actual Qwen chat-template tokens before any network request.

    The tokenizer is loaded from a caller-provided local directory.  Long-input
    classes are rejected unless their rendered lengths are exactly 4096/4608;
    neither rendered token IDs nor tokenizer paths enter public artifacts.
    """
    if config.tokenizer_directory is None:
        raise ValueError("The formal matrix requires --tokenizer-dir")
    tokenizer_directory = Path(config.tokenizer_directory)
    if not tokenizer_directory.is_dir():
        raise ValueError("The tokenizer directory does not exist")

    # Import lazily so matrix schema/unit tests do not initialize ML runtimes.
    tokenizer = _load_chat_tokenizer(tokenizer_directory)
    validated: dict[PromptClass, LoadedPrompt] = {}
    for prompt_class, loaded in loaded_prompts.items():
        token_ids = _single_user_chat_token_ids(
            tokenizer,
            tokenizer_directory,
            loaded.plaintext,
            allow_qwen_fallback=(config.model == LOCKED_MODEL),
        )
        actual_prompt_tokens = len(token_ids)
        fixed_target = PROMPT_TOKEN_TARGETS[prompt_class]
        configured_target = loaded.evidence.target_prompt_tokens
        if configured_target not in (None, fixed_target):
            raise ValueError(
                f"Prompt class {prompt_class} must target exactly {fixed_target} tokens"
            )
        target = fixed_target
        if actual_prompt_tokens != target:
            raise ValueError(
                f"Prompt class {prompt_class} rendered to {actual_prompt_tokens} "
                f"tokens, expected {target}"
            )
        validated[prompt_class] = LoadedPrompt(
            plaintext=loaded.plaintext,
            evidence=loaded.evidence.model_copy(
                update={
                    "target_prompt_tokens": target,
                    "actual_prompt_tokens": actual_prompt_tokens,
                }
            ),
        )

    tokenizer_files = sorted(
        path
        for path in tokenizer_directory.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    if not tokenizer_files:
        raise ValueError("The tokenizer directory contains no metadata files")
    tokenizer_digest = hashlib.sha256()
    for path in tokenizer_files:
        relative_path = path.relative_to(tokenizer_directory).as_posix().encode("utf-8")
        tokenizer_digest.update(len(relative_path).to_bytes(8, "big"))
        tokenizer_digest.update(relative_path)
        content = path.read_bytes()
        tokenizer_digest.update(len(content).to_bytes(8, "big"))
        tokenizer_digest.update(content)
    return validated, TokenizerEvidence(
        metadata_sha256=f"sha256:{tokenizer_digest.hexdigest()}",
        file_count=len(tokenizer_files),
    )


def _load_chat_tokenizer(tokenizer_directory: Path) -> ChatTokenizer:
    transformers_module = cast(
        TransformersModule, cast(object, importlib.import_module("transformers"))
    )
    return transformers_module.AutoTokenizer.from_pretrained(
        tokenizer_directory,
        local_files_only=True,
        trust_remote_code=False,
    )


def _single_user_chat_token_ids(
    tokenizer: ChatTokenizer,
    tokenizer_directory: Path,
    prompt: str,
    *,
    allow_qwen_fallback: bool,
) -> list[int]:
    try:
        untyped_token_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
    except ImportError:
        if not allow_qwen_fallback:
            raise ValueError(
                "Chat-template rendering is unavailable for this tokenizer"
            ) from None
        _validate_qwen_single_user_template(tokenizer_directory)
        rendered = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        untyped_token_ids = tokenizer.encode(rendered, add_special_tokens=False)
    if not isinstance(untyped_token_ids, list):
        raise ValueError("The tokenizer returned non-list chat token IDs")
    token_objects = cast(list[object], untyped_token_ids)
    if not all(
        isinstance(token_id, int) and not isinstance(token_id, bool)
        for token_id in token_objects
    ):
        raise ValueError("The tokenizer returned non-integer chat token IDs")
    return cast(list[int], token_objects)


def _validate_qwen_single_user_template(tokenizer_directory: Path) -> None:
    configuration_path = tokenizer_directory / "tokenizer_config.json"
    if not configuration_path.is_file():
        raise ValueError("Qwen fallback requires tokenizer_config.json")
    payload = JSON_VALUE_ADAPTER.validate_json(configuration_path.read_bytes())
    if not isinstance(payload, dict):
        raise ValueError("Qwen tokenizer configuration is not an object")
    chat_template = payload.get("chat_template")
    if not isinstance(chat_template, str) or not all(
        marker in chat_template
        for marker in ("<|im_start|>", "<|im_end|>", "add_generation_prompt")
    ):
        raise ValueError("Qwen tokenizer chat template does not match the fallback")


def prepare_matrix_environment(
    client: httpx.Client, config: QualityMatrixConfig
) -> MatrixEnvironment:
    """Create an exact saved preview or bind a unique already-live instance."""
    resolved_config = config
    if config.placement_preview_file is not None:
        preview = PlacementPreview.model_validate_json(
            Path(config.placement_preview_file).read_bytes()
        )
        if preview.instance is None or preview.error is not None:
            raise ValueError("The saved placement preview has no usable instance")
        if str(preview.model_id) != config.model:
            raise ValueError("The saved placement preview is for another model")
        expected_instance = preview.instance
        state = _fetch_state(client)
        live_instance = state.instances.get(expected_instance.instance_id)
        if live_instance is None:
            response = client.post(
                "/instance",
                json={
                    "instance": expected_instance.model_dump(mode="json", by_alias=True)
                },
            )
            response.raise_for_status()
            _wait_for_instance(
                client,
                expected_instance,
                timeout_seconds=config.placement_timeout_seconds,
                poll_interval_seconds=config.poll_interval_seconds,
            )
        elif live_instance != expected_instance:
            raise ValueError("The live instance ID conflicts with the saved preview")
        if config.instance_id is not None and config.instance_id != str(
            expected_instance.instance_id
        ):
            raise ValueError("Configured instance ID differs from the saved preview")
        resolved_config = config.model_copy(
            update={"instance_id": str(expected_instance.instance_id)}
        )
    environment = inspect_matrix_environment(client, resolved_config)
    _wait_for_instance(
        client,
        environment.instance,
        timeout_seconds=config.placement_timeout_seconds,
        poll_interval_seconds=config.poll_interval_seconds,
    )
    # Re-bind the placement and both identities after readiness so the manifest
    # cannot mix pre-ready state with a later provider epoch.
    return inspect_matrix_environment(client, resolved_config)


def _fetch_state(client: httpx.Client) -> State:
    response = client.get("/state")
    response.raise_for_status()
    return State.model_validate_json(response.content)


def _wait_for_instance(
    client: httpx.Client,
    expected_instance: Instance,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        state = _fetch_state(client)
        live_instance = state.instances.get(expected_instance.instance_id)
        if live_instance is not None:
            if live_instance != expected_instance:
                raise ValueError("Created instance differs from the exact preview")
            runner_ids = set(expected_instance.shard_assignments.runner_to_shard)
            statuses = [state.runners.get(runner_id) for runner_id in runner_ids]
            if any(isinstance(status, RunnerFailed) for status in statuses):
                # Do not include RunnerFailed.error_message: it is remote text
                # and therefore outside the redacted evidence boundary.
                raise RuntimeError("Placement runner failed before becoming ready")
            if statuses and all(isinstance(status, RunnerReady) for status in statuses):
                return
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            raise TimeoutError("Timed out waiting for the exact preview instance")
        time.sleep(min(poll_interval_seconds, remaining_seconds))


def inspect_matrix_environment(
    client: httpx.Client, config: QualityMatrixConfig
) -> MatrixEnvironment:
    """Bind both live node identities to one ready two-rank pipeline placement."""
    state = _fetch_state(client)
    matching_instances = [
        instance
        for instance in state.instances.values()
        if str(instance.shard_assignments.model_id) == config.model
    ]
    if config.instance_id is None and len(matching_instances) != 1:
        raise ValueError("The matrix requires exactly one matching model instance")
    if config.instance_id is None:
        instance = matching_instances[0]
    else:
        selected = [
            instance
            for instance in matching_instances
            if str(instance.instance_id) == config.instance_id
        ]
        if len(selected) != 1:
            raise ValueError("The configured instance does not match live state")
        instance = selected[0]

    expected_by_rank: dict[int, tuple[NodeId, PipelineShardMetadata]] = {}
    for node_id, runner_id in instance.shard_assignments.node_to_runner.items():
        shard = instance.shard_assignments.runner_to_shard[runner_id]
        if not isinstance(shard, PipelineShardMetadata):
            raise ValueError("The matrix requires pure pipeline sharding")
        if shard.device_rank in expected_by_rank:
            raise ValueError("The matrix placement contains a duplicate rank")
        expected_by_rank[shard.device_rank] = (node_id, shard)
    if set(expected_by_rank) != {0, 1}:
        raise ValueError("The matrix requires exactly two contiguous pipeline ranks")
    if any(shard.world_size != 2 for _, shard in expected_by_rank.values()):
        raise ValueError("The matrix requires pipeline world size two")
    ingress_node_id, ingress_shard = expected_by_rank[0]
    if not ingress_shard.is_first_layer:
        raise ValueError("Pipeline rank zero must own the first model layer")
    downstream_node_id, downstream_shard = expected_by_rank[1]
    if downstream_shard.is_first_layer:
        raise ValueError("The downstream rank cannot own the first model layer")

    identities: dict[NodeId, VerifiableProviderIdentity] = {}
    urls_by_node: dict[NodeId, str] = {}
    labels_by_node: dict[NodeId, str] = {}
    for endpoint in config.nodes:
        api_url = endpoint.api_url.rstrip("/")
        node_response = client.get(f"{api_url}/node_id")
        node_response.raise_for_status()
        untyped_node_id = JSON_VALUE_ADAPTER.validate_json(node_response.content)
        if not isinstance(untyped_node_id, str):
            raise ValueError("A matrix node API returned an invalid node id")
        node_id = NodeId(untyped_node_id)
        identity_response = client.get(f"{api_url}/v1/verifiable/identity")
        identity_response.raise_for_status()
        identity = VerifiableProviderIdentity.model_validate_json(
            identity_response.content
        )
        if identity.node_id != node_id:
            raise ValueError("A live provider identity belongs to another node")
        if identity.provider_id != provider_id_from_public_key(identity.public_key):
            raise ValueError(
                "A live provider fingerprint does not match its public key"
            )
        if node_id in identities:
            raise ValueError("Two API endpoints returned the same node id")
        identities[node_id] = identity
        urls_by_node[node_id] = api_url
        labels_by_node[node_id] = endpoint.label.lower()

    placement_nodes = set(instance.shard_assignments.node_to_runner)
    if set(identities) != placement_nodes:
        raise ValueError("Node APIs do not exactly match the committed placement")
    if not all(node_id in state.node_backends for node_id in placement_nodes):
        raise ValueError("Backend evidence is missing for a placement node")
    observed_backends = {
        backend
        for node_id in placement_nodes
        for backend in state.node_backends[node_id]
    }
    if not {Backend.MlxMetal, Backend.MlxCuda}.issubset(observed_backends):
        raise ValueError("The matrix requires live MLX Metal and MLX CUDA providers")
    if (
        Backend.MlxMetal
        not in state.node_backends[
            next(node for node, label in labels_by_node.items() if label == "mini1")
        ]
    ):
        raise ValueError("mini1 must expose the live MLX Metal backend")
    if (
        Backend.MlxCuda
        not in state.node_backends[
            next(node for node, label in labels_by_node.items() if label == "rtx3090")
        ]
    ):
        raise ValueError("RTX3090 must expose the live MLX CUDA backend")

    return MatrixEnvironment(
        instance=instance,
        world_size=2,
        ingress_node_id=ingress_node_id,
        ingress_url=urls_by_node[ingress_node_id],
        downstream_node_id=downstream_node_id,
        downstream_url=urls_by_node[downstream_node_id],
        identities=identities,
        backends={
            node_id: list(state.node_backends[node_id]) for node_id in placement_nodes
        },
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_matrix_cells() -> list[MatrixCell]:
    """Return the locked 30 quality, 15 clean-audit, and 4 negative cells."""
    cells: list[MatrixCell] = []
    for prompt_class in PROMPT_CLASSES:
        for max_output_tokens in MAX_OUTPUT_TOKENS:
            for repeat_index in range(REPEAT_COUNT):
                cells.append(
                    MatrixCell(
                        cell_id=(
                            f"quality-{prompt_class}-out{max_output_tokens}-"
                            f"repeat{repeat_index}"
                        ),
                        plane="quality",
                        prompt_class=prompt_class,
                        max_output_tokens=max_output_tokens,
                        repeat_index=repeat_index,
                    )
                )

    for prompt_class in PROMPT_CLASSES:
        for repeat_index in range(REPEAT_COUNT):
            cells.append(
                MatrixCell(
                    cell_id=f"clean-audit-{prompt_class}-repeat{repeat_index}",
                    plane="clean_audit",
                    prompt_class=prompt_class,
                    max_output_tokens=16,
                    repeat_index=repeat_index,
                )
            )

    for negative_control in NEGATIVE_CONTROLS:
        cells.append(
            MatrixCell(
                cell_id=f"negative-{negative_control}",
                plane="negative",
                max_output_tokens=16,
                negative_control=negative_control,
            )
        )
    return cells


CellExecutor = Callable[
    [
        httpx.Client,
        QualityMatrixConfig,
        MatrixEnvironment,
        MatrixCell,
        LoadedPrompt | None,
        str,
    ],
    CellExecutionResult,
]


def run_quality_matrix(
    client: httpx.Client,
    config: QualityMatrixConfig,
    *,
    execute_cell: CellExecutor | None = None,
) -> MatrixSummary:
    """Run or resume the locked matrix while retaining every failed attempt.

    Exceptions are deliberately classified at the per-cell boundary without
    serializing exception messages: HTTP errors may contain requester input.
    Scientific failures are terminal; an infrastructure failure can be retried
    once, and only by a later ``resume=True`` invocation.
    """
    formal_execution = execute_cell is None
    cells = build_matrix_cells()
    output_directory = Path(config.output_directory)
    manifest_path = output_directory / "manifest.json"
    attempts_path = output_directory / "attempts.jsonl"
    summary_path = output_directory / "summary.json"
    provenance_artifact = output_directory / "runtime_provenance.json"
    artifact_paths = (
        manifest_path,
        attempts_path,
        summary_path,
        provenance_artifact,
    )

    existing_manifest: MatrixManifest | None = None
    if config.resume:
        if not manifest_path.is_file():
            raise ValueError("Cannot resume without an existing result manifest")
        existing_manifest = MatrixManifest.model_validate_json(
            manifest_path.read_bytes()
        )
        _validate_early_resume_manifest(
            existing_manifest,
            config,
            cells,
            formal_execution=formal_execution,
        )
    elif any(path.exists() for path in artifact_paths):
        raise ValueError("Result artifacts already exist; use --resume")

    output_directory.mkdir(parents=True, exist_ok=True)
    journal = AttemptJournal(attempts_path)
    initial_records = journal.validate(run_id=config.run_id, cells=cells)
    if config.resume and _orphaned_attempt_ids(initial_records):
        raise ValueError(
            "Cannot resume because an earlier attempt has an unknown outcome"
        )
    raw_prompts = load_prompt_set(config)
    prompts, tokenizer_evidence = validate_prompt_token_counts(config, raw_prompts)
    environment = prepare_matrix_environment(client, config)
    runtime_provenance: RuntimeProvenance | None = None
    if formal_execution:
        _validate_formal_timing(config)
        controller_branch, controller_commit, controller_tree_sha256 = (
            _require_clean_controller_source()
        )
        runtime_provenance = _load_runtime_provenance(config)
        _validate_runtime_provenance(
            runtime_provenance,
            config,
            environment,
            controller_branch=controller_branch,
            controller_commit=controller_commit,
            controller_tree_sha256=controller_tree_sha256,
            tokenizer_evidence=tokenizer_evidence,
        )
    manifest = _build_manifest(
        config,
        environment,
        prompts,
        tokenizer_evidence,
        cells,
        runtime_provenance,
        formal_execution=formal_execution,
    )
    if existing_manifest is not None:
        if existing_manifest.reproducibility_sha256 != manifest.reproducibility_sha256:
            raise ValueError("Resume manifest does not match the live experiment")
    else:
        _atomic_write_model(manifest_path, manifest)
    if runtime_provenance is not None:
        if provenance_artifact.exists():
            existing_provenance = RuntimeProvenance.model_validate_json(
                provenance_artifact.read_bytes()
            )
            if existing_provenance != runtime_provenance:
                raise ValueError("Runtime provenance artifact does not match this run")
        else:
            _atomic_write_model(provenance_artifact, runtime_provenance)

    terminal_cells = journal.terminal_cell_ids()
    executor = execute_cell or execute_matrix_cell
    for cell in cells:
        if cell.cell_id in terminal_cells:
            continue
        if _attempt_count(journal.records(), cell.cell_id) >= 2:
            continue
        if (
            cell.negative_control == "cross_epoch_replay"
            and not _all_prior_cells_succeeded(cell, cells, journal.records())
        ):
            break
        loaded_prompt = _private_prompt_for_cell(config, cell, prompts)
        handle = journal.start(
            config.run_id,
            cell,
            loaded_prompt.evidence if loaded_prompt is not None else None,
        )
        started = time.monotonic()
        try:
            result = executor(
                client,
                config,
                environment,
                cell,
                loaded_prompt,
                handle.request_id,
            )
        except Exception as error:
            result = _cell_result_from_exception(error)
        journal.finish(
            handle,
            status=result.status,
            failure_class=result.failure_class,
            duration_seconds=time.monotonic() - started,
            evidence=result.evidence,
        )
        # Deliberately do not retry infrastructure failures within this run.
        # A human-visible resume preserves the failed attempt and its timing.

    summary = _summarize(
        config.run_id,
        cells,
        journal.records(),
        formal_execution=formal_execution,
    )
    _atomic_write_model(summary_path, summary)
    return summary


def _validate_early_resume_manifest(
    manifest: MatrixManifest,
    config: QualityMatrixConfig,
    cells: Sequence[MatrixCell],
    *,
    formal_execution: bool,
) -> None:
    """Reject cross-run resume before reading private input or touching EXO."""
    if manifest.run_id != config.run_id:
        raise ValueError("Resume manifest belongs to another run")
    if manifest.model != config.model:
        raise ValueError("Resume manifest names another model")
    if manifest.formal_execution != formal_execution:
        raise ValueError("Resume manifest execution mode does not match")
    if manifest.cells != list(cells):
        raise ValueError("Resume manifest does not contain the locked matrix")
    if (
        manifest.seed != config.seed
        or manifest.request_timeout_seconds != config.request_timeout_seconds
        or manifest.event_timeout_seconds != config.event_timeout_seconds
        or manifest.audit_timeout_seconds != config.audit_timeout_seconds
        or manifest.placement_timeout_seconds != config.placement_timeout_seconds
        or manifest.poll_interval_seconds != config.poll_interval_seconds
    ):
        raise ValueError("Resume manifest configuration does not match")


def execute_matrix_cell(
    client: httpx.Client,
    config: QualityMatrixConfig,
    environment: MatrixEnvironment,
    cell: MatrixCell,
    loaded_prompt: LoadedPrompt | None,
    request_id: str,
) -> CellExecutionResult:
    """Execute one public matrix coordinate without writing artifacts."""
    if cell.plane == "quality":
        if loaded_prompt is None or cell.max_output_tokens is None:
            raise ValueError("A quality cell requires a private prompt and output cap")
        return _execute_quality_cell(
            client, config, environment, cell, loaded_prompt, request_id
        )
    if cell.plane == "clean_audit":
        if loaded_prompt is None or cell.max_output_tokens is None:
            raise ValueError("A clean-audit cell requires a private canary")
        return _execute_clean_audit_cell(
            client, config, environment, cell, loaded_prompt, request_id
        )
    if cell.negative_control is None:
        raise ValueError("A negative cell requires a control name")
    return _execute_negative_cell(
        client, config, environment, cell.negative_control, request_id
    )


def _execute_quality_cell(
    client: httpx.Client,
    config: QualityMatrixConfig,
    environment: MatrixEnvironment,
    cell: MatrixCell,
    loaded_prompt: LoadedPrompt,
    request_id: str,
) -> CellExecutionResult:
    report = run_quality_check(
        client,
        QualityCheckConfig(
            model=ModelId(config.model),
            prompt=loaded_prompt.plaintext,
            request_id=request_id,
            instance_id=environment.instance.instance_id,
            expected_instance=environment.instance,
            ingress_url=environment.ingress_url,
            max_output_tokens=cast(int, cell.max_output_tokens),
            seed=config.seed,
            events_timeout_seconds=config.event_timeout_seconds,
            events_poll_interval_seconds=config.poll_interval_seconds,
            audit_timeout_seconds=config.audit_timeout_seconds,
            audit_poll_interval_seconds=config.poll_interval_seconds,
        ),
    )
    audit = VerifiableAuditResponse(
        request_id=report.request_id,
        execution_id=report.audit.execution_id,
        expected_ranks=report.audit.expected_ranks,
        receipts=report.audit.receipts,
    )
    events = _fetch_cell_events(
        client,
        report.verifiable.command_id,
        timeout_seconds=config.event_timeout_seconds,
        poll_interval_seconds=config.poll_interval_seconds,
    )
    raw_receipts = _raw_receipt_evidence(
        events,
        audit,
        request_id=request_id,
        execution_id=CommandId(report.verifiable.command_id),
        expected_ranks=environment.world_size,
    )
    live_identity_bound = _receipts_match_live_identities(audit, environment)
    frozen_placement_bound = _receipts_match_frozen_placement(audit, environment)
    receipt_prompt_counts = {receipt.prompt_token_count for receipt in audit.receipts}
    receipt_prompt_tokens = (
        next(iter(receipt_prompt_counts)) if len(receipt_prompt_counts) == 1 else None
    )
    preflight_prompt_tokens = loaded_prompt.evidence.actual_prompt_tokens
    prompt_token_count_matches = (
        preflight_prompt_tokens is not None
        and receipt_prompt_tokens == preflight_prompt_tokens
    )
    evidence = QualityCellEvidence(
        report=report,
        all_receipts_live_identity_bound=live_identity_bound,
        all_receipts_frozen_placement_bound=frozen_placement_bound,
        preflight_prompt_tokens=preflight_prompt_tokens,
        receipt_prompt_tokens=receipt_prompt_tokens,
        prompt_token_count_matches=prompt_token_count_matches,
        raw_receipts=raw_receipts,
        receipts=audit.receipts,
    )
    if (
        report.passed
        and live_identity_bound
        and frozen_placement_bound
        and prompt_token_count_matches
        and raw_receipts.matches_aggregate
    ):
        return CellExecutionResult(status="pass", evidence=evidence)
    if not report.comparison.token_ids_exact or not report.comparison.final_text_exact:
        failure_class: FailureClass = "quality_mismatch"
    elif not report.audit.complete:
        failure_class = "audit_incomplete"
    else:
        failure_class = "audit_binding_invalid"
    return CellExecutionResult(
        status="scientific_fail",
        failure_class=failure_class,
        evidence=evidence,
    )


def _execute_clean_audit_cell(
    client: httpx.Client,
    config: QualityMatrixConfig,
    environment: MatrixEnvironment,
    cell: MatrixCell,
    loaded_prompt: LoadedPrompt,
    request_id: str,
) -> CellExecutionResult:
    _assert_request_id_fresh(client, request_id)
    ingress_identity = environment.identities[environment.ingress_node_id]
    encrypted_request = _build_encrypted_request(
        config,
        environment,
        ingress_identity,
        loaded_prompt.plaintext,
        request_id,
        cast(int, cell.max_output_tokens),
    )
    response = client.post(
        f"{environment.ingress_url}/v1/verifiable/chat/completions",
        json=encrypted_request.model_dump(mode="json", by_alias=True),
    )
    response.raise_for_status()
    command_id, output_text = _chat_result(response)
    audit = _fetch_audit(
        client,
        request_id,
        CommandId(command_id),
        environment.world_size,
        timeout_seconds=config.audit_timeout_seconds,
        poll_interval_seconds=config.poll_interval_seconds,
    )
    events = _fetch_cell_events(
        client,
        command_id,
        timeout_seconds=config.event_timeout_seconds,
        poll_interval_seconds=config.poll_interval_seconds,
    )
    binding_errors, ingress_only = _validate_audit_bindings(
        audit,
        encrypted_request,
        environment,
        CommandId(command_id),
    )
    raw_receipts = _raw_receipt_evidence(
        events,
        audit,
        request_id=request_id,
        execution_id=CommandId(command_id),
        expected_ranks=environment.world_size,
    )
    binding_errors = sorted({*binding_errors, *raw_receipts.binding_errors})
    task_input_empty, task_envelope_bound = _task_created_evidence(
        events, command_id, request_id, environment.instance.instance_id
    )
    ranks = {receipt.device_rank for receipt in audit.receipts}
    complete = len(audit.receipts) == environment.world_size and ranks == set(
        range(environment.world_size)
    )
    prompt_token_counts = {receipt.prompt_token_count for receipt in audit.receipts}
    prompt_token_count = (
        next(iter(prompt_token_counts)) if len(prompt_token_counts) == 1 else None
    )
    if (
        loaded_prompt.evidence.actual_prompt_tokens is not None
        and prompt_token_count != loaded_prompt.evidence.actual_prompt_tokens
    ):
        binding_errors = sorted(
            {*binding_errors, "receipt_preflight_prompt_length_mismatch"}
        )
    canary_absent = not any(
        _json_contains_text(events, sensitive_text)
        for sensitive_text in (loaded_prompt.plaintext, *loaded_prompt.leak_sentinels)
    )
    evidence = CleanAuditCellEvidence(
        execution_id=CommandId(command_id),
        output_sha256=_sha256_text(output_text),
        output_utf8_bytes=len(output_text.encode("utf-8")),
        ciphertext_sha256=_sha256_text(encrypted_request.encrypted_input.ciphertext),
        prompt_token_count=prompt_token_count,
        receipt_count=len(audit.receipts),
        complete=complete,
        bindings_valid=not binding_errors,
        binding_errors=binding_errors,
        ingress_only_private_access=ingress_only,
        task_input_empty=task_input_empty,
        task_envelope_bound=task_envelope_bound,
        canary_absent_from_public_events=canary_absent,
        raw_receipts=raw_receipts,
        receipts=audit.receipts,
    )
    passed = (
        complete
        and not binding_errors
        and ingress_only
        and task_input_empty
        and task_envelope_bound
        and canary_absent
    )
    if passed:
        return CellExecutionResult(status="pass", evidence=evidence)
    failure_class: FailureClass
    if not canary_absent or not task_input_empty:
        failure_class = "privacy_violation"
    elif not complete:
        failure_class = "audit_incomplete"
    else:
        failure_class = "audit_binding_invalid"
    return CellExecutionResult(
        status="scientific_fail",
        failure_class=failure_class,
        evidence=evidence,
    )


def _execute_negative_cell(
    client: httpx.Client,
    config: QualityMatrixConfig,
    environment: MatrixEnvironment,
    control: NegativeControl,
    request_id: str,
) -> CellExecutionResult:
    _assert_request_id_fresh(client, request_id)
    ingress_identity = environment.identities[environment.ingress_node_id]
    private_canary = f"negative-control-{control}-{uuid4()}"
    encrypted_request = _build_encrypted_request(
        config,
        environment,
        ingress_identity,
        private_canary,
        request_id,
        16,
    )
    target_url = environment.ingress_url
    submitted_request = encrypted_request
    if control == "non_ingress_submission":
        target_url = environment.downstream_url
    elif control == "placement_digest_tamper":
        submitted_request = encrypted_request.model_copy(
            update={"placement_digest": "sha256:" + "0" * 64}
        )
    elif control == "recipient_identity_tamper":
        downstream_identity = environment.identities[environment.downstream_node_id]
        submitted_request = encrypted_request.model_copy(
            update={
                "recipient": encrypted_request.recipient.model_copy(
                    update={
                        "provider_id": downstream_identity.provider_id,
                        "key_id": downstream_identity.key_id,
                    }
                )
            }
        )

    def submit_and_observe() -> CellExecutionResult:
        response = client.post(
            f"{target_url}/v1/verifiable/chat/completions",
            json=submitted_request.model_dump(mode="json", by_alias=True),
        )
        rejected = response.status_code >= 400
        protocol_rejection_valid = _is_expected_negative_rejection(
            response,
            control,
            stale_instance_id=str(encrypted_request.instance_id),
        )
        task_created, receipt_event_created, audit_created = (
            _observe_negative_side_effects(
                client,
                request_id,
                timeout_seconds=max(
                    config.event_timeout_seconds,
                    config.audit_timeout_seconds,
                ),
                poll_interval_seconds=config.poll_interval_seconds,
            )
        )
        evidence = NegativeCellEvidence(
            control=control,
            http_status=response.status_code,
            rejected=rejected,
            protocol_rejection_valid=protocol_rejection_valid,
            task_created=task_created,
            receipt_event_created=receipt_event_created,
            audit_created=audit_created,
        )
        if (
            protocol_rejection_valid
            and not task_created
            and not receipt_event_created
            and not audit_created
        ):
            return CellExecutionResult(status="expected_reject", evidence=evidence)
        return CellExecutionResult(
            status="unexpected_accept",
            failure_class="unexpected_acceptance",
            evidence=evidence,
        )

    if control == "cross_epoch_replay":
        return run_cross_epoch_operation_with_primary_restore(
            client,
            config,
            environment,
            submit_and_observe,
        )
    return submit_and_observe()


def _build_encrypted_request(
    config: QualityMatrixConfig,
    environment: MatrixEnvironment,
    identity: VerifiableProviderIdentity,
    prompt: str,
    request_id: str,
    max_output_tokens: int,
) -> VerifiableChatCompletionRequest:
    return build_verifiable_chat_request(
        instance=environment.instance,
        identity=identity,
        private_payload=VerifiablePrivateTaskPayload(
            input=[InputMessage(role="user", content=InputMessageContent(prompt))]
        ),
        generation=VerifiableGenerationParams(
            max_output_tokens=max_output_tokens,
            temperature=0.0,
            seed=config.seed,
            stream=False,
        ),
        request_id=request_id,
    )


def _is_expected_negative_rejection(
    response: httpx.Response,
    control: NegativeControl,
    *,
    stale_instance_id: str,
) -> bool:
    expected_status = 404 if control == "cross_epoch_replay" else 400
    expected_messages: dict[NegativeControl, str] = {
        "non_ingress_submission": (
            "Verifiable requests must be submitted to the placement ingress node API"
        ),
        "placement_digest_tamper": (
            "Encrypted request placement digest does not match the instance"
        ),
        "recipient_identity_tamper": (
            "Encrypted recipient does not match the ingress node's local delivery "
            "identity"
        ),
        "cross_epoch_replay": f"Instance {stale_instance_id} was not found",
    }
    if response.status_code != expected_status:
        return False
    try:
        payload = JSON_VALUE_ADAPTER.validate_json(response.content)
    except ValidationError:
        return False
    if not isinstance(payload, dict):
        return False
    error = payload.get("error")
    if not isinstance(error, dict):
        return False
    return (
        error.get("message") == expected_messages[control]
        and error.get("type") == HTTPStatus(expected_status).phrase
        and error.get("code") == expected_status
    )


def _observe_negative_side_effects(
    client: httpx.Client,
    request_id: str,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> tuple[bool, bool, bool]:
    """Observe the full bounded window; an early empty sample proves nothing."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        events_response = client.get("/events")
        events_response.raise_for_status()
        events = JSON_VALUE_ADAPTER.validate_json(events_response.content)
        task_created = _has_task_for_request(events, request_id)
        receipt_event_created = _has_raw_receipt_for_request(events, request_id)

        audit_response = client.get(f"/v1/verifiable/audit/{request_id}")
        audit_created = audit_response.status_code in (200, 409)
        if audit_response.status_code not in (200, 404, 409):
            audit_response.raise_for_status()
        if task_created or receipt_event_created or audit_created:
            return task_created, receipt_event_created, audit_created

        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            return False, False, False
        time.sleep(min(poll_interval_seconds, remaining_seconds))


def _assert_request_id_fresh(client: httpx.Client, request_id: str) -> None:
    response = client.get(f"/v1/verifiable/audit/{request_id}")
    if response.status_code == 404:
        return
    if response.status_code == 200:
        raise ValueError("The fresh request ID already has audit state")
    response.raise_for_status()


def _chat_result(response: httpx.Response) -> tuple[str, str]:
    payload = JSON_VALUE_ADAPTER.validate_json(response.content)
    if not isinstance(payload, dict):
        raise ValueError("Chat response is not an object")
    command_id = payload.get("id")
    choices = payload.get("choices")
    if not isinstance(command_id, str) or not isinstance(choices, list) or not choices:
        raise ValueError("Chat response has no command or choice")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ValueError("Chat response choice is not an object")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("Chat response has no assistant message")
    reasoning = message.get("reasoning_content", "")
    content = message.get("content", "")
    if reasoning is None:
        reasoning = ""
    if content is None:
        content = ""
    if not isinstance(reasoning, str) or not isinstance(content, str):
        raise ValueError("Chat response contains non-text output")
    output = json.dumps(
        {"reasoning_content": reasoning, "content": content},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return command_id, output


def _fetch_audit(
    client: httpx.Client,
    request_id: str,
    execution_id: CommandId,
    expected_ranks: int,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> VerifiableAuditResponse:
    deadline = time.monotonic() + timeout_seconds
    latest: VerifiableAuditResponse | None = None
    while True:
        response = client.get(f"/v1/verifiable/audit/{request_id}")
        if response.status_code == 200:
            latest = VerifiableAuditResponse.model_validate_json(response.content)
            if (
                len({receipt.device_rank for receipt in latest.receipts})
                >= expected_ranks
            ):
                return latest
        elif response.status_code != 404:
            response.raise_for_status()
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            return latest or VerifiableAuditResponse(
                request_id=request_id,
                execution_id=execution_id,
                expected_ranks=expected_ranks,
                receipts=[],
            )
        time.sleep(min(poll_interval_seconds, remaining_seconds))


def _fetch_cell_events(
    client: httpx.Client,
    command_id: str,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> JsonValue:
    deadline = time.monotonic() + timeout_seconds
    latest: JsonValue = []
    while True:
        response = client.get("/events")
        response.raise_for_status()
        latest = JSON_VALUE_ADAPTER.validate_json(response.content)
        if not isinstance(latest, list):
            raise ValueError("EXO /events did not return a list")
        if _has_task_for_command(latest, command_id) and _has_complete_token_output(
            latest, command_id
        ):
            return latest
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            # Never return a partial event snapshot: the caller validates
            # privacy/audit evidence and would otherwise have no terminal bit
            # in its pass predicate.  Keep the message independent of remote
            # event bodies because those are outside the redaction boundary.
            raise TimeoutError("Timed out waiting for complete task output evidence")
        time.sleep(min(poll_interval_seconds, remaining_seconds))


def _event_field(
    payload: dict[str, JsonValue], snake_name: str, camel_name: str
) -> JsonValue:
    return payload.get(snake_name, payload.get(camel_name))


def _json_contains_text(value: JsonValue, needle: str) -> bool:
    """Search decoded JSON string values without serialization escaping."""
    if isinstance(value, str):
        return needle in value
    if isinstance(value, list):
        return any(_json_contains_text(item, needle) for item in value)
    if isinstance(value, dict):
        return any(_json_contains_text(item, needle) for item in value.values())
    return False


def _raw_receipt_evidence(
    events: JsonValue,
    audit: VerifiableAuditResponse,
    *,
    request_id: str,
    execution_id: CommandId,
    expected_ranks: int,
) -> RawReceiptEvidence:
    """Cross-check raw receipt events against the aggregate audit projection."""
    errors: list[str] = []
    by_event_id: dict[str, VerifiableInputReceipt] = {}
    if not isinstance(events, list):
        return RawReceiptEvidence(
            event_ids=[],
            receipt_count=0,
            matches_aggregate=False,
            binding_errors=["raw_receipt_events_not_list"],
        )

    for untyped_event in events:
        if not isinstance(untyped_event, dict):
            continue
        prepared = untyped_event.get("VerifiableInputPrepared")
        if not isinstance(prepared, dict):
            continue
        receipt_payload = prepared.get("receipt")
        if not isinstance(receipt_payload, dict):
            continue
        if _event_field(receipt_payload, "request_id", "requestId") != request_id:
            continue
        event_id = _event_field(prepared, "event_id", "eventId")
        if not isinstance(event_id, str):
            errors.append("raw_receipt_event_id_missing")
            continue
        try:
            receipt = VerifiableInputReceipt.model_validate(receipt_payload)
        except ValidationError:
            errors.append("raw_receipt_invalid")
            continue
        previous = by_event_id.get(event_id)
        if previous is not None:
            if previous != receipt:
                errors.append("raw_receipt_event_id_conflict")
            continue
        by_event_id[event_id] = receipt

    raw_by_rank: dict[int, VerifiableInputReceipt] = {}
    for receipt in by_event_id.values():
        if receipt.execution_id != execution_id:
            errors.append("raw_receipt_execution_id_mismatch")
        previous = raw_by_rank.get(receipt.device_rank)
        if previous is not None:
            errors.append("raw_receipt_duplicate_rank")
        else:
            raw_by_rank[receipt.device_rank] = receipt

    if set(raw_by_rank) != set(range(expected_ranks)):
        errors.append("raw_receipt_rank_set_incomplete")
    aggregate_by_rank = {receipt.device_rank: receipt for receipt in audit.receipts}
    for rank in set(raw_by_rank) | set(aggregate_by_rank):
        if raw_by_rank.get(rank) != aggregate_by_rank.get(rank):
            errors.append("raw_receipt_aggregate_mismatch")
            break

    unique_errors = sorted(set(errors))
    return RawReceiptEvidence(
        event_ids=sorted(by_event_id),
        receipt_count=len(by_event_id),
        matches_aggregate=not unique_errors,
        binding_errors=unique_errors,
    )


def _text_generation_tasks(events: JsonValue) -> list[dict[str, JsonValue]]:
    if not isinstance(events, list):
        return []
    tasks: list[dict[str, JsonValue]] = []
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
        if isinstance(task, dict):
            tasks.append(task)
    return tasks


def _has_task_for_command(events: JsonValue, command_id: str) -> bool:
    return any(
        _event_field(task, "command_id", "commandId") == command_id
        for task in _text_generation_tasks(events)
    )


def _has_task_for_request(events: JsonValue, request_id: str) -> bool:
    for task in _text_generation_tasks(events):
        task_params = _event_field(task, "task_params", "taskParams")
        if not isinstance(task_params, dict):
            continue
        verifiable = task_params.get("verifiable")
        if isinstance(verifiable, dict) and (
            _event_field(verifiable, "request_id", "requestId") == request_id
        ):
            return True
    return False


def _has_raw_receipt_for_request(events: JsonValue, request_id: str) -> bool:
    if not isinstance(events, list):
        return False
    for untyped_event in events:
        if not isinstance(untyped_event, dict):
            continue
        prepared = untyped_event.get("VerifiableInputPrepared")
        if not isinstance(prepared, dict):
            continue
        receipt = prepared.get("receipt")
        if isinstance(receipt, dict) and (
            _event_field(receipt, "request_id", "requestId") == request_id
        ):
            return True
    return False


def _has_complete_token_output(events: JsonValue, command_id: str) -> bool:
    if not isinstance(events, list):
        return False
    task_ids = {
        task_id
        for task in _text_generation_tasks(events)
        if _event_field(task, "command_id", "commandId") == command_id
        and isinstance(task_id := _event_field(task, "task_id", "taskId"), str)
    }
    if len(task_ids) != 1:
        return False
    task_id = next(iter(task_ids))
    statuses: set[str] = set()
    seen_events: dict[str, str] = {}
    terminal_count = 0
    last_token_terminal = False
    token_count = 0
    for untyped_event in events:
        if not isinstance(untyped_event, dict):
            continue
        status_event = untyped_event.get("TaskStatusUpdated")
        if isinstance(status_event, dict) and (
            _event_field(status_event, "task_id", "taskId") == task_id
        ):
            status = _event_field(status_event, "task_status", "taskStatus")
            if isinstance(status, str):
                statuses.add(status)
        generated = untyped_event.get("ChunkGenerated")
        if not isinstance(generated, dict):
            continue
        if _event_field(generated, "command_id", "commandId") != command_id:
            continue
        chunk = generated.get("chunk")
        if not isinstance(chunk, dict):
            continue
        canonical_event = json.dumps(
            generated,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        event_id = generated.get("event_id")
        event_identity = (
            f"event-id:{event_id}" if isinstance(event_id, str) else canonical_event
        )
        previous_event = seen_events.get(event_identity)
        if previous_event is not None:
            if previous_event != canonical_event:
                raise ValueError(
                    "A clean-audit event id was reused for a conflicting payload"
                )
            continue
        seen_events[event_identity] = canonical_event
        if any(
            isinstance(chunk.get(chunk_type), dict)
            for chunk_type in ("ErrorChunk", "ToolCallChunk")
        ):
            raise ValueError("Clean audit received non-token terminal output")
        token_chunk = chunk.get("TokenChunk")
        if not isinstance(token_chunk, dict):
            continue
        token_count += 1
        last_token_terminal = (
            _event_field(token_chunk, "finish_reason", "finishReason") is not None
        )
        if last_token_terminal:
            terminal_count += 1
    if statuses.intersection({"Failed", "TimedOut", "Cancelled"}):
        raise ValueError("Clean-audit generation task terminated unsuccessfully")
    if terminal_count > 1:
        raise ValueError("Clean audit received multiple terminal token chunks")
    return (
        token_count > 0
        and terminal_count == 1
        and last_token_terminal
        and "Complete" in statuses
    )


def _task_created_evidence(
    events: JsonValue,
    command_id: str,
    request_id: str,
    instance_id: InstanceId,
) -> tuple[bool, bool]:
    matching = [
        task
        for task in _text_generation_tasks(events)
        if _event_field(task, "command_id", "commandId") == command_id
    ]
    if not matching:
        return False, False
    input_empty = True
    envelope_bound = True
    for task in matching:
        if _event_field(task, "instance_id", "instanceId") != str(instance_id):
            envelope_bound = False
        task_params = _event_field(task, "task_params", "taskParams")
        if not isinstance(task_params, dict):
            return False, False
        if task_params.get("input") != []:
            input_empty = False
        verifiable = task_params.get("verifiable")
        if not isinstance(verifiable, dict) or (
            _event_field(verifiable, "request_id", "requestId") != request_id
        ):
            envelope_bound = False
    return input_empty, envelope_bound


def _receipts_match_live_identities(
    audit: VerifiableAuditResponse, environment: MatrixEnvironment
) -> bool:
    for receipt in audit.receipts:
        identity = environment.identities.get(receipt.node_id)
        if identity is None or (
            receipt.reporting_provider_id != identity.provider_id
            or receipt.reporting_key_id != identity.key_id
        ):
            return False
    return len(audit.receipts) == environment.world_size


def _receipts_match_frozen_placement(
    audit: VerifiableAuditResponse, environment: MatrixEnvironment
) -> bool:
    ingress_identity = environment.identities[environment.ingress_node_id]
    frozen_digest = placement_digest(
        environment.instance,
        VerifiableRecipient(
            node_id=ingress_identity.node_id,
            provider_id=ingress_identity.provider_id,
            key_id=ingress_identity.key_id,
        ),
    )
    return len(audit.receipts) == environment.world_size and all(
        receipt.placement_digest == frozen_digest for receipt in audit.receipts
    )


def _validate_audit_bindings(
    audit: VerifiableAuditResponse,
    request: VerifiableChatCompletionRequest,
    environment: MatrixEnvironment,
    execution_id: CommandId,
) -> tuple[list[str], bool]:
    errors: list[str] = []
    if audit.request_id != request.request_id:
        errors.append("audit_request_id_mismatch")
    if audit.execution_id != execution_id:
        errors.append("audit_execution_id_mismatch")
    if audit.expected_ranks != environment.world_size:
        errors.append("audit_world_size_mismatch")
    expected_ciphertext_digest = _sha256_text(request.encrypted_input.ciphertext)
    expected_by_rank: dict[int, tuple[NodeId, PipelineShardMetadata]] = {}
    for (
        node_id,
        runner_id,
    ) in environment.instance.shard_assignments.node_to_runner.items():
        shard = environment.instance.shard_assignments.runner_to_shard[runner_id]
        if isinstance(shard, PipelineShardMetadata):
            expected_by_rank[shard.device_rank] = (node_id, shard)

    seen_ranks: set[int] = set()
    prompt_counts: set[int] = set()
    for receipt in audit.receipts:
        if receipt.device_rank in seen_ranks:
            errors.append("duplicate_receipt_rank")
        seen_ranks.add(receipt.device_rank)
        prompt_counts.add(receipt.prompt_token_count)
        expected = expected_by_rank.get(receipt.device_rank)
        if expected is None:
            errors.append("unexpected_receipt_rank")
        else:
            expected_node, expected_shard = expected
            if receipt.node_id != expected_node:
                errors.append("receipt_rank_node_mismatch")
            if (
                receipt.start_layer != expected_shard.start_layer
                or receipt.end_layer != expected_shard.end_layer
            ):
                errors.append("receipt_rank_layers_mismatch")
        if receipt.request_id != request.request_id:
            errors.append("receipt_request_id_mismatch")
        if receipt.execution_id != execution_id:
            errors.append("receipt_execution_id_mismatch")
        if receipt.instance_id != str(environment.instance.instance_id):
            errors.append("receipt_instance_id_mismatch")
        if receipt.placement_digest != request.placement_digest:
            errors.append("receipt_placement_digest_mismatch")
        if receipt.ciphertext_digest != expected_ciphertext_digest:
            errors.append("receipt_ciphertext_digest_mismatch")
        if receipt.world_size != environment.world_size:
            errors.append("receipt_world_size_mismatch")
        if (
            receipt.recipient_provider_id != request.recipient.provider_id
            or receipt.recipient_key_id != request.recipient.key_id
        ):
            errors.append("receipt_recipient_identity_mismatch")
        live_identity = environment.identities.get(receipt.node_id)
        if live_identity is None or (
            receipt.reporting_provider_id != live_identity.provider_id
            or receipt.reporting_key_id != live_identity.key_id
        ):
            errors.append("receipt_live_identity_mismatch")

    if seen_ranks != set(range(environment.world_size)):
        errors.append("receipt_rank_set_incomplete")
    if len(prompt_counts) != 1:
        errors.append("receipt_prompt_length_mismatch")
    ingress_only = all(
        (
            receipt.node_id == environment.ingress_node_id
            and receipt.device_rank == 0
            and receipt.private_input_accessed
            and receipt.input_source == "decrypted_envelope"
        )
        or (
            receipt.node_id != environment.ingress_node_id
            and receipt.device_rank != 0
            and not receipt.private_input_accessed
            and receipt.input_source == "shape_only_dummy"
        )
        for receipt in audit.receipts
    ) and bool(audit.receipts)
    if not ingress_only:
        errors.append("private_access_not_ingress_only")
    return sorted(set(errors)), ingress_only


def run_cross_epoch_operation_with_primary_restore(
    client: httpx.Client,
    config: QualityMatrixConfig,
    environment: MatrixEnvironment,
    operation: Callable[[], OperationResult],
) -> OperationResult:
    """Run one stale-envelope operation in epoch 2, then restore exact epoch 1."""
    replay_instance = _validated_replay_instance(config, environment)
    try:
        _transition_to_replay_epoch(
            client,
            config,
            environment,
            replay_instance,
        )
        result = operation()
    except Exception as operation_error:
        try:
            _restore_primary_epoch(
                client,
                config,
                environment,
                replay_instance,
            )
        except Exception as restoration_error:
            raise ExceptionGroup(
                "Cross-epoch operation and primary restoration both failed",
                [operation_error, restoration_error],
            ) from operation_error
        raise
    _restore_primary_epoch(
        client,
        config,
        environment,
        replay_instance,
    )
    return result


def _validated_replay_instance(
    config: QualityMatrixConfig,
    environment: MatrixEnvironment,
) -> Instance:
    if (
        not config.allow_placement_epoch_transition
        or config.replay_placement_preview_file is None
    ):
        raise ValueError(
            "Cross-epoch replay requires an exact second preview and explicit consent"
        )
    preview = PlacementPreview.model_validate_json(
        Path(config.replay_placement_preview_file).read_bytes()
    )
    if preview.instance is None or preview.error is not None:
        raise ValueError("The replay placement preview has no usable instance")
    next_instance = preview.instance
    if next_instance.instance_id == environment.instance.instance_id:
        raise ValueError("Replay epoch must use a fresh instance ID")
    if str(next_instance.shard_assignments.model_id) != config.model:
        raise ValueError("Replay epoch preview is for another model")
    if set(next_instance.shard_assignments.node_to_runner) != set(
        environment.identities
    ):
        raise ValueError("Replay epoch must use the same two live provider nodes")
    return next_instance


def _transition_to_replay_epoch(
    client: httpx.Client,
    config: QualityMatrixConfig,
    environment: MatrixEnvironment,
    next_instance: Instance,
) -> None:
    state = _fetch_state(client)
    live_primary = state.instances.get(environment.instance.instance_id)
    if live_primary is not None and live_primary != environment.instance:
        raise ValueError("The primary instance ID no longer matches frozen placement")
    live_replay = state.instances.get(next_instance.instance_id)
    if live_replay is not None and live_replay != next_instance:
        raise ValueError("The replay instance ID conflicts with the exact preview")

    if live_primary is not None:
        _delete_exact_instance(client, config, environment.instance)
    _ensure_exact_instance_ready(
        client,
        config,
        next_instance,
    )
    # Re-validate both APIs and both live identities in the new epoch before
    # submitting the stale old-epoch envelope.
    inspect_matrix_environment(
        client,
        config.model_copy(
            update={
                "instance_id": str(next_instance.instance_id),
                "placement_preview_file": None,
            }
        ),
    )


def _restore_primary_epoch(
    client: httpx.Client,
    config: QualityMatrixConfig,
    environment: MatrixEnvironment,
    replay_instance: Instance,
) -> None:
    _delete_exact_instance(client, config, replay_instance)
    _ensure_exact_instance_ready(client, config, environment.instance)
    inspect_matrix_environment(
        client,
        config.model_copy(
            update={
                "instance_id": str(environment.instance.instance_id),
                "placement_preview_file": None,
            }
        ),
    )


def _delete_exact_instance(
    client: httpx.Client,
    config: QualityMatrixConfig,
    expected_instance: Instance,
) -> None:
    live_instance = _fetch_state(client).instances.get(expected_instance.instance_id)
    if live_instance is None:
        return
    if live_instance != expected_instance:
        raise ValueError("Refusing to delete an instance that differs from the preview")
    delete_response = client.delete(f"/instance/{expected_instance.instance_id}")
    delete_response.raise_for_status()
    _wait_for_instance_absence(
        client,
        expected_instance.instance_id,
        timeout_seconds=config.placement_timeout_seconds,
        poll_interval_seconds=config.poll_interval_seconds,
    )


def _ensure_exact_instance_ready(
    client: httpx.Client,
    config: QualityMatrixConfig,
    expected_instance: Instance,
) -> None:
    live_instance = _fetch_state(client).instances.get(expected_instance.instance_id)
    if live_instance is None:
        create_response = client.post(
            "/instance",
            json={"instance": expected_instance.model_dump(mode="json", by_alias=True)},
        )
        create_response.raise_for_status()
    elif live_instance != expected_instance:
        raise ValueError("The live instance ID conflicts with the exact preview")
    _wait_for_instance(
        client,
        expected_instance,
        timeout_seconds=config.placement_timeout_seconds,
        poll_interval_seconds=config.poll_interval_seconds,
    )


def _wait_for_instance_absence(
    client: httpx.Client,
    instance_id: InstanceId,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        if instance_id not in _fetch_state(client).instances:
            return
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            raise TimeoutError("Timed out waiting for the old placement epoch to end")
        time.sleep(min(poll_interval_seconds, remaining_seconds))


def _private_prompt_for_cell(
    config: QualityMatrixConfig,
    cell: MatrixCell,
    prompts: dict[PromptClass, LoadedPrompt],
) -> LoadedPrompt | None:
    if cell.prompt_class is None:
        return None
    source = prompts[cell.prompt_class]
    if cell.plane != "clean_audit":
        return source
    canary_nonce = str(uuid4())
    canary_marker = f"[matrix-canary:{config.run_id}:{canary_nonce}]"
    canary = f"{source.plaintext}\n{canary_marker}"
    encoded = canary.encode("utf-8")
    actual_prompt_tokens: int | None = None
    if config.tokenizer_directory is not None:
        tokenizer_directory = Path(config.tokenizer_directory)
        actual_prompt_tokens = len(
            _single_user_chat_token_ids(
                _load_chat_tokenizer(tokenizer_directory),
                tokenizer_directory,
                canary,
                allow_qwen_fallback=(config.model == LOCKED_MODEL),
            )
        )
    return LoadedPrompt(
        plaintext=canary,
        evidence=PromptEvidence(
            prompt_class=cell.prompt_class,
            sha256=f"sha256:{hashlib.sha256(encoded).hexdigest()}",
            utf8_bytes=len(encoded),
            actual_prompt_tokens=actual_prompt_tokens,
        ),
        leak_sentinels=(canary_marker, canary_nonce),
    )


def _build_manifest(
    config: QualityMatrixConfig,
    environment: MatrixEnvironment,
    prompts: dict[PromptClass, LoadedPrompt],
    tokenizer: TokenizerEvidence,
    cells: list[MatrixCell],
    runtime_provenance: RuntimeProvenance | None,
    *,
    formal_execution: bool,
) -> MatrixManifest:
    branch, commit, source_dirty = _git_metadata()
    project_root = Path(__file__).resolve().parents[3]
    source_sha256 = _hash_source_tree(project_root)
    lockfile = project_root / "uv.lock"
    lockfile_sha256 = (
        _sha256_bytes(lockfile.read_bytes()) if lockfile.is_file() else None
    )
    placement_preview_sha256 = _configured_file_sha256(config.placement_preview_file)
    replay_placement_preview_sha256 = _configured_file_sha256(
        config.replay_placement_preview_file
    )
    runtime_provenance_sha256 = _runtime_provenance_sha256(runtime_provenance)
    ingress_identity = environment.identities[environment.ingress_node_id]
    committed_placement_digest = placement_digest(
        environment.instance,
        VerifiableRecipient(
            node_id=ingress_identity.node_id,
            provider_id=ingress_identity.provider_id,
            key_id=ingress_identity.key_id,
        ),
    )
    endpoint_by_url = {
        endpoint.api_url.rstrip("/"): endpoint for endpoint in config.nodes
    }
    nodes: list[NodeManifest] = []
    for node_id, identity in sorted(
        environment.identities.items(), key=lambda item: str(item[0])
    ):
        runner_id = environment.instance.shard_assignments.node_to_runner[node_id]
        shard = environment.instance.shard_assignments.runner_to_shard[runner_id]
        if not isinstance(shard, PipelineShardMetadata):
            raise ValueError("Manifest requires pure pipeline shards")
        api_url = (
            environment.ingress_url
            if node_id == environment.ingress_node_id
            else environment.downstream_url
        )
        endpoint = endpoint_by_url[api_url.rstrip("/")]
        nodes.append(
            NodeManifest(
                label=endpoint.label,
                api_url=api_url,
                node_id=node_id,
                provider_id=identity.provider_id,
                key_id=identity.key_id,
                public_key_sha256=_sha256_text(identity.public_key),
                backends=environment.backends.get(node_id, []),
                device_rank=shard.device_rank,
                world_size=shard.world_size,
                start_layer=shard.start_layer,
                end_layer=shard.end_layer,
            )
        )
    reproducible_payload: dict[str, object] = {
        "run_id": config.run_id,
        "formal_execution": formal_execution,
        "model": config.model,
        "instance_id": str(environment.instance.instance_id),
        "placement_digest": committed_placement_digest,
        "placement_preview_sha256": placement_preview_sha256,
        "replay_placement_preview_sha256": replay_placement_preview_sha256,
        "runtime_provenance_sha256": runtime_provenance_sha256,
        "seed": config.seed,
        "request_timeout_seconds": config.request_timeout_seconds,
        "event_timeout_seconds": config.event_timeout_seconds,
        "audit_timeout_seconds": config.audit_timeout_seconds,
        "placement_timeout_seconds": config.placement_timeout_seconds,
        "poll_interval_seconds": config.poll_interval_seconds,
        "source_sha256": source_sha256,
        "lockfile_sha256": lockfile_sha256,
        "tokenizer": tokenizer.model_dump(mode="json"),
        "prompts": [
            prompts[prompt_class].evidence.model_dump(mode="json")
            for prompt_class in PROMPT_CLASSES
        ],
        "nodes": [node.model_dump(mode="json") for node in nodes],
        "cells": [cell.model_dump(mode="json") for cell in cells],
    }
    reproducibility_sha256 = _sha256_bytes(
        json.dumps(
            reproducible_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    return MatrixManifest(
        run_id=config.run_id,
        created_at=_utc_now(),
        formal_execution=formal_execution,
        model=config.model,
        instance_id=str(environment.instance.instance_id),
        placement_digest=committed_placement_digest,
        placement_preview_sha256=placement_preview_sha256,
        replay_placement_preview_sha256=replay_placement_preview_sha256,
        runtime_provenance_sha256=runtime_provenance_sha256,
        seed=config.seed,
        request_timeout_seconds=config.request_timeout_seconds,
        event_timeout_seconds=config.event_timeout_seconds,
        audit_timeout_seconds=config.audit_timeout_seconds,
        placement_timeout_seconds=config.placement_timeout_seconds,
        poll_interval_seconds=config.poll_interval_seconds,
        quality_output_tokens=list(MAX_OUTPUT_TOKENS),
        clean_audit_output_tokens=16,
        branch=branch,
        commit=commit,
        source_dirty=source_dirty,
        source_sha256=source_sha256,
        lockfile_sha256=lockfile_sha256,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        tokenizer=tokenizer,
        prompts=[prompts[prompt_class].evidence for prompt_class in PROMPT_CLASSES],
        nodes=nodes,
        cells=cells,
        reproducibility_sha256=reproducibility_sha256,
    )


def _load_runtime_provenance(config: QualityMatrixConfig) -> RuntimeProvenance:
    if config.runtime_provenance_file is None:
        raise ValueError("The formal matrix requires --runtime-provenance")
    path = Path(config.runtime_provenance_file)
    if not path.is_file():
        raise ValueError("Runtime provenance file does not exist")
    return RuntimeProvenance.model_validate_json(path.read_bytes())


def _validate_formal_timing(config: QualityMatrixConfig) -> None:
    if config.request_timeout_seconds < 300.0:
        raise ValueError("Formal request timeout must be at least 300 seconds")
    if config.event_timeout_seconds < 30.0 or config.audit_timeout_seconds < 30.0:
        raise ValueError(
            "Formal evidence observation windows must be at least 30 seconds"
        )
    if config.placement_timeout_seconds < 180.0:
        raise ValueError("Formal placement timeout must be at least 180 seconds")
    if config.poll_interval_seconds > 1.0:
        raise ValueError("Formal evidence polling interval must not exceed one second")


def _require_clean_controller_source() -> tuple[str, str, str]:
    branch, commit, source_dirty = _git_metadata()
    if branch is None or commit is None or source_dirty is not False:
        raise ValueError("Formal execution requires a clean committed controller")
    project_root = Path(__file__).resolve().parents[3]
    return branch, commit, _hash_source_tree(project_root)


def _validate_runtime_provenance(
    provenance: RuntimeProvenance,
    config: QualityMatrixConfig,
    environment: MatrixEnvironment,
    *,
    controller_branch: str,
    controller_commit: str,
    controller_tree_sha256: str,
    tokenizer_evidence: TokenizerEvidence,
) -> None:
    node_by_api_url = {
        environment.ingress_url.rstrip("/"): environment.ingress_node_id,
        environment.downstream_url.rstrip("/"): environment.downstream_node_id,
    }
    endpoint_by_label = {endpoint.label.lower(): endpoint for endpoint in config.nodes}
    provenance_by_label = {node.label: node for node in provenance.nodes}
    for label, endpoint in endpoint_by_label.items():
        runtime = provenance_by_label[label]
        expected_node_id = node_by_api_url.get(endpoint.api_url.rstrip("/"))
        if expected_node_id is None or runtime.node_id != expected_node_id:
            raise ValueError("Runtime provenance node does not match its live API")
        identity = environment.identities[expected_node_id]
        if (
            runtime.provider_id != identity.provider_id
            or runtime.key_id != identity.key_id
        ):
            raise ValueError("Runtime provenance provider identity is not live")
        if set(runtime.backends) != set(environment.backends[expected_node_id]):
            raise ValueError("Runtime provenance backends differ from live state")
        endpoint_port = httpx.URL(endpoint.api_url).port
        if endpoint_port is None or runtime.api_port != endpoint_port:
            raise ValueError("Runtime provenance API port differs from the live URL")
        if runtime.source_branch != controller_branch:
            raise ValueError("Runtime and controller branches must match")
        if (
            runtime.source_commit != controller_commit
            or runtime.source_tree_sha256 != controller_tree_sha256
        ):
            raise ValueError("Runtime nodes must execute the controller source tree")
        if runtime.model_id != config.model:
            raise ValueError("Runtime provenance names a different model")
        if (
            runtime.tokenizer_metadata_sha256 != tokenizer_evidence.metadata_sha256
            or runtime.tokenizer_metadata_file_count != tokenizer_evidence.file_count
        ):
            raise ValueError("Runtime tokenizer metadata differs from preflight")
        entrypoint_text = " ".join(runtime.process_argv)
        if "exo" not in entrypoint_text:
            raise ValueError("Runtime process argv does not launch EXO")
        if (
            "--api-port" not in runtime.process_argv
            or str(runtime.api_port) not in runtime.process_argv
        ):
            raise ValueError("Runtime process argv does not bind the recorded API port")
        if not {"--no-batch", "--offline"}.issubset(runtime.process_argv):
            raise ValueError("Runtime process argv violates the frozen execution mode")


def _runtime_provenance_sha256(
    provenance: RuntimeProvenance | None,
) -> str | None:
    if provenance is None:
        return None
    canonical = json.dumps(
        provenance.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _sha256_bytes(canonical)


def _git_metadata() -> tuple[str | None, str | None, bool | None]:
    project_root = Path(__file__).resolve().parents[3]
    try:
        branch_result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None, None, None
    return (
        branch_result.stdout.strip() or None,
        commit_result.stdout.strip() or None,
        bool(status_result.stdout.strip()),
    )


def _configured_file_sha256(path_value: str | None) -> str | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if not path.is_file():
        raise ValueError("Configured reproducibility input does not exist")
    return _sha256_bytes(path.read_bytes())


def _hash_source_tree(project_root: Path) -> str:
    source_paths = sorted(
        [path for path in (project_root / "src").rglob("*.py") if path.is_file()]
        + [project_root / "pyproject.toml"]
    )
    digest = hashlib.sha256()
    for path in source_paths:
        if not path.is_file():
            continue
        relative = path.relative_to(project_root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return f"sha256:{digest.hexdigest()}"


def _attempt_count(records: list[AttemptRecord], cell_id: str) -> int:
    return len(
        {
            record.attempt_id
            for record in records
            if record.cell_id == cell_id and record.event == "started"
        }
    )


def _all_prior_cells_succeeded(
    current_cell: MatrixCell,
    cells: list[MatrixCell],
    records: list[AttemptRecord],
) -> bool:
    latest_finished: dict[str, AttemptRecord] = {}
    for record in records:
        if record.event != "finished":
            continue
        previous = latest_finished.get(record.cell_id)
        if previous is None or record.attempt_index >= previous.attempt_index:
            latest_finished[record.cell_id] = record
    for cell in cells:
        if cell.cell_id == current_cell.cell_id:
            return True
        latest = latest_finished.get(cell.cell_id)
        if latest is None or latest.status not in ("pass", "expected_reject"):
            return False
    raise ValueError("Current matrix cell is not part of the locked plan")


def _summarize(
    run_id: str,
    cells: list[MatrixCell],
    records: list[AttemptRecord],
    *,
    formal_execution: bool,
) -> MatrixSummary:
    allowed_cell_ids = {cell.cell_id for cell in cells}
    if any(
        record.run_id != run_id or record.cell_id not in allowed_cell_ids
        for record in records
    ):
        raise ValueError("Cannot summarize records outside the locked matrix run")
    scoped_records = [
        record for record in records if record.cell_id in allowed_cell_ids
    ]
    latest_finished: dict[str, AttemptRecord] = {}
    for record in scoped_records:
        if record.event == "finished":
            previous = latest_finished.get(record.cell_id)
            if previous is None or record.attempt_index >= previous.attempt_index:
                latest_finished[record.cell_id] = record
    statuses = Counter(
        record.status
        for record in latest_finished.values()
        if record.status is not None
    )
    attempt_statuses = Counter(
        record.status
        for record in scoped_records
        if record.event == "finished" and record.status is not None
    )
    attempts_by_cell = Counter(
        record.cell_id for record in scoped_records if record.event == "started"
    )
    terminal = sum(record.status != "infra_fail" for record in latest_finished.values())
    total = len(cells)
    passed = terminal == total and all(
        record.status in ("pass", "expected_reject")
        for record in latest_finished.values()
    )
    return MatrixSummary(
        run_id=run_id,
        updated_at=_utc_now(),
        formal_execution=formal_execution,
        total_cells=total,
        terminal_cells=terminal,
        missing_cells=total - terminal,
        status_counts={str(key): value for key, value in sorted(statuses.items())},
        attempt_count=sum(attempts_by_cell.values()),
        attempt_status_counts={
            str(key): value for key, value in sorted(attempt_statuses.items())
        },
        retried_cells=sorted(
            cell_id for cell_id, count in attempts_by_cell.items() if count > 1
        ),
        passed=passed,
        formal_passed=formal_execution and passed,
    )


def _atomic_write_model(path: Path, model: FrozenModel) -> None:
    temporary_path = path.with_name(f".{path.name}.{uuid4()}.tmp")
    file_descriptor = os.open(
        temporary_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
            _ = file.write(model.model_dump_json(indent=2))
            _ = file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _classify_exception(error: Exception) -> FailureClass:
    if isinstance(error, httpx.ConnectError):
        return "transport_connect"
    if isinstance(error, (httpx.TimeoutException, TimeoutError)):
        return "transport_timeout"
    if isinstance(error, httpx.HTTPStatusError):
        return "http_error"
    if isinstance(error, QualityCheckError):
        return "protocol_decode"
    if isinstance(error, (json.JSONDecodeError, ValueError)):
        return "protocol_decode"
    return "harness_internal"


def _cell_result_from_exception(error: Exception) -> CellExecutionResult:
    """Separate retryable infrastructure faults from terminal evidence faults."""
    client_protocol_error = (
        isinstance(error, httpx.HTTPStatusError)
        and 400 <= error.response.status_code < 500
    )
    if isinstance(error, (QualityCheckError, ValueError)) or client_protocol_error:
        return CellExecutionResult(
            status="scientific_fail",
            failure_class="protocol_evidence_invalid",
        )
    return CellExecutionResult(
        status="infra_fail",
        failure_class=_classify_exception(error),
    )


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _named_value(value: str, option: str) -> tuple[str, str]:
    name, separator, untyped_value = value.partition("=")
    if not separator or not name or not untyped_value:
        raise argparse.ArgumentTypeError(f"{option} must use NAME=VALUE")
    return name, untyped_value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the redacted two-provider VerifiableEXO Q/A/N matrix. "
            "Private prompts and token IDs are never written to results."
        )
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument(
        "--node",
        action="append",
        required=True,
        metavar="LABEL=API_URL",
        help="Repeat exactly twice; placement rank 0, not argument order, is ingress",
    )
    parser.add_argument("--model", default=LOCKED_MODEL)
    parser.add_argument("--instance-id")
    parser.add_argument(
        "--placement-preview",
        help="Saved exact PlacementPreview JSON to POST when the instance is absent",
    )
    parser.add_argument(
        "--replay-placement-preview",
        help="Fresh exact PlacementPreview used only by the final cross-epoch control",
    )
    parser.add_argument(
        "--allow-placement-epoch-transition",
        action="store_true",
        help="Consent to delete the old instance and create the replay-test epoch",
    )
    parser.add_argument(
        "--runtime-provenance",
        help="Redacted JSON record for the exact remote node runtimes",
    )
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument(
        "--prompt",
        action="append",
        required=True,
        metavar="CLASS=FILE",
        help="Repeat once for every locked prompt class",
    )
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--events-timeout", type=float, default=30.0)
    parser.add_argument("--audit-timeout", type=float, default=30.0)
    parser.add_argument("--placement-timeout", type=float, default=180.0)
    parser.add_argument("--poll-interval", type=float, default=0.2)
    parser.add_argument("--resume", action="store_true")
    return parser


def _config_from_args(args: argparse.Namespace) -> QualityMatrixConfig:
    untyped_nodes = cast(list[str], args.node)
    nodes = [
        MatrixNodeEndpoint(label=label, api_url=api_url)
        for label, api_url in (_named_value(value, "--node") for value in untyped_nodes)
    ]
    untyped_prompts = cast(list[str], args.prompt)
    prompt_files: list[MatrixPromptFile] = []
    for value in untyped_prompts:
        prompt_class, prompt_path = _named_value(value, "--prompt")
        if prompt_class not in PROMPT_CLASSES:
            raise argparse.ArgumentTypeError(
                f"Unknown prompt class; expected one of {', '.join(PROMPT_CLASSES)}"
            )
        typed_prompt_class = prompt_class
        prompt_files.append(
            MatrixPromptFile(
                prompt_class=typed_prompt_class,
                path=prompt_path,
                target_prompt_tokens=PROMPT_TOKEN_TARGETS[typed_prompt_class],
            )
        )
    return QualityMatrixConfig(
        base_url=cast(str, args.base_url),
        nodes=nodes,
        instance_id=cast(str | None, args.instance_id),
        placement_preview_file=cast(str | None, args.placement_preview),
        replay_placement_preview_file=cast(str | None, args.replay_placement_preview),
        runtime_provenance_file=cast(str | None, args.runtime_provenance),
        allow_placement_epoch_transition=cast(
            bool, args.allow_placement_epoch_transition
        ),
        tokenizer_directory=cast(str, args.tokenizer_dir),
        output_directory=cast(str, args.output_directory),
        run_id=cast(str, args.run_id),
        prompts=prompt_files,
        model=cast(str, args.model),
        seed=cast(int, args.seed),
        request_timeout_seconds=cast(float, args.timeout),
        event_timeout_seconds=cast(float, args.events_timeout),
        audit_timeout_seconds=cast(float, args.audit_timeout),
        placement_timeout_seconds=cast(float, args.placement_timeout),
        poll_interval_seconds=cast(float, args.poll_interval),
        resume=cast(bool, args.resume),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    try:
        config = _config_from_args(parser.parse_args(argv))
        with httpx.Client(
            base_url=config.base_url,
            timeout=config.request_timeout_seconds,
            follow_redirects=True,
        ) as client:
            summary = run_quality_matrix(client, config)
    except Exception as error:
        # Never print exception text: a remote body or local path can contain a
        # private prompt.  The detailed, redacted attempt class stays in JSONL.
        print(
            json.dumps(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "passed": False,
                    "error_type": type(error).__name__,
                    "failure_class": "preflight_failed",
                },
                separators=(",", ":"),
            )
        )
        return 2
    print(summary.model_dump_json(indent=2))
    if summary.passed:
        return 0
    if summary.missing_cells:
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
