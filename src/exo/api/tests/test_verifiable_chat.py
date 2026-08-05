"""Public API contract tests for placement-bound encrypted chat requests."""

from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from exo.api.main import API
from exo.api.types import VerifiableChatCompletionRequest, VerifiableRecipient
from exo.shared.models import model_cards
from exo.shared.models.model_cards import (
    ModelCard,
    ModelId,
    ModelTask,
    SamplingDefaults,
)
from exo.shared.types.backends import Backend
from exo.shared.types.commands import TextGeneration
from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.worker.instances import InstanceId, MlxRingInstance
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.verifiable.identity import (
    EXO_VERIFIABLE_KEY_PATH,
    local_delivery_identity,
)
from exo.verifiable.placement import placement_digest


def _encrypted_request() -> dict[str, object]:
    return {
        "protocol_version": "verifiable-exo-v1",
        "request_id": "request-00000000-0000-4000-8000-000000000001",
        "model": "mlx-community/Qwen3-0.6B-8bit",
        "instance_id": "instance-00000000-0000-4000-8000-000000000001",
        "placement_digest": "sha256:" + "a" * 64,
        "recipient": {
            "node_id": "node-ingress",
            "provider_id": "sha256:" + "b" * 64,
            "key_id": "delivery-key-v1",
        },
        "generation": {
            "max_output_tokens": 32,
            "temperature": 0.0,
            "seed": 42,
            "stream": False,
            "logprobs": True,
            "top_logprobs": 5,
        },
        "encrypted_input": {
            "scheme": "X25519-HKDF-SHA256-AES256GCM",
            "ephemeral_public_key": "ZXBoZW1lcmFsLXB1YmxpYy1rZXk=",
            "nonce": "MDEyMzQ1Njc4OWFi",
            "ciphertext": "Y2lwaGVydGV4dA==",
        },
    }


def test_verifiable_chat_request_rejects_plaintext_messages() -> None:
    """The encrypted endpoint must not accept a plaintext messages escape hatch."""
    payload = _encrypted_request()
    payload["messages"] = [{"role": "user", "content": "secret prompt"}]

    with pytest.raises(ValidationError):
        VerifiableChatCompletionRequest.model_validate(payload)


def test_verifiable_identity_exposes_only_public_delivery_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(EXO_VERIFIABLE_KEY_PATH, str(tmp_path / "delivery.key"))
    api = object.__new__(API)
    api.node_id = NodeId("node-ingress")

    identity = api.get_verifiable_identity()

    serialized = identity.model_dump_json()
    assert identity.node_id == api.node_id
    assert identity.provider_id.startswith("sha256:")
    assert identity.key_id == "delivery-key-v1"
    assert identity.public_key
    assert "private" not in serialized.lower()


def _pipeline_instance() -> tuple[MlxRingInstance, NodeId, NodeId]:
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
    return (
        MlxRingInstance(
            instance_id=InstanceId("instance-00000000-0000-4000-8000-000000000001"),
            shard_assignments=ShardAssignments(
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
            ),
            hosts_by_node={},
            ephemeral_port=50000,
        ),
        ingress_node,
        downstream_node,
    )


def test_verifiable_chat_rejects_non_ingress_recipient() -> None:
    """The encrypted recipient must be the node assigned the first layer."""
    instance, ingress_node, downstream_node = _pipeline_instance()
    api = object.__new__(API)
    api.app = FastAPI()
    api.node_id = ingress_node
    api.state = State(instances={instance.instance_id: instance})
    api._setup_exception_handlers()  # pyright: ignore[reportPrivateUsage]
    api.app.post("/v1/verifiable/chat/completions")(api.verifiable_chat_completions)

    payload = _encrypted_request()
    recipient = payload["recipient"]
    assert isinstance(recipient, dict)
    recipient["node_id"] = str(downstream_node)

    response = TestClient(api.app).post("/v1/verifiable/chat/completions", json=payload)

    assert response.status_code == 400
    assert "first pipeline shard" in response.json()["error"]["message"]


def test_verifiable_chat_rejects_wrong_placement_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The API must recompute placement binding instead of trusting the requester."""
    instance, ingress_node, _ = _pipeline_instance()
    monkeypatch.setenv(EXO_VERIFIABLE_KEY_PATH, str(tmp_path / "delivery.key"))
    identity = local_delivery_identity(ingress_node)
    api = object.__new__(API)
    api.app = FastAPI()
    api.node_id = ingress_node
    api.state = State(instances={instance.instance_id: instance})
    api._setup_exception_handlers()  # pyright: ignore[reportPrivateUsage]
    api.app.post("/v1/verifiable/chat/completions")(api.verifiable_chat_completions)

    payload = _encrypted_request()
    payload["recipient"] = {
        "node_id": str(ingress_node),
        "provider_id": identity.provider_id,
        "key_id": identity.key_id,
    }
    payload["placement_digest"] = "sha256:" + "0" * 64

    response = TestClient(api.app).post("/v1/verifiable/chat/completions", json=payload)

    assert response.status_code == 400
    assert "placement digest" in response.json()["error"]["message"]


async def test_verifiable_chat_must_be_submitted_to_ingress_node_api() -> None:
    instance, ingress_node, downstream_node = _pipeline_instance()
    api = object.__new__(API)
    api.node_id = downstream_node
    api.state = State(instances={instance.instance_id: instance})
    api.paused = False
    send_mock = AsyncMock()
    api._send = send_mock  # pyright: ignore[reportPrivateUsage]

    payload = _encrypted_request()
    recipient_payload = payload["recipient"]
    assert isinstance(recipient_payload, dict)
    recipient_payload["node_id"] = str(ingress_node)
    recipient = VerifiableRecipient.model_validate(recipient_payload)
    payload["placement_digest"] = placement_digest(instance, recipient)
    request = VerifiableChatCompletionRequest.model_validate(payload)

    with pytest.raises(HTTPException) as raised:
        await api.verifiable_chat_completions(request)

    assert raised.value.status_code == 400
    assert "ingress node API" in str(raised.value.detail)
    send_mock.assert_not_awaited()


@pytest.mark.parametrize(
    ("recipient_field", "substituted_value"),
    [
        ("provider_id", "sha256:" + "0" * 64),
        ("key_id", "delivery-key-v2"),
    ],
)
async def test_verifiable_chat_rejects_substituted_provider_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recipient_field: str,
    substituted_value: str,
) -> None:
    instance, ingress_node, _ = _pipeline_instance()
    monkeypatch.setenv(EXO_VERIFIABLE_KEY_PATH, str(tmp_path / "delivery.key"))
    identity = local_delivery_identity(ingress_node)
    local_recipient = VerifiableRecipient(
        node_id=identity.node_id,
        provider_id=identity.provider_id,
        key_id=identity.key_id,
    )
    substituted_recipient = local_recipient.model_copy(
        update={recipient_field: substituted_value}
    )

    api = object.__new__(API)
    api.node_id = ingress_node
    api.state = State(instances={instance.instance_id: instance})
    api.paused = False
    send_mock = AsyncMock()
    api._send = send_mock  # pyright: ignore[reportPrivateUsage]

    payload = _encrypted_request()
    payload["recipient"] = substituted_recipient.model_dump(mode="json")
    payload["placement_digest"] = placement_digest(instance, substituted_recipient)
    request = VerifiableChatCompletionRequest.model_validate(payload)

    with pytest.raises(HTTPException) as raised:
        await api.verifiable_chat_completions(request)

    assert raised.value.status_code == 400
    assert "local delivery identity" in str(raised.value.detail)
    send_mock.assert_not_awaited()


async def test_verifiable_chat_dispatches_only_encrypted_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid request becomes an instance-bound task without plaintext fields."""
    instance, ingress_node, _ = _pipeline_instance()
    monkeypatch.setenv(EXO_VERIFIABLE_KEY_PATH, str(tmp_path / "delivery.key"))
    identity = local_delivery_identity(ingress_node)
    api = object.__new__(API)
    api.node_id = ingress_node
    api.state = State(instances={instance.instance_id: instance})
    api.paused = False
    send_mock = AsyncMock()
    api._send = send_mock  # pyright: ignore[reportPrivateUsage]

    cached_card = next(
        iter(instance.shard_assignments.runner_to_shard.values())
    ).model_card.model_copy(
        update={
            "sampling_defaults": SamplingDefaults(
                repetition_penalty=1.1,
                presence_penalty=0.2,
                frequency_penalty=0.3,
            )
        }
    )
    monkeypatch.setitem(model_cards.card_cache.cc, cached_card.model_id, cached_card)

    payload = _encrypted_request()
    request_recipient = VerifiableRecipient(
        node_id=identity.node_id,
        provider_id=identity.provider_id,
        key_id=identity.key_id,
    )
    payload["recipient"] = request_recipient.model_dump(mode="json")
    payload["placement_digest"] = placement_digest(instance, request_recipient)
    request = VerifiableChatCompletionRequest.model_validate(payload)

    await api.verifiable_chat_completions(request)

    send_mock.assert_awaited_once()
    awaited = send_mock.await_args
    assert awaited is not None
    command = cast(TextGeneration, awaited.args[0])
    assert isinstance(command, TextGeneration)
    assert command.instance_id == instance.instance_id
    assert command.task_params.input == []
    assert command.task_params.verifiable is not None
    assert command.task_params.logprobs is True
    assert command.task_params.top_logprobs == 5
    assert command.task_params.repetition_penalty is None
    assert command.task_params.presence_penalty is None
    assert command.task_params.frequency_penalty is None
    serialized = command.model_dump_json()
    assert "secret prompt" not in serialized
    assert request.encrypted_input.ciphertext in serialized


def test_verifiable_chat_rejects_nonpositive_top_logprobs() -> None:
    payload = _encrypted_request()
    generation = payload["generation"]
    assert isinstance(generation, dict)
    generation["top_logprobs"] = 0

    with pytest.raises(ValidationError):
        VerifiableChatCompletionRequest.model_validate(payload)
