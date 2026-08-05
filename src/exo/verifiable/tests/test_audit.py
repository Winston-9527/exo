"""Public, non-sensitive audit receipt aggregation tests."""

from typing import Literal

import pytest
from fastapi import HTTPException

from exo.api.main import API
from exo.shared.apply import event_apply
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.events import VerifiableInputPrepared
from exo.shared.types.state import State
from exo.shared.types.verifiable import VerifiableInputReceipt


def _receipt(
    rank: int,
    source: Literal["decrypted_envelope", "shape_only_dummy"],
    *,
    execution_id: str = "execution-a",
) -> VerifiableInputReceipt:
    return VerifiableInputReceipt(
        request_id="request-audit-1",
        execution_id=CommandId(execution_id),
        instance_id="instance-1",
        placement_digest="sha256:" + "a" * 64,
        node_id=NodeId(f"node-{rank}"),
        reporting_provider_id="sha256:" + str(rank) * 64,
        reporting_key_id="delivery-key-v1",
        recipient_provider_id="sha256:" + "b" * 64,
        recipient_key_id="delivery-key-v1",
        device_rank=rank,
        world_size=2,
        start_layer=rank * 14,
        end_layer=(rank + 1) * 14,
        input_source=source,
        private_input_accessed=rank == 0,
        prompt_token_count=19,
        ciphertext_digest="sha256:" + "c" * 64,
    )


def test_receipts_are_aggregated_by_request_without_sensitive_input() -> None:
    state = State()
    state = event_apply(
        VerifiableInputPrepared(receipt=_receipt(0, "decrypted_envelope")), state
    )
    state = event_apply(
        VerifiableInputPrepared(receipt=_receipt(1, "shape_only_dummy")), state
    )

    receipts = state.verifiable_receipts["request-audit-1"]
    assert [receipt.device_rank for receipt in receipts] == [0, 1]
    assert [receipt.private_input_accessed for receipt in receipts] == [True, False]
    serialized = "".join(receipt.model_dump_json() for receipt in receipts)
    assert "secret prompt" not in serialized
    assert "input_ids" not in serialized


def test_receipts_preserve_executions_and_deduplicate_only_within_execution() -> None:
    state = State()
    state = event_apply(
        VerifiableInputPrepared(
            receipt=_receipt(0, "decrypted_envelope", execution_id="execution-a")
        ),
        state,
    )
    state = event_apply(
        VerifiableInputPrepared(
            receipt=_receipt(0, "decrypted_envelope", execution_id="execution-b")
        ),
        state,
    )
    replacement = _receipt(
        0, "decrypted_envelope", execution_id="execution-a"
    ).model_copy(update={"prompt_token_count": 20})
    state = event_apply(VerifiableInputPrepared(receipt=replacement), state)

    receipts = state.verifiable_receipts["request-audit-1"]

    assert [(receipt.execution_id, receipt.device_rank) for receipt in receipts] == [
        (CommandId("execution-a"), 0),
        (CommandId("execution-b"), 0),
    ]
    assert receipts[0].prompt_token_count == 20


def test_audit_endpoint_returns_all_rank_receipts() -> None:
    receipts = (_receipt(0, "decrypted_envelope"), _receipt(1, "shape_only_dummy"))
    api = object.__new__(API)
    api.state = State(verifiable_receipts={"request-audit-1": receipts})

    response = api.get_verifiable_audit("request-audit-1")

    assert response.request_id == "request-audit-1"
    assert response.execution_id == CommandId("execution-a")
    assert response.expected_ranks == 2
    assert len(response.receipts) == 2


def test_audit_endpoint_rejects_request_id_with_multiple_executions() -> None:
    receipts = (
        _receipt(0, "decrypted_envelope", execution_id="execution-a"),
        _receipt(1, "shape_only_dummy", execution_id="execution-b"),
    )
    api = object.__new__(API)
    api.state = State(verifiable_receipts={"request-audit-1": receipts})

    with pytest.raises(HTTPException) as raised:
        api.get_verifiable_audit("request-audit-1")

    assert raised.value.status_code == 409
    assert "multiple executions" in str(raised.value.detail)


@pytest.mark.parametrize(
    ("field", "conflicting_value"),
    [
        ("request_id", "request-conflict"),
        ("instance_id", "instance-conflict"),
        ("placement_digest", "sha256:" + "d" * 64),
        ("ciphertext_digest", "sha256:" + "e" * 64),
        ("world_size", 3),
        ("recipient_provider_id", "sha256:" + "f" * 64),
        ("recipient_key_id", "delivery-key-conflict"),
        ("prompt_token_count", 20),
    ],
)
def test_audit_endpoint_rejects_conflicting_execution_receipts(
    field: str, conflicting_value: str | int
) -> None:
    conflicting_receipt = _receipt(1, "shape_only_dummy").model_copy(
        update={field: conflicting_value}
    )
    receipts = (_receipt(0, "decrypted_envelope"), conflicting_receipt)
    api = object.__new__(API)
    api.state = State(verifiable_receipts={"request-audit-1": receipts})

    with pytest.raises(HTTPException) as raised:
        api.get_verifiable_audit("request-audit-1")

    assert raised.value.status_code == 409
    assert field in str(raised.value.detail)


def test_audit_endpoint_rejects_receipt_stored_under_another_request_id() -> None:
    api = object.__new__(API)
    api.state = State(
        verifiable_receipts={"request-route": (_receipt(0, "decrypted_envelope"),)}
    )

    with pytest.raises(HTTPException) as raised:
        api.get_verifiable_audit("request-route")

    assert raised.value.status_code == 409
    assert "request_id" in str(raised.value.detail)
