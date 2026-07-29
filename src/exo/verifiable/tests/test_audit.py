"""Public, non-sensitive audit receipt aggregation tests."""

from typing import Literal

from exo.api.main import API
from exo.shared.apply import event_apply
from exo.shared.types.common import NodeId
from exo.shared.types.events import VerifiableInputPrepared
from exo.shared.types.state import State
from exo.shared.types.verifiable import VerifiableInputReceipt


def _receipt(
    rank: int, source: Literal["decrypted_envelope", "shape_only_dummy"]
) -> VerifiableInputReceipt:
    return VerifiableInputReceipt(
        request_id="request-audit-1",
        instance_id="instance-1",
        placement_digest="sha256:" + "a" * 64,
        node_id=NodeId(f"node-{rank}"),
        provider_id="sha256:" + "b" * 64,
        key_id="delivery-key-v1",
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


def test_audit_endpoint_returns_all_rank_receipts() -> None:
    receipts = (_receipt(0, "decrypted_envelope"), _receipt(1, "shape_only_dummy"))
    api = object.__new__(API)
    api.state = State(verifiable_receipts={"request-audit-1": receipts})

    response = api.get_verifiable_audit("request-audit-1")

    assert response.request_id == "request-audit-1"
    assert response.expected_ranks == 2
    assert len(response.receipts) == 2
