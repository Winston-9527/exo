"""Canonical commitments for an immutable EXO instance placement."""

import hashlib
import json

from exo.shared.types.verifiable import VerifiableRecipient
from exo.shared.types.worker.instances import Instance


def placement_digest(instance: Instance, recipient: VerifiableRecipient) -> str:
    """Commit to the model placement and its authorized ingress identity."""
    assignments = instance.shard_assignments
    shards: list[dict[str, int | str]] = []
    for node_id, runner_id in assignments.node_to_runner.items():
        shard = assignments.runner_to_shard[runner_id]
        shards.append(
            {
                "node_id": str(node_id),
                "shard_type": type(shard).__name__,
                "device_rank": shard.device_rank,
                "world_size": shard.world_size,
                "start_layer": shard.start_layer,
                "end_layer": shard.end_layer,
                "n_layers": shard.n_layers,
            }
        )
    shards.sort(key=lambda shard: (shard["device_rank"], shard["node_id"]))
    canonical = json.dumps(
        {
            "protocol_version": "verifiable-exo-v1",
            "instance_id": str(instance.instance_id),
            "model_id": str(assignments.model_id),
            "recipient": {
                "node_id": str(recipient.node_id),
                "provider_id": recipient.provider_id,
                "key_id": recipient.key_id,
            },
            "shards": shards,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()
