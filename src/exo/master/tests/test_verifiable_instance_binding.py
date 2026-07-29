"""Master-side tests for placement-bound verifiable requests."""

import pytest

from exo.master.main import (
    _select_text_generation_instance_id,  # pyright: ignore[reportPrivateUsage]
)
from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.commands import TextGeneration
from exo.shared.types.common import ModelId, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.instances import InstanceId, MlxRingInstance
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata


def _instance(instance_id: str, node_id: str) -> MlxRingInstance:
    model = ModelCard(
        model_id=ModelId("mlx-community/Qwen3-0.6B-8bit"),
        storage_size=Memory.from_bytes(1),
        n_layers=28,
        hidden_size=1024,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxCuda],
    )
    runner_id = RunnerId(f"runner-{instance_id}")
    node = NodeId(node_id)
    return MlxRingInstance(
        instance_id=InstanceId(instance_id),
        shard_assignments=ShardAssignments(
            model_id=model.model_id,
            runner_to_shard={
                runner_id: PipelineShardMetadata(
                    model_card=model,
                    device_rank=0,
                    world_size=1,
                    start_layer=0,
                    end_layer=28,
                    n_layers=28,
                )
            },
            node_to_runner={node: runner_id},
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )


def test_master_honors_exact_instance_binding() -> None:
    """Load balancing must not move a placement-bound request."""
    first = _instance("instance-first", "node-first")
    bound = _instance("instance-bound", "node-bound")
    state = State(instances={first.instance_id: first, bound.instance_id: bound})
    command = TextGeneration(
        instance_id=bound.instance_id,
        task_params=TextGenerationTaskParams(
            model=bound.shard_assignments.model_id, input=[]
        ),
    )

    selected = _select_text_generation_instance_id(state, command)

    assert selected == bound.instance_id


def test_master_rejects_unknown_exact_instance_binding() -> None:
    instance = _instance("instance-existing", "node-existing")
    state = State(instances={instance.instance_id: instance})
    command = TextGeneration(
        instance_id=InstanceId("instance-missing"),
        task_params=TextGenerationTaskParams(
            model=instance.shard_assignments.model_id, input=[]
        ),
    )

    with pytest.raises(ValueError, match="requested instance"):
        _select_text_generation_instance_id(state, command)
