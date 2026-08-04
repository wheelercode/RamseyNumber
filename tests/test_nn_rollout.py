"""Test neural rollout collection and advantage calculations."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")


from ramsey.RColoring import RColoring
from ramsey.REnvironment import REnvironment
from ramsey.REnvironmentConfig import (
    REnvironmentConfig,
)
from ramsey.REnvironmentMemory import (
    RNullMemory,
)
from ramsey.RGraph import RGraph
from ramsey.RObjective import (
    RDangerObjective,
    RMonochromaticObjective,
    danger_energy,
)
from ramsey.RProblem import RProblem
from ramsey.RState import RSearchState
from ramsey.nn.RModel import (
    RModelConfig,
    RPairPolicyValueNetwork,
)
from ramsey.nn.RRollout import (
    RRolloutConfig,
    RRolloutReward,
    calculate_advantages,
    collect_rollout,
)


def make_rollout_components(
    n_vertices: int = 10,
    max_steps: int = 4,
    danger: bool = False,
):
    graph = RGraph(RProblem.r55(n_vertices=n_vertices))

    if danger:
        objective = RDangerObjective(0.25)
    else:
        objective = RMonochromaticObjective()

    environment = REnvironment(
        graph=graph,
        objective=objective,
        memory=RNullMemory(),
        config=REnvironmentConfig(max_steps=max_steps),
    )

    coloring = RColoring(
        graph,
        np.zeros(
            graph.number_of_edges,
            dtype=np.uint8,
        ),
    )

    network = RPairPolicyValueNetwork(
        graph,
        RModelConfig(
            hidden_size=8,
            number_of_layers=1,
        ),
    )

    return (
        graph,
        environment,
        coloring,
        network,
    )


def test_calculate_advantages_matches_manual_result() -> None:
    (
        advantages,
        returns,
    ) = calculate_advantages(
        rewards=np.asarray(
            [1.0, 2.0],
            dtype=np.float32,
        ),
        values=np.asarray(
            [0.5, 0.25],
            dtype=np.float32,
        ),
        terminated=np.asarray(
            [False, True],
            dtype=np.bool_,
        ),
        last_value=99.0,
        discount=0.9,
        gae_lambda=0.8,
    )

    assert np.allclose(
        advantages,
        [
            1.985,
            1.75,
        ],
    )

    assert np.allclose(
        returns,
        [
            2.485,
            2.0,
        ],
    )

    assert advantages.dtype == np.float32

    assert returns.dtype == np.float32


@pytest.mark.parametrize(
    (
        "values",
        "exception_type",
    ),
    [
        (
            {"rollout_steps": 0},
            ValueError,
        ),
        (
            {"rollout_steps": 2.5},
            TypeError,
        ),
        (
            {"discount": -0.1},
            ValueError,
        ),
        (
            {"gae_lambda": 1.1},
            ValueError,
        ),
        (
            {"reward_scale": 0.0},
            ValueError,
        ),
        (
            {"reward_source": "unknown"},
            ValueError,
        ),
        (
            {"normalize_advantages": 1},
            TypeError,
        ),
    ],
)
def test_rollout_config_validation(
    values,
    exception_type,
) -> None:
    with pytest.raises(exception_type):
        RRolloutConfig(**values)


def test_collect_rollout_produces_cpu_batch_and_exact_rewards() -> None:
    torch.manual_seed(411)

    (
        graph,
        environment,
        coloring,
        network,
    ) = make_rollout_components()

    rollout = collect_rollout(
        network,
        environment,
        coloring,
        device="cpu",
        config=RRolloutConfig(
            rollout_steps=10,
            reward_scale=10.0,
            normalize_advantages=True,
        ),
    )

    assert rollout.number_of_steps == 4

    assert rollout.truncated
    assert not rollout.terminated

    assert rollout.pair_inputs.shape == (
        4,
        10,
        10,
        3,
    )

    assert rollout.available_masks.shape == (
        4,
        graph.number_of_edges,
    )

    assert rollout.actions.shape == (4,)

    assert rollout.pair_inputs.device.type == "cpu"

    assert rollout.available_masks.dtype == torch.bool

    selected_availability = rollout.available_masks.gather(
        1,
        rollout.actions[:, None],
    )

    assert torch.all(selected_availability)

    assert np.isclose(
        rollout.total_scaled_reward,
        (rollout.initial_score - rollout.final_score) / 10.0,
    )

    assert rollout.best_score <= rollout.initial_score

    assert rollout.best_coloring.graph.problem == graph.problem

    assert np.isclose(
        float(rollout.advantages.mean()),
        0.0,
        atol=1.0e-6,
    )


def test_collect_rollout_can_use_objective_reward() -> None:
    torch.manual_seed(412)

    (
        _,
        environment,
        coloring,
        network,
    ) = make_rollout_components(danger=True)

    initial_energy = danger_energy(
        RSearchState(coloring).histogram,
        decay=0.25,
    )

    rollout = collect_rollout(
        network,
        environment,
        coloring,
        device="cpu",
        config=RRolloutConfig(
            rollout_steps=3,
            reward_scale=5.0,
            reward_source=RRolloutReward.OBJECTIVE,
        ),
    )

    final_energy = danger_energy(
        environment.state.histogram,
        decay=0.25,
    )

    assert np.isclose(
        rollout.total_scaled_reward,
        (initial_energy - final_energy) / 5.0,
    )


def test_terminal_seed_produces_empty_rollout() -> None:
    (
        graph,
        environment,
        _,
        network,
    ) = make_rollout_components(
        n_vertices=5,
        max_steps=4,
    )

    colors = np.zeros(
        graph.number_of_edges,
        dtype=np.uint8,
    )

    colors[0] = 1

    coloring = RColoring(
        graph,
        colors,
    )

    rollout = collect_rollout(
        network,
        environment,
        coloring,
        device="cpu",
        config=RRolloutConfig(rollout_steps=4),
    )

    assert rollout.number_of_steps == 0

    assert rollout.terminated
    assert not rollout.truncated

    assert rollout.pair_inputs.shape == (
        0,
        5,
        5,
        3,
    )

    assert rollout.available_masks.shape == (
        0,
        graph.number_of_edges,
    )

    assert rollout.rewards.numel() == 0
