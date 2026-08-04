"""Test interchangeable random and greedy action policies."""

import numpy as np
import pytest

from ramsey.RColoring import RColoring
from ramsey.REnvironment import REnvironment
from ramsey.REnvironmentConfig import (
    REnvironmentConfig,
    RTabuMemoryConfig,
)
from ramsey.REnvironmentMemory import (
    RTabuMemory,
)
from ramsey.RGraph import RGraph
from ramsey.RObjective import (
    RDangerObjective,
)
from ramsey.RPolicy import (
    RGreedyPolicy,
    RRandomPolicy,
)
from ramsey.RProblem import RProblem


@pytest.fixture(scope="module")
def graph() -> RGraph:
    return RGraph(RProblem.r55(n_vertices=10))


def make_environment(
    graph: RGraph,
    seed: int,
) -> REnvironment:
    colors = np.random.default_rng(seed).integers(
        0,
        2,
        size=graph.number_of_edges,
        dtype=np.uint8,
    )

    environment = REnvironment(
        graph=graph,
        objective=RDangerObjective(decay=0.25),
        memory=RTabuMemory(
            graph.number_of_edges,
            RTabuMemoryConfig(
                edge_tenure=5,
                visited_state_window=50,
            ),
        ),
        config=REnvironmentConfig(max_steps=100),
    )

    environment.reset(
        RColoring(
            graph,
            colors,
        )
    )

    return environment


def test_random_policy_selects_an_available_action(
    graph: RGraph,
) -> None:
    environment = make_environment(
        graph,
        seed=91,
    )

    policy = RRandomPolicy(np.random.default_rng(92))

    available_mask = environment.available_action_mask_fast()

    edge = policy.select_action(environment)

    assert available_mask[edge]
    assert not policy.requires_full_analysis


def test_policy_selection_does_not_mutate_state(
    graph: RGraph,
) -> None:
    environment = make_environment(
        graph,
        seed=93,
    )

    policy = RRandomPolicy(np.random.default_rng(94))

    version = environment.state.version

    colors = environment.state.colors.copy()

    policy.select_action(environment)

    assert environment.state.version == version

    assert np.array_equal(
        environment.state.colors,
        colors,
    )


@pytest.mark.parametrize(
    "use_objective_reward",
    [False, True],
)
def test_greedy_policy_selects_a_maximum_available_reward(
    graph: RGraph,
    use_objective_reward: bool,
) -> None:
    environment = make_environment(
        graph,
        seed=95,
    )

    policy = RGreedyPolicy(
        np.random.default_rng(96),
        use_objective_reward=use_objective_reward,
    )

    analysis = environment.analyze_actions()

    if use_objective_reward:
        rewards = analysis.objective_rewards
    else:
        rewards = analysis.action_analysis.immediate_rewards

    expected_reward = np.max(rewards[analysis.available_mask])

    edge = policy.select_action(environment)

    assert analysis.available_mask[edge]

    assert rewards[edge] == expected_reward

    assert policy.requires_full_analysis


def test_policy_random_tie_breaking_is_reproducible(
    graph: RGraph,
) -> None:
    first_environment = make_environment(
        graph,
        seed=97,
    )

    second_environment = make_environment(
        graph,
        seed=97,
    )

    first = RGreedyPolicy(np.random.default_rng(98))

    second = RGreedyPolicy(np.random.default_rng(98))

    assert first.select_action(first_environment) == second.select_action(
        second_environment
    )


def test_greedy_policy_validates_reward_mode() -> None:
    with pytest.raises(
        TypeError,
        match="use_objective_reward",
    ):
        RGreedyPolicy(
            np.random.default_rng(99),
            use_objective_reward=1,
        )
