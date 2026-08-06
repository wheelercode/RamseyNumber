"""Test greedy selection among vertex-balance transfers."""

import numpy as np
import pytest

from ramsey.RColoring import RColoring
from ramsey.RGraph import RGraph
from ramsey.RObjective import all_danger_rewards
from ramsey.RProblem import RProblem
from ramsey.RState import RSearchState
from ramsey.RVertexBalanceAction import (
    analyze_vertex_balance_transfers,
    apply_vertex_balance_transfer,
)
from ramsey.RVertexBalancePolicy import (
    RVertexBalancePolicyConfig,
    select_greedy_vertex_balance_action,
)


@pytest.fixture(scope="module")
def graph() -> RGraph:
    return RGraph(
        RProblem.r55(
            n_vertices=10,
        )
    )


def make_state(
    graph: RGraph,
    seed: int,
) -> RSearchState:
    colors = np.random.default_rng(
        seed
    ).integers(
        0,
        2,
        size=graph.number_of_edges,
        dtype=np.uint8,
    )

    return RSearchState(
        RColoring(
            graph,
            colors,
        )
    )


def test_balance_first_policy_maximizes_balance_then_danger(
    graph: RGraph,
) -> None:
    state = make_state(
        graph,
        seed=511,
    )

    analysis = analyze_vertex_balance_transfers(
        state
    )

    danger_rewards = all_danger_rewards(
        analysis.histogram_deltas,
        decay=0.25,
    )

    maximum_balance = np.max(
        analysis.balance_rewards
    )

    eligible = (
        analysis.balance_rewards
        == maximum_balance
    )

    expected_danger = np.max(
        danger_rewards[eligible]
    )

    selection = select_greedy_vertex_balance_action(
        state,
        RVertexBalancePolicyConfig(
            use_danger_reward=True,
            danger_decay=0.25,
            prioritize_balance=True,
        ),
    )

    assert (
        selection.balance_reward
        == int(maximum_balance)
    )
    assert np.isclose(
        selection.objective_reward,
        expected_danger,
    )


def test_ramsey_first_policy_chooses_best_exact_reward(
    graph: RGraph,
) -> None:
    state = make_state(
        graph,
        seed=512,
    )

    analysis = analyze_vertex_balance_transfers(
        state
    )

    selection = select_greedy_vertex_balance_action(
        state,
        RVertexBalancePolicyConfig(
            use_danger_reward=False,
            prioritize_balance=False,
        ),
    )

    assert (
        selection.exact_reward
        == int(
            np.max(
                analysis.exact_rewards
            )
        )
    )

    before_score = state.score

    actual_reward = apply_vertex_balance_transfer(
        state,
        selection.transfer,
    )

    assert actual_reward == selection.exact_reward
    assert state.score == selection.resulting_score
    assert (
        before_score - state.score
        == selection.exact_reward
    )