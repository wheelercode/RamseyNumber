"""Test vertex color-balance transfer mechanics and predictions."""

import numpy as np
import pytest

from ramsey.RColoring import RColoring
from ramsey.RGraph import RGraph
from ramsey.RProblem import RProblem
from ramsey.RState import RSearchState
from ramsey.RVerification import verify_search_state
from ramsey.RVertexBalanceAction import (
    analyze_vertex_balance_transfers,
    apply_vertex_balance_transfer,
    vertex_balance_energy,
    vertex_blue_degrees,
    vertex_color_imbalances,
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


def test_vertex_imbalance_matches_blue_and_red_degrees(
    graph: RGraph,
) -> None:
    state = make_state(
        graph,
        seed=501,
    )

    blue_degrees = vertex_blue_degrees(
        state
    )
    imbalances = vertex_color_imbalances(
        state
    )

    assert np.array_equal(
        imbalances,
        2 * blue_degrees
        - (graph.problem.n_vertices - 1),
    )


def test_transfer_moves_only_donor_and_recipient_degrees(
    graph: RGraph,
) -> None:
    state = make_state(
        graph,
        seed=502,
    )

    analysis = analyze_vertex_balance_transfers(
        state
    )

    assert analysis.number_of_actions > 0

    action_index = 0
    transfer = analysis.transfer(
        action_index
    )

    before_degrees = vertex_blue_degrees(
        state
    )
    before_blue_edges = int(
        np.count_nonzero(
            state.colors == 1
        )
    )
    before_energy = vertex_balance_energy(
        state
    )

    actual_reward = apply_vertex_balance_transfer(
        state,
        transfer,
    )

    after_degrees = vertex_blue_degrees(
        state
    )

    expected_degrees = before_degrees.copy()
    expected_degrees[
        transfer.donor_vertex
    ] -= 1
    expected_degrees[
        transfer.recipient_vertex
    ] += 1

    assert np.array_equal(
        after_degrees,
        expected_degrees,
    )

    assert int(
        np.count_nonzero(
            state.colors == 1
        )
    ) == before_blue_edges

    assert (
        before_energy
        - vertex_balance_energy(state)
        == int(
            analysis.balance_rewards[
                action_index
            ]
        )
    )

    assert actual_reward == int(
        analysis.exact_rewards[
            action_index
        ]
    )

    verify_search_state(
        state
    ).require_consistent()


def test_all_analyzed_transfer_predictions_are_exact_for_sample(
    graph: RGraph,
) -> None:
    state = make_state(
        graph,
        seed=503,
    )

    analysis = analyze_vertex_balance_transfers(
        state
    )

    sample = np.linspace(
        0,
        analysis.number_of_actions - 1,
        num=min(
            12,
            analysis.number_of_actions,
        ),
        dtype=np.int32,
    )

    for action_index in sample:
        action_index = int(
            action_index
        )

        candidate = state.copy()
        before_histogram = candidate.histogram.copy()

        actual_reward = apply_vertex_balance_transfer(
            candidate,
            analysis.transfer(action_index),
        )

        assert actual_reward == int(
            analysis.exact_rewards[
                action_index
            ]
        )

        assert candidate.score == int(
            analysis.resulting_scores[
                action_index
            ]
        )

        assert np.array_equal(
            candidate.histogram
            - before_histogram,
            analysis.histogram_deltas[
                action_index
            ],
        )