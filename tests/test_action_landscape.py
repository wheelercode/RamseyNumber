"""Test directional single-edge action-landscape analysis."""

import numpy as np
import pytest

from ramsey.RAction import analyze_actions
from ramsey.RActionLandscape import calculate_action_landscape
from ramsey.RColoring import RColoring
from ramsey.RGraph import RGraph
from ramsey.RProblem import RProblem
from ramsey.RState import RSearchState


@pytest.fixture(scope="module")
def graph() -> RGraph:
    return RGraph(RProblem.r55(n_vertices=6))


def all_red_state(graph: RGraph) -> RSearchState:
    return RSearchState(
        RColoring(
            graph,
            np.zeros(graph.number_of_edges, dtype=np.uint8),
        )
    )


def test_profiles_keep_red_and_blue_completion_deficits_separate(
    graph: RGraph,
) -> None:
    state = all_red_state(graph)
    landscape = calculate_action_landscape(state)

    # Every edge of K6 belongs to four K5s.  In an all-red state those
    # K5s have red deficit zero and blue deficit ten.
    assert np.all(landscape.red_profiles[:, 0] == 4)
    assert np.all(landscape.blue_profiles[:, 10] == 4)
    assert np.all(landscape.red_profiles[:, 1:] == 0)
    assert np.all(landscape.blue_profiles[:, :10] == 0)


def test_all_red_flip_reports_exact_directional_consequences(
    graph: RGraph,
) -> None:
    state = all_red_state(graph)
    landscape = calculate_action_landscape(state)

    edge = 0

    assert landscape.red_violations_destroyed[edge] == 4
    assert landscape.red_violations_created[edge] == 0
    assert landscape.blue_violations_destroyed[edge] == 0
    assert landscape.blue_violations_created[edge] == 0
    assert landscape.exact_rewards[edge] == 4

    # Four K5s move from red deficit zero to red deficit one.
    assert landscape.red_deficit_deltas[edge, 0] == -4
    assert landscape.red_deficit_deltas[edge, 1] == 4

    # The same K5s move from blue deficit ten to blue deficit nine.
    assert landscape.blue_deficit_deltas[edge, 10] == -4
    assert landscape.blue_deficit_deltas[edge, 9] == 4


def test_counterfactual_delta_matches_real_edge_flip(
    graph: RGraph,
) -> None:
    rng = np.random.default_rng(901)
    colors = rng.integers(
        0,
        2,
        size=graph.number_of_edges,
        dtype=np.uint8,
    )
    state = RSearchState(RColoring(graph, colors))
    landscape = calculate_action_landscape(state)

    edge = 7
    red_before = state.histogram.copy()
    blue_before = state.histogram[::-1].copy()

    state.apply_edge_flip(edge)

    assert np.array_equal(
        state.histogram - red_before,
        landscape.red_deficit_deltas[edge],
    )
    assert np.array_equal(
        state.histogram[::-1] - blue_before,
        landscape.blue_deficit_deltas[edge],
    )


def test_landscape_can_reuse_existing_action_analysis(
    graph: RGraph,
) -> None:
    state = all_red_state(graph)
    analysis = analyze_actions(state)
    landscape = calculate_action_landscape(
        state,
        analysis,
    )

    assert landscape.analysis is analysis


def test_landscape_rejects_stale_action_analysis(
    graph: RGraph,
) -> None:
    state = all_red_state(graph)
    analysis = analyze_actions(state)

    state.apply_edge_flip(0)

    with pytest.raises(ValueError):
        calculate_action_landscape(
            state,
            analysis,
        )
