"""Compare action predictions with the untouched reference implementation."""

import numpy as np
import pytest

import RamseyAction as reference_action
import RamseyGraph as reference_graph
import RamseySearch as reference_search

from ramsey.RAction import (
    all_histogram_deltas,
    all_immediate_rewards,
    analyze_actions,
    edge_clique_profiles,
    immediate_rewards_for_edges,
    resulting_histograms,
)
from ramsey.RColoring import RColoring
from ramsey.RGraph import RGraph
from ramsey.RProblem import RProblem
from ramsey.RState import RSearchState


@pytest.fixture(scope="module")
def graph() -> RGraph:
    """
    Return a small symmetric graph for action tests.
    """
    return RGraph(RProblem.r55(n_vertices=10))


def make_states(
    graph: RGraph,
    seed: int,
):
    """
    Return equivalent new and reference states.
    """
    colors = reference_graph.random_coloring(
        graph.number_of_edges,
        np.random.default_rng(seed),
    )

    new_state = RSearchState(
        RColoring(
            graph,
            colors,
        )
    )

    reference_state = reference_search.create_search_state(
        colors,
        graph.subgraph_index(5).clique_edges,
    )

    return (
        new_state,
        reference_state,
    )


def test_profiles_match_reference(
    graph: RGraph,
) -> None:
    state, reference_state = make_states(
        graph,
        41,
    )

    edge_to_cliques = graph.subgraph_index(5).edge_to_cliques

    expected = reference_action.edge_kn_profiles(
        reference_state,
        edge_to_cliques,
    )

    actual = edge_clique_profiles(state)

    assert actual.dtype == expected.dtype

    assert np.array_equal(
        actual,
        expected,
    )


def test_histogram_deltas_and_rewards_match_reference(
    graph: RGraph,
) -> None:
    state, reference_state = make_states(
        graph,
        42,
    )

    edge_to_cliques = graph.subgraph_index(5).edge_to_cliques

    reference_profiles = reference_action.edge_kn_profiles(
        reference_state,
        edge_to_cliques,
    )

    expected_deltas = reference_action.all_action_histogram_deltas(
        reference_state.coloring,
        reference_profiles,
    )

    expected_rewards = reference_action.all_immediate_rewards(expected_deltas)

    profiles = edge_clique_profiles(state)

    actual_deltas = all_histogram_deltas(
        state,
        profiles,
    )

    actual_rewards = all_immediate_rewards(actual_deltas)

    assert np.array_equal(
        actual_deltas,
        expected_deltas,
    )

    assert np.array_equal(
        actual_rewards,
        expected_rewards,
    )


def test_complete_analysis_contains_exact_resulting_scores(
    graph: RGraph,
) -> None:
    state, _ = make_states(
        graph,
        43,
    )

    analysis = analyze_actions(state)

    assert analysis.number_of_actions == graph.number_of_edges

    assert analysis.applies_to(state)

    assert np.array_equal(
        analysis.resulting_scores,
        (state.score - analysis.immediate_rewards),
    )

    assert not (analysis.profiles.flags.writeable)

    assert not (analysis.histogram_deltas.flags.writeable)

    assert not (analysis.immediate_rewards.flags.writeable)

    assert not (analysis.resulting_scores.flags.writeable)


def test_every_prediction_matches_applied_flip(
    graph: RGraph,
) -> None:
    state, _ = make_states(
        graph,
        44,
    )

    analysis = analyze_actions(state)

    predicted_histograms = resulting_histograms(
        state,
        analysis,
    )

    for edge in range(state.number_of_edges):
        candidate = state.copy()

        reward = candidate.apply_edge_flip(edge)

        assert reward == int(analysis.immediate_rewards[edge])

        assert candidate.score == int(analysis.resulting_scores[edge])

        assert np.array_equal(
            candidate.histogram,
            predicted_histograms[edge],
        )


def test_selected_edge_rewards_match_full_analysis(
    graph: RGraph,
) -> None:
    state, _ = make_states(
        graph,
        45,
    )

    analysis = analyze_actions(state)

    selected_edges = np.asarray(
        [
            0,
            3,
            11,
            22,
            44,
        ],
        dtype=np.int32,
    )

    selected_rewards = immediate_rewards_for_edges(
        state,
        selected_edges,
    )

    assert np.array_equal(
        selected_rewards,
        analysis.immediate_rewards[selected_edges],
    )


def test_analysis_becomes_stale_after_state_mutation(
    graph: RGraph,
) -> None:
    state, _ = make_states(
        graph,
        46,
    )

    analysis = analyze_actions(state)

    state.apply_edge_flip(0)

    assert not analysis.applies_to(state)

    with pytest.raises(
        ValueError,
        match="earlier state version",
    ):
        resulting_histograms(
            state,
            analysis,
        )


def test_analysis_does_not_apply_to_different_state(
    graph: RGraph,
) -> None:
    state, _ = make_states(
        graph,
        47,
    )

    other_state = state.copy()

    analysis = analyze_actions(state)

    assert state.version == other_state.version

    assert not analysis.applies_to(other_state)
