"""Tests for two-edge Ramsey causal analysis."""

from __future__ import annotations

import numpy as np
import pytest

from ramsey import (
    RColoring,
    RGraph,
    RProblem,
    RSearchState,
)
from ramsey.RAction import analyze_actions
from ramsey.RTwoEdgeFlipCausalAnalysis import (
    analyze_two_edge_flip_causality,
)


def make_state(
    *,
    n_vertices: int = 7,
    seed: int = 12345,
) -> RSearchState:
    """Return a deterministic small symmetric Ramsey search state."""
    graph = RGraph(
        RProblem.r55(
            n_vertices=n_vertices,
        )
    )
    rng = np.random.default_rng(seed)
    colors = rng.integers(
        0,
        2,
        size=graph.number_of_edges,
        dtype=np.uint8,
    )
    coloring = RColoring(
        graph=graph,
        colors=colors,
    )
    return RSearchState(coloring)


def test_analysis_does_not_mutate_supplied_state() -> None:
    state = make_state()
    colors_before = state.colors.copy()
    counts_before = state.color_one_counts.copy()
    histogram_before = state.histogram.copy()
    score_before = state.score
    version_before = state.version

    analyze_two_edge_flip_causality(
        state,
        0,
        1,
    )

    assert np.array_equal(state.colors, colors_before)
    assert np.array_equal(
        state.color_one_counts,
        counts_before,
    )
    assert np.array_equal(
        state.histogram,
        histogram_before,
    )
    assert state.score == score_before
    assert state.version == version_before


def test_exact_reward_matches_direct_two_flip_result() -> None:
    state = make_state()
    analysis = analyze_two_edge_flip_causality(
        state,
        0,
        1,
    )

    direct = state.copy()
    direct.apply_edge_flip(0)
    direct.apply_edge_flip(1)

    assert analysis.score_before == state.score
    assert analysis.score_after == direct.score
    assert analysis.exact_reward == state.score - direct.score


def test_individual_and_interaction_rewards_are_exact() -> None:
    state = make_state()
    rewards = analyze_actions(
        state
    ).immediate_rewards

    analysis = analyze_two_edge_flip_causality(
        state,
        0,
        1,
    )

    expected_individual = (
        int(rewards[0]),
        int(rewards[1]),
    )

    assert analysis.individual_rewards == expected_individual
    assert analysis.interaction_reward == (
        analysis.exact_reward
        - expected_individual[0]
        - expected_individual[1]
    )


def test_event_decomposition_matches_participation_deltas() -> None:
    state = make_state()
    analysis = analyze_two_edge_flip_causality(
        state,
        0,
        1,
    )

    vertex_delta = np.zeros_like(
        analysis.vertex_participation_delta
    )
    edge_delta = np.zeros_like(
        analysis.edge_participation_delta
    )

    for change in analysis.clique_changes:
        vertex_delta[
            change.vertices,
            change.color,
        ] += change.delta
        edge_delta[
            change.edges,
            change.color,
        ] += change.delta

    assert np.array_equal(
        vertex_delta,
        analysis.vertex_participation_delta,
    )
    assert np.array_equal(
        edge_delta,
        analysis.edge_participation_delta,
    )


def test_pair_order_does_not_change_net_analysis() -> None:
    state = make_state()

    forward = analyze_two_edge_flip_causality(
        state,
        0,
        1,
    )
    reverse = analyze_two_edge_flip_causality(
        state,
        1,
        0,
    )

    assert forward.score_before == reverse.score_before
    assert forward.score_after == reverse.score_after
    assert forward.exact_reward == reverse.exact_reward
    assert forward.interaction_reward == reverse.interaction_reward
    assert np.array_equal(
        forward.participation_after.vertices,
        reverse.participation_after.vertices,
    )
    assert np.array_equal(
        forward.participation_after.edges,
        reverse.participation_after.edges,
    )

    forward_events = {
        (
            change.clique_index,
            change.color,
            change.delta,
        )
        for change in forward.clique_changes
    }
    reverse_events = {
        (
            change.clique_index,
            change.color,
            change.delta,
        )
        for change in reverse.clique_changes
    }

    assert forward_events == reverse_events


def test_shared_vertex_property() -> None:
    state = make_state()
    graph_edges = state.graph.edges

    first_edge = 0
    first_vertices = set(
        int(vertex)
        for vertex in graph_edges[first_edge]
    )

    sharing_edge = next(
        edge
        for edge in range(1, state.number_of_edges)
        if first_vertices
        & {
            int(vertex)
            for vertex in graph_edges[edge]
        }
    )
    disjoint_edge = next(
        edge
        for edge in range(1, state.number_of_edges)
        if not (
            first_vertices
            & {
                int(vertex)
                for vertex in graph_edges[edge]
            }
        )
    )

    sharing = analyze_two_edge_flip_causality(
        state,
        first_edge,
        sharing_edge,
    )
    disjoint = analyze_two_edge_flip_causality(
        state,
        first_edge,
        disjoint_edge,
    )

    assert sharing.shared_vertex
    assert not disjoint.shared_vertex


def test_returned_arrays_are_read_only() -> None:
    state = make_state()
    analysis = analyze_two_edge_flip_causality(
        state,
        0,
        1,
    )

    assert not analysis.greedy_rewards_before.flags.writeable
    assert not analysis.greedy_rewards_after.flags.writeable
    assert not analysis.participation_before.vertices.flags.writeable
    assert not analysis.participation_before.edges.flags.writeable
    assert not analysis.participation_after.vertices.flags.writeable
    assert not analysis.participation_after.edges.flags.writeable


@pytest.mark.parametrize(
    ("first_edge", "second_edge", "exception"),
    [
        (True, 1, TypeError),
        (0.5, 1, TypeError),
        (0, False, TypeError),
        (0, 1.5, TypeError),
        (-1, 1, IndexError),
        (0, -1, IndexError),
        (10_000, 1, IndexError),
        (0, 10_000, IndexError),
        (0, 0, ValueError),
    ],
)
def test_invalid_edge_arguments_raise(
    first_edge,
    second_edge,
    exception,
) -> None:
    state = make_state()

    with pytest.raises(exception):
        analyze_two_edge_flip_causality(
            state,
            first_edge,
            second_edge,
        )
