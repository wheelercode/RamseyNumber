"""Tests for the causal footprint of one Ramsey edge flip."""

import numpy as np

from ramsey.RColoring import RColoring
from ramsey.REdgeFlipCausalAnalysis import (
    analyze_edge_flip_causality,
    monochromatic_participation,
)
from ramsey.RGraph import RGraph
from ramsey.RProblem import RProblem
from ramsey.RState import RSearchState


def _all_red_state(
    n_vertices: int = 6,
) -> RSearchState:
    graph = RGraph(
        RProblem.r55(
            n_vertices=n_vertices,
        )
    )

    return RSearchState(
        RColoring(
            graph,
            np.zeros(
                graph.number_of_edges,
                dtype=np.uint8,
            ),
        )
    )


def test_monochromatic_participation_counts_all_red_k6() -> None:
    state = _all_red_state()

    participation = monochromatic_participation(
        state
    )

    # K6 contains six K5s.  Each vertex belongs to five of them,
    # and each edge belongs to four of them.
    assert np.all(
        participation.red_vertices == 5
    )
    assert np.all(
        participation.blue_vertices == 0
    )
    assert np.all(
        participation.red_edges == 4
    )
    assert np.all(
        participation.blue_edges == 0
    )


def test_causal_analysis_does_not_mutate_source_state() -> None:
    state = _all_red_state()
    colors_before = state.colors.copy()
    score_before = state.score
    version_before = state.version

    analyze_edge_flip_causality(
        state,
        edge=0,
    )

    assert np.array_equal(
        state.colors,
        colors_before,
    )
    assert state.score == score_before
    assert state.version == version_before


def test_all_red_k6_flip_has_exact_causal_decomposition() -> None:
    state = _all_red_state()

    analysis = analyze_edge_flip_causality(
        state,
        edge=0,
    )

    # One K6 edge belongs to C(4, 3) = 4 K5s.  All four red
    # monochromatic K5s are destroyed by changing that edge blue.
    assert analysis.score_before == 6
    assert analysis.score_after == 2
    assert analysis.exact_reward == 4
    assert analysis.old_color == 0
    assert analysis.new_color == 1
    assert len(analysis.clique_changes) == 4

    assert all(
        change.color == 0
        and change.destroyed
        for change in analysis.clique_changes
    )

    # The two flipped-edge endpoints occur in all four destroyed
    # K5s.  Each of the other four vertices occurs in three.
    endpoint_a, endpoint_b = analysis.endpoints
    red_vertex_delta = (
        analysis.vertex_participation_delta[:, 0]
    )

    assert red_vertex_delta[endpoint_a] == -4
    assert red_vertex_delta[endpoint_b] == -4

    other_vertices = [
        vertex
        for vertex in range(6)
        if vertex not in analysis.endpoints
    ]

    assert np.all(
        red_vertex_delta[other_vertices] == -3
    )

    assert np.array_equal(
        analysis.vertex_event_counts,
        -red_vertex_delta,
    )


def test_causal_events_reconstruct_edge_participation_delta() -> None:
    state = _all_red_state()

    analysis = analyze_edge_flip_causality(
        state,
        edge=0,
    )

    replayed_delta = np.zeros_like(
        analysis.edge_participation_delta
    )

    for change in analysis.clique_changes:
        replayed_delta[
            change.edges,
            change.color,
        ] += change.delta

    assert np.array_equal(
        replayed_delta,
        analysis.edge_participation_delta,
    )


def test_causal_analysis_records_next_action_landscape() -> None:
    state = _all_red_state()

    analysis = analyze_edge_flip_causality(
        state,
        edge=0,
    )

    assert analysis.greedy_rewards_before.shape == (
        state.number_of_edges,
    )
    assert analysis.greedy_rewards_after.shape == (
        state.number_of_edges,
    )
    assert analysis.greedy_reward_delta.shape == (
        state.number_of_edges,
    )

    # On K6, the four changed K5s collectively touch every host edge.
    assert len(analysis.changed_structure_edges) == state.number_of_edges