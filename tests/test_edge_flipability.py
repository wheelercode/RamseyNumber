"""Tests for edge flipability derived from local K5 roles."""

import numpy as np
import pytest

from ramsey.RAction import analyze_actions
from ramsey.RColoring import RColoring
from ramsey.REdgeFlipability import calculate_edge_flipability
from ramsey.RGraph import RGraph
from ramsey.RProblem import RProblem
from ramsey.RState import RSearchState


def make_k5_state(colors: np.ndarray) -> RSearchState:
    graph = RGraph(RProblem.r55(n_vertices=5))
    return RSearchState(RColoring(graph, colors))


def test_unique_minority_edge_has_zero_local_flipability() -> None:
    graph = RGraph(RProblem.r55(n_vertices=5))
    colors = np.zeros(graph.number_of_edges, dtype=np.uint8)

    # Nine red edges and one blue edge.
    colors[0] = 1
    state = RSearchState(RColoring(graph, colors))

    result = calculate_edge_flipability(state)

    assert result.flipabilities[0] == 0
    assert np.array_equal(
        result.flipabilities[1:],
        np.full(9, 8, dtype=np.int64),
    )

    assert result.role_profiles[0, 0] == 1
    assert np.all(result.role_profiles[1:, 8] == 1)


def test_monochromatic_k5_edges_are_highly_flippable() -> None:
    graph = RGraph(RProblem.r55(n_vertices=5))
    colors = np.zeros(graph.number_of_edges, dtype=np.uint8)
    state = RSearchState(RColoring(graph, colors))

    result = calculate_edge_flipability(state)

    assert np.array_equal(
        result.flipabilities,
        np.full(10, 9, dtype=np.int64),
    )
    assert np.all(result.role_profiles[:, 9] == 1)


def test_role_profiles_include_every_incident_k5() -> None:
    graph = RGraph(RProblem.r55(n_vertices=8))
    rng = np.random.default_rng(801)
    colors = rng.integers(
        0,
        2,
        size=graph.number_of_edges,
        dtype=np.uint8,
    )
    state = RSearchState(RColoring(graph, colors))

    result = calculate_edge_flipability(state)

    assert np.all(
        result.role_profiles.sum(axis=1)
        == state.index.cliques_per_edge
    )


def test_existing_action_analysis_can_be_reused() -> None:
    graph = RGraph(RProblem.r55(n_vertices=8))
    colors = np.zeros(graph.number_of_edges, dtype=np.uint8)
    state = RSearchState(RColoring(graph, colors))
    analysis = analyze_actions(state)

    direct = calculate_edge_flipability(state)
    reused = calculate_edge_flipability(
        state,
        analysis,
    )

    assert np.array_equal(
        direct.role_profiles,
        reused.role_profiles,
    )
    assert np.array_equal(
        direct.flipabilities,
        reused.flipabilities,
    )


def test_stale_action_analysis_is_rejected() -> None:
    graph = RGraph(RProblem.r55(n_vertices=8))
    colors = np.zeros(graph.number_of_edges, dtype=np.uint8)
    state = RSearchState(RColoring(graph, colors))
    analysis = analyze_actions(state)

    state.apply_edge_flip(0)

    with pytest.raises(ValueError):
        calculate_edge_flipability(
            state,
            analysis,
        )


def test_results_are_read_only() -> None:
    graph = RGraph(RProblem.r55(n_vertices=5))
    colors = np.zeros(graph.number_of_edges, dtype=np.uint8)
    state = RSearchState(RColoring(graph, colors))

    result = calculate_edge_flipability(state)

    assert not result.role_profiles.flags.writeable
    assert not result.flipabilities.flags.writeable