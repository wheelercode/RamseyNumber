"""Compare incremental state with the untouched reference implementation."""

import numpy as np
import pytest

import RamseyGraph as reference_graph
import RamseySearch as reference_search
import RamseyState as reference_state

from ramsey.RColoring import RColoring
from ramsey.RGraph import RGraph
from ramsey.RProblem import RProblem
from ramsey.RScoring import binary_histogram
from ramsey.RState import RSearchState


@pytest.fixture(scope="module")
def graph() -> RGraph:
    """
    Return a small symmetric graph for state tests.
    """
    return RGraph(RProblem.r55(n_vertices=10))


def make_states(
    graph: RGraph,
    seed: int,
):
    """
    Return equivalent new and reference search states.
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

    reference = reference_search.create_search_state(
        colors,
        graph.subgraph_index(5).clique_edges,
    )

    return (
        new_state,
        reference,
    )


def test_initial_state_matches_reference(
    graph: RGraph,
) -> None:
    state, reference = make_states(
        graph,
        31,
    )

    assert np.array_equal(
        state.colors,
        reference.coloring,
    )

    assert np.array_equal(
        state.color_one_counts,
        reference.blue_counts,
    )

    assert np.array_equal(
        state.histogram,
        reference.histogram,
    )

    assert state.score == reference.score

    assert state.version == 0


def test_state_arrays_are_exposed_read_only(
    graph: RGraph,
) -> None:
    state, _ = make_states(
        graph,
        32,
    )

    assert not (state.colors.flags.writeable)

    assert not (state.color_one_counts.flags.writeable)

    assert not (state.histogram.flags.writeable)

    with pytest.raises(ValueError):
        state.colors[0] = 1


def test_flip_sequence_matches_reference_exactly(
    graph: RGraph,
) -> None:
    state, reference = make_states(
        graph,
        33,
    )

    rng = np.random.default_rng(34)

    edge_to_cliques = graph.subgraph_index(5).edge_to_cliques

    edges = rng.integers(
        0,
        state.number_of_edges,
        size=500,
    )

    for (
        expected_version,
        edge,
    ) in enumerate(
        edges,
        start=1,
    ):
        edge = int(edge)

        new_reward = state.apply_edge_flip(edge)

        reference_reward = reference_state.apply_edge_flip(
            reference,
            edge_to_cliques,
            edge,
        )

        assert new_reward == reference_reward

        assert state.score == reference.score

        assert state.version == expected_version

        assert np.array_equal(
            state.colors,
            reference.coloring,
        )

        assert np.array_equal(
            state.color_one_counts,
            reference.blue_counts,
        )

        assert np.array_equal(
            state.histogram,
            reference.histogram,
        )


def test_coloring_snapshot_matches_current_state(
    graph: RGraph,
) -> None:
    state, _ = make_states(
        graph,
        35,
    )

    state.apply_edge_flip(7)

    snapshot = state.coloring_snapshot()

    assert np.array_equal(
        snapshot.colors,
        state.colors,
    )

    assert np.array_equal(
        binary_histogram(snapshot),
        state.histogram,
    )


def test_state_copy_is_independent(
    graph: RGraph,
) -> None:
    state, _ = make_states(
        graph,
        36,
    )

    copied = state.copy()

    original_colors = state.colors.copy()

    copied.apply_edge_flip(0)

    assert np.array_equal(
        state.colors,
        original_colors,
    )

    assert not np.array_equal(
        copied.colors,
        state.colors,
    )


def test_state_rejects_asymmetric_problem() -> None:
    graph = RGraph(
        RProblem(
            n_vertices=6,
            forbidden_clique_sizes=(
                3,
                4,
            ),
        )
    )

    coloring = RColoring(
        graph,
        np.zeros(
            graph.number_of_edges,
            dtype=np.uint8,
        ),
    )

    with pytest.raises(
        ValueError,
        match="symmetric two-color",
    ):
        RSearchState(coloring)
