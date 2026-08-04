"""Compare coloring behavior with the untouched reference functions."""

import numpy as np
import pytest

import RamseyGraph as reference_graph

from ramsey.RColoring import RColoring
from ramsey.RGraph import RGraph
from ramsey.RProblem import RProblem


@pytest.fixture(scope="module")
def graph() -> RGraph:
    """
    Return a small symmetric graph for coloring tests.
    """
    return RGraph(RProblem.r55(n_vertices=10))


def test_coloring_owns_read_only_copy(
    graph: RGraph,
) -> None:
    source = reference_graph.random_coloring(
        graph.number_of_edges,
        np.random.default_rng(11),
    )

    coloring = RColoring(
        graph,
        source,
    )

    expected = source.copy()

    source[:] = np.uint8(1) - source

    assert np.array_equal(
        coloring.colors,
        expected,
    )

    assert not (coloring.colors.flags.writeable)

    assert coloring.mutable_copy().flags.writeable


def test_color_matrix_matches_reference(
    graph: RGraph,
) -> None:
    colors = reference_graph.random_coloring(
        graph.number_of_edges,
        np.random.default_rng(12),
    )

    coloring = RColoring(
        graph,
        colors,
    )

    expected = reference_graph.coloring_to_matrix(
        colors,
        graph.edges,
        n_vertices=10,
    )

    actual = coloring.to_color_matrix()

    assert np.array_equal(
        actual,
        expected,
    )


def test_color_matrix_round_trip(
    graph: RGraph,
) -> None:
    original = RColoring(
        graph,
        reference_graph.random_coloring(
            graph.number_of_edges,
            np.random.default_rng(13),
        ),
    )

    restored = RColoring.from_color_matrix(
        graph,
        original.to_color_matrix(),
    )

    assert restored.exact_equals(original)


def test_adjacency_matrix_round_trip(
    graph: RGraph,
) -> None:
    original = RColoring(
        graph,
        reference_graph.random_coloring(
            graph.number_of_edges,
            np.random.default_rng(14),
        ),
    )

    adjacency = original.to_adjacency_matrix(edge_color=1)

    restored = RColoring.from_adjacency_matrix(
        graph,
        adjacency,
        edge_color=1,
    )

    assert restored.exact_equals(original)

    assert np.array_equal(
        adjacency,
        adjacency.T,
    )

    assert np.all(np.diag(adjacency) == 0)


def test_complement_swaps_color_projections(
    graph: RGraph,
) -> None:
    coloring = RColoring(
        graph,
        reference_graph.random_coloring(
            graph.number_of_edges,
            np.random.default_rng(15),
        ),
    )

    complement = coloring.complement()

    assert np.array_equal(
        complement.colors,
        (np.uint8(1) - coloring.colors),
    )

    assert np.array_equal(
        coloring.to_adjacency_matrix(edge_color=0),
        complement.to_adjacency_matrix(edge_color=1),
    )

    assert complement.complement().exact_equals(coloring)


def test_vertex_degrees_and_isolated_vertices(
    graph: RGraph,
) -> None:
    empty_blue_graph = RColoring(
        graph,
        np.zeros(
            graph.number_of_edges,
            dtype=np.uint8,
        ),
    )

    assert np.array_equal(
        empty_blue_graph.vertex_degrees(edge_color=1),
        np.zeros(
            10,
            dtype=np.int64,
        ),
    )

    assert np.array_equal(
        empty_blue_graph.isolated_vertices(edge_color=1),
        np.arange(
            10,
            dtype=np.int32,
        ),
    )

    assert np.array_equal(
        empty_blue_graph.vertex_degrees(edge_color=0),
        np.full(
            10,
            9,
            dtype=np.int64,
        ),
    )


def test_exact_hash_is_stable_and_problem_specific(
    graph: RGraph,
) -> None:
    colors = reference_graph.random_coloring(
        graph.number_of_edges,
        np.random.default_rng(16),
    )

    first = RColoring(
        graph,
        colors,
    )

    second = RColoring(
        RGraph(RProblem.r55(n_vertices=10)),
        colors.copy(),
    )

    assert first.exact_hash() == second.exact_hash()

    assert first.exact_equals(second)

    assert len(first.exact_hash()) == 64


def test_coloring_rejects_invalid_shape_and_color(
    graph: RGraph,
) -> None:
    with pytest.raises(
        ValueError,
        match="coloring shape",
    ):
        RColoring(
            graph,
            np.zeros(
                44,
                dtype=np.uint8,
            ),
        )

    invalid = np.zeros(
        graph.number_of_edges,
        dtype=np.uint8,
    )

    invalid[0] = 2

    with pytest.raises(
        ValueError,
        match="valid color range",
    ):
        RColoring(
            graph,
            invalid,
        )

    noninteger = np.zeros(
        graph.number_of_edges,
        dtype=np.float64,
    )

    noninteger[0] = 0.5

    with pytest.raises(
        TypeError,
        match="must be integers",
    ):
        RColoring(
            graph,
            noninteger,
        )
