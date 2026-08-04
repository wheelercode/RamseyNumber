"""Test interchangeable seed-coloring constructions."""

import numpy as np
import pytest

from ramsey.RColoring import RColoring
from ramsey.RConstruction import (
    RCyclicConstruction,
    RFixedConstruction,
    RRandomConstruction,
)
from ramsey.RGraph import RGraph
from ramsey.RProblem import RProblem
from ramsey.RScoring import (
    binary_histogram,
    score_coloring,
)

EXOO_HISTOGRAM = np.asarray(
    [
        43,
        8_815,
        43_516,
        130_161,
        239_467,
        253_055,
        175_225,
        80_969,
        25_929,
        5_418,
        0,
    ],
    dtype=np.int64,
)


def test_random_construction_is_reproducible() -> None:
    graph = RGraph(RProblem.r55(n_vertices=10))

    first = RRandomConstruction(np.random.default_rng(201))

    second = RRandomConstruction(np.random.default_rng(201))

    first_coloring = first.construct(graph)

    second_coloring = second.construct(graph)

    assert first_coloring.exact_equals(second_coloring)

    assert first_coloring.colors.shape == (graph.number_of_edges,)


def test_cyclic_construction_uses_circular_distance() -> None:
    graph = RGraph(RProblem.r55(n_vertices=7))

    construction = RCyclicConstruction((0, 1, 0))

    coloring = construction.construct(graph)

    edge_lookup = {
        tuple(map(int, endpoints)): edge for edge, endpoints in enumerate(graph.edges)
    }

    assert coloring.color_of_edge(edge_lookup[(0, 1)]) == 0

    assert coloring.color_of_edge(edge_lookup[(0, 2)]) == 1

    assert coloring.color_of_edge(edge_lookup[(0, 3)]) == 0

    # Distance six in K7 wraps to circular distance one.
    assert coloring.color_of_edge(edge_lookup[(0, 6)]) == 0

    # Difference five wraps to circular distance two.
    assert coloring.color_of_edge(edge_lookup[(1, 6)]) == 1


def test_cyclic_construction_validates_distance_count() -> None:
    graph = RGraph(RProblem.r55(n_vertices=10))

    construction = RCyclicConstruction((0, 1))

    with pytest.raises(
        ValueError,
        match="circular edge distances",
    ):
        construction.construct(graph)


def test_exoo_construction_matches_golden_histogram() -> None:
    graph = RGraph(RProblem.r55())

    coloring = RCyclicConstruction.exoo().construct(graph)

    assert np.array_equal(
        binary_histogram(coloring),
        EXOO_HISTOGRAM,
    )

    assert score_coloring(coloring) == 43


def test_fixed_construction_rebinds_equivalent_graph() -> None:
    first_graph = RGraph(RProblem.r55(n_vertices=10))

    second_graph = RGraph(RProblem.r55(n_vertices=10))

    colors = np.random.default_rng(202).integers(
        0,
        2,
        size=first_graph.number_of_edges,
        dtype=np.uint8,
    )

    original = RColoring(
        first_graph,
        colors,
    )

    restored = RFixedConstruction(original).construct(second_graph)

    assert restored.graph is second_graph

    assert restored.exact_equals(original)


def test_fixed_construction_rejects_different_problem() -> None:
    first_graph = RGraph(RProblem.r55(n_vertices=10))

    second_graph = RGraph(RProblem.r55(n_vertices=11))

    coloring = RColoring(
        first_graph,
        np.zeros(
            first_graph.number_of_edges,
            dtype=np.uint8,
        ),
    )

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        RFixedConstruction(coloring).construct(second_graph)
