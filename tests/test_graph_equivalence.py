"""Compare the parallel graph implementation with the reference code."""

import numpy as np
import pytest

import RamseyGraph as reference_graph
import RamseySearch as reference_search

from ramsey.RGraph import (
    RGraph,
    build_edge_to_cliques,
    enumerate_clique_edges,
    enumerate_edges,
)
from ramsey.RProblem import RProblem


def test_edge_enumeration_matches_reference() -> None:
    expected = reference_graph.enumerate_edges(10)

    actual = enumerate_edges(10)

    assert actual.dtype == expected.dtype

    assert np.array_equal(
        actual,
        expected,
    )


def test_clique_enumeration_matches_reference() -> None:
    edges = enumerate_edges(10)

    expected = reference_graph.enumerate_kn_edges(
        edges,
        n_vertices=10,
        k_size=5,
    )

    actual = enumerate_clique_edges(
        edges,
        n_vertices=10,
        clique_size=5,
    )

    assert actual.dtype == expected.dtype

    assert np.array_equal(
        actual,
        expected,
    )


def test_edge_incidence_matches_reference() -> None:
    edges = enumerate_edges(10)

    clique_edges = enumerate_clique_edges(
        edges,
        n_vertices=10,
        clique_size=5,
    )

    expected = reference_search.build_edge_to_kn(
        clique_edges,
        n_vertices=10,
        number_of_edges=len(edges),
        k_size=5,
    )

    actual = build_edge_to_cliques(
        clique_edges,
        n_vertices=10,
        number_of_edges=len(edges),
        clique_size=5,
    )

    assert actual.dtype == expected.dtype

    assert np.array_equal(
        actual,
        expected,
    )


def test_graph_owns_read_only_topology() -> None:
    problem = RProblem.r55(n_vertices=10)

    graph = RGraph(problem)

    index = graph.subgraph_index(5)

    assert graph.problem is problem
    assert graph.number_of_edges == 45
    assert index.clique_count == 252
    assert index.edges_per_clique == 10
    assert index.cliques_per_edge == 56

    assert not (graph.edges.flags.writeable)

    assert not (index.clique_edges.flags.writeable)

    assert not (index.edge_to_cliques.flags.writeable)

    with pytest.raises(ValueError):
        graph.edges[0, 0] = 9


def test_graph_indexes_each_distinct_asymmetric_clique_size() -> None:
    problem = RProblem(
        n_vertices=10,
        forbidden_clique_sizes=(
            4,
            6,
        ),
    )

    graph = RGraph(problem)

    assert tuple(graph.subgraph_indexes) == (
        4,
        6,
    )

    assert graph.subgraph_index(4).clique_count == 210

    assert graph.subgraph_index(6).clique_count == 210


def test_full_r55_graph_matches_reference(
    r55_data,
) -> None:
    graph = RGraph(RProblem.r55())

    index = graph.subgraph_index(5)

    assert np.array_equal(
        graph.edges,
        r55_data.edges,
    )

    assert np.array_equal(
        index.clique_edges,
        r55_data.kn_edges,
    )

    assert np.array_equal(
        index.edge_to_cliques,
        r55_data.edge_to_kn,
    )
