"""Compare exact scoring with the untouched reference implementation."""

from math import comb

import numpy as np
import pytest

import RamseyGraph as reference_graph
import RamseyScoring as reference_scoring
from Exoo import exoo_cyclic_coloring

from ramsey.RColoring import RColoring
from ramsey.RGraph import RGraph
from ramsey.RProblem import RProblem
from ramsey.RScoring import (
    binary_histogram,
    count_color_edges_per_clique,
    evaluate_coloring,
    score_coloring,
)


@pytest.fixture(scope="module")
def graph() -> RGraph:
    """
    Return a small symmetric graph for exact scoring tests.
    """
    return RGraph(RProblem.r55(n_vertices=10))


def test_blue_counts_match_reference(
    graph: RGraph,
) -> None:
    colors = reference_graph.random_coloring(
        graph.number_of_edges,
        np.random.default_rng(21),
    )

    coloring = RColoring(
        graph,
        colors,
    )

    clique_edges = graph.subgraph_index(5).clique_edges

    expected = reference_scoring.count_blue_edges_per_kn(
        colors,
        clique_edges,
    )

    actual = count_color_edges_per_clique(
        coloring,
        clique_size=5,
        color=1,
    )

    assert np.array_equal(
        actual,
        expected,
    )


def test_binary_histogram_and_score_match_reference(
    graph: RGraph,
) -> None:
    colors = reference_graph.random_coloring(
        graph.number_of_edges,
        np.random.default_rng(22),
    )

    coloring = RColoring(
        graph,
        colors,
    )

    clique_edges = graph.subgraph_index(5).clique_edges

    expected_histogram = reference_scoring.kn_histogram(
        colors,
        clique_edges,
    )

    expected_score = reference_scoring.score_coloring(
        colors,
        clique_edges,
    )

    assert np.array_equal(
        binary_histogram(coloring),
        expected_histogram,
    )

    assert score_coloring(coloring) == expected_score


def test_score_report_matches_legacy_histogram_endpoints(
    graph: RGraph,
) -> None:
    coloring = RColoring(
        graph,
        reference_graph.random_coloring(
            graph.number_of_edges,
            np.random.default_rng(23),
        ),
    )

    legacy_histogram = binary_histogram(coloring)

    report = evaluate_coloring(coloring)

    assert report.by_color == (
        int(legacy_histogram[0]),
        int(legacy_histogram[-1]),
    )

    assert report.total == int(legacy_histogram[0] + legacy_histogram[-1])

    assert not (report.histogram_for_color(0).flags.writeable)

    assert not (report.histogram_for_color(1).flags.writeable)


def test_asymmetric_score_uses_color_specific_clique_sizes() -> None:
    graph = RGraph(
        RProblem(
            n_vertices=6,
            forbidden_clique_sizes=(
                3,
                4,
            ),
        )
    )

    all_color_zero = RColoring(
        graph,
        np.zeros(
            graph.number_of_edges,
            dtype=np.uint8,
        ),
    )

    all_color_one = all_color_zero.complement()

    zero_report = evaluate_coloring(all_color_zero)

    one_report = evaluate_coloring(all_color_one)

    assert zero_report.by_color == (
        comb(6, 3),
        0,
    )

    assert zero_report.total == comb(
        6,
        3,
    )

    assert one_report.by_color == (
        0,
        comb(6, 4),
    )

    assert one_report.total == comb(
        6,
        4,
    )


def test_full_exoo_score_matches_reference(
    r55_data,
) -> None:
    graph = RGraph(RProblem.r55())

    colors = exoo_cyclic_coloring(
        r55_data.edges,
        n_vertices=43,
    )

    coloring = RColoring(
        graph,
        colors,
    )

    expected_histogram = reference_scoring.kn_histogram(
        colors,
        r55_data.kn_edges,
    )

    assert np.array_equal(
        binary_histogram(coloring),
        expected_histogram,
    )

    assert score_coloring(coloring) == 43
