"""Tests for exact K5 connection-class imbalance analysis."""

from itertools import combinations

import numpy as np

from ramsey import (
    RGraph,
    RProblem,
    RRandomConstruction,
    RSearchState,
    calculate_k5_connection_analysis,
)


def test_connection_classes_count_every_neighbor_once() -> None:
    """Exact O1-O4 classes must not double-count larger intersections."""
    graph = RGraph(RProblem.r55(n_vertices=9))
    coloring = RRandomConstruction(np.random.default_rng(7)).construct(graph)
    state = RSearchState(coloring)

    result = calculate_k5_connection_analysis(state)
    target_index = 0
    histograms = result.connection_histograms(state, target_index)

    expected_sizes = np.asarray(
        [
            5,
            40,
            60,
            20,
        ],
        dtype=np.int64,
    )

    np.testing.assert_array_equal(
        histograms.sum(axis=1),
        expected_sizes,
    )
    assert int(histograms.sum()) == 125


def test_connection_imbalance_sums_match_brute_force() -> None:
    """Subset-incidence inversion must equal direct exact intersections."""
    graph = RGraph(RProblem.r55(n_vertices=9))
    coloring = RRandomConstruction(np.random.default_rng(11)).construct(graph)
    state = RSearchState(coloring)

    result = calculate_k5_connection_analysis(state)
    vertices = list(combinations(range(9), 5))

    for target_index in (0, 17, len(vertices) - 1):
        target = set(vertices[target_index])
        brute_sums = np.zeros(4, dtype=np.int64)

        for neighbor_index, neighbor_vertices in enumerate(vertices):
            if neighbor_index == target_index:
                continue

            shared = len(target.intersection(neighbor_vertices))

            if 1 <= shared <= 4:
                brute_sums[shared - 1] += int(
                    result.local_imbalances[neighbor_index]
                )

        np.testing.assert_array_equal(
            result.class_imbalance_sums[target_index],
            brute_sums,
        )