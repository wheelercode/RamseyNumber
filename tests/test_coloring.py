"""Coloring validation and conversion round trips."""

"""Characterize coloring generation and matrix conversion."""

import numpy as np
import pytest

import RamseyGraph as graph


def test_random_coloring_is_binary_and_reproducible() -> None:
    first = graph.random_coloring(
        45,
        np.random.default_rng(12345),
    )

    second = graph.random_coloring(
        45,
        np.random.default_rng(12345),
    )

    assert first.dtype == np.uint8
    assert np.array_equal(
        first,
        second,
    )
    assert np.all((first == 0) | (first == 1))


def test_coloring_round_trips_through_symmetric_matrix() -> None:
    edges = graph.enumerate_edges(10)

    coloring = graph.random_coloring(
        len(edges),
        np.random.default_rng(7),
    )

    matrix = graph.coloring_to_matrix(
        coloring,
        edges,
        n_vertices=10,
    )

    restored = graph.matrix_to_coloring(
        matrix,
        edges,
    )

    assert matrix.shape == (
        10,
        10,
    )

    assert np.array_equal(
        matrix,
        matrix.T,
    )

    assert np.all(np.diag(matrix) == 0)

    assert np.array_equal(
        restored,
        coloring,
    )

    assert not np.shares_memory(
        restored,
        matrix,
    )


def test_matrix_to_coloring_rejects_asymmetric_matrix() -> None:
    edges = graph.enumerate_edges(3)

    matrix = np.zeros(
        (3, 3),
        dtype=np.uint8,
    )

    matrix[0, 1] = 1

    with pytest.raises(
        ValueError,
        match="symmetric",
    ):
        graph.matrix_to_coloring(
            matrix,
            edges,
        )


def test_matrix_to_coloring_rejects_nonbinary_values() -> None:
    edges = graph.enumerate_edges(3)

    matrix = np.zeros(
        (3, 3),
        dtype=np.uint8,
    )

    matrix[0, 1] = 2
    matrix[1, 0] = 2

    with pytest.raises(
        ValueError,
        match="zero or one",
    ):
        graph.matrix_to_coloring(
            matrix,
            edges,
        )
