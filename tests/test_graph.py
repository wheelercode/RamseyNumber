"""Graph enumeration and incidence invariants."""

"""Characterize graph enumeration and incidence invariants."""

from math import comb

import numpy as np
import pytest

import RamseyGraph as graph


def test_k10_edge_enumeration_is_canonical() -> None:
    edges = graph.enumerate_edges(10)

    assert edges.shape == (
        comb(10, 2),
        2,
    )

    assert edges.dtype == np.uint8
    assert np.array_equal(
        edges[0],
        [0, 1],
    )
    assert np.array_equal(
        edges[-1],
        [8, 9],
    )
    assert np.all(edges[:, 0] < edges[:, 1])


def test_k10_triangle_edge_rows_match_known_encoding() -> None:
    edges = graph.enumerate_edges(10)

    triangles = graph.enumerate_kn_edges(
        edges,
        n_vertices=10,
        k_size=3,
    )

    assert triangles.shape == (
        comb(10, 3),
        comb(3, 2),
    )

    assert np.array_equal(
        triangles[0],
        [0, 1, 9],
    )
    assert np.array_equal(
        triangles[1],
        [0, 2, 10],
    )
    assert np.array_equal(
        triangles[-1],
        [42, 43, 44],
    )


def test_k43_k5_dimensions_and_edge_incidence(
    r55_data,
) -> None:
    assert r55_data.edges.shape == (
        903,
        2,
    )

    assert r55_data.kn_edges.shape == (
        962_598,
        10,
    )

    assert r55_data.edge_to_kn.shape == (
        903,
        10_660,
    )

    first_edge_cliques = r55_data.kn_edges[r55_data.edge_to_kn[0]]

    assert np.all(
        np.any(
            first_edge_cliques == 0,
            axis=1,
        )
    )


def test_enumerate_kn_edges_rejects_wrong_edge_table_shape() -> None:
    wrong_edges = np.empty(
        (44, 2),
        dtype=np.uint8,
    )

    with pytest.raises(
        ValueError,
        match="edge-table shape",
    ):
        graph.enumerate_kn_edges(
            wrong_edges,
            n_vertices=10,
            k_size=3,
        )
