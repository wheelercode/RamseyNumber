"""Shared pytest fixtures for RamseySearch tests."""

"""Shared fixtures built from the untouched reference implementation."""

from dataclasses import dataclass

import numpy as np
import pytest

import RamseyGraph as reference_graph
import RamseySearch as reference_search


@dataclass(frozen=True)
class ReferenceGraphData:
    """Precomputed reference arrays for one complete graph."""

    n_vertices: int
    k_size: int
    edges: np.ndarray
    kn_edges: np.ndarray
    edge_to_kn: np.ndarray


def build_reference_graph_data(
    n_vertices: int,
    k_size: int,
) -> ReferenceGraphData:
    """Build graph arrays using only the current production modules."""
    edges = reference_graph.enumerate_edges(n_vertices)

    kn_edges = reference_graph.enumerate_kn_edges(
        edges,
        n_vertices=n_vertices,
        k_size=k_size,
    )

    edge_to_kn = reference_search.build_edge_to_kn(
        kn_edges,
        n_vertices=n_vertices,
        number_of_edges=len(edges),
        k_size=k_size,
    )

    return ReferenceGraphData(
        n_vertices=n_vertices,
        k_size=k_size,
        edges=edges,
        kn_edges=kn_edges,
        edge_to_kn=edge_to_kn,
    )


@pytest.fixture(scope="session")
def k10_k5_data() -> ReferenceGraphData:
    """
    Small, fast fixture with the same ten-edge K5 scoring shape.
    """
    return build_reference_graph_data(
        n_vertices=10,
        k_size=5,
    )


@pytest.fixture(scope="session")
def r55_data() -> ReferenceGraphData:
    """
    Full K43/K5 fixture used only for essential golden checks.
    """
    return build_reference_graph_data(
        n_vertices=43,
        k_size=5,
    )
