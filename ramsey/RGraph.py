"""Immutable host-graph topology and precomputed incidence indexes."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from types import MappingProxyType
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from .RProblem import RProblem

EdgeTable = NDArray[np.uint8]
CliqueTable = NDArray[np.uint16]
EdgeToCliqueTable = NDArray[np.uint32]


def enumerate_edges(
    n_vertices: int,
) -> EdgeTable:
    """
    Enumerate every unordered edge of the complete graph K_n.
    """
    if n_vertices < 2:
        raise ValueError("n_vertices must be at least two.")

    if n_vertices > 256:
        raise ValueError("n_vertices cannot exceed 256 " "with uint8 vertex indexes.")

    return np.asarray(
        list(
            combinations(
                range(n_vertices),
                2,
            )
        ),
        dtype=np.uint8,
    )


def enumerate_clique_edges(
    edges: EdgeTable,
    n_vertices: int,
    clique_size: int,
) -> CliqueTable:
    """
    Enumerate host-edge indexes belonging to every requested clique.
    """
    expected_edges = comb(
        n_vertices,
        2,
    )

    if edges.shape != (
        expected_edges,
        2,
    ):
        raise ValueError(
            f"Expected edge-table shape "
            f"({expected_edges}, 2), "
            f"received {edges.shape}."
        )

    if clique_size < 2 or clique_size > n_vertices:
        raise ValueError("clique_size must be between " "two and n_vertices.")

    if expected_edges > np.iinfo(np.uint16).max:
        raise ValueError(
            "The host graph has too many edges " "for uint16 edge indexes."
        )

    edge_lookup = {
        (
            int(i),
            int(j),
        ): edge
        for edge, (i, j) in enumerate(edges)
    }

    number_of_cliques = comb(
        n_vertices,
        clique_size,
    )

    edges_per_clique = comb(
        clique_size,
        2,
    )

    clique_edges = np.empty(
        (
            number_of_cliques,
            edges_per_clique,
        ),
        dtype=np.uint16,
    )

    vertex_subsets = combinations(
        range(n_vertices),
        clique_size,
    )

    for (
        clique_index,
        vertices,
    ) in enumerate(vertex_subsets):
        clique_edges[clique_index] = [
            edge_lookup[(i, j)]
            for i, j in combinations(
                vertices,
                2,
            )
        ]

    return clique_edges


def build_edge_to_cliques(
    clique_edges: CliqueTable,
    n_vertices: int,
    number_of_edges: int,
    clique_size: int,
) -> EdgeToCliqueTable:
    """
    List every requested clique containing each host-graph edge.
    """
    (
        number_of_cliques,
        edges_per_clique,
    ) = clique_edges.shape

    expected_shape = (
        comb(
            n_vertices,
            clique_size,
        ),
        comb(
            clique_size,
            2,
        ),
    )

    if clique_edges.shape != expected_shape:
        raise ValueError(
            f"Expected clique-edge shape "
            f"{expected_shape}, received "
            f"{clique_edges.shape}."
        )

    flat_edges = clique_edges.reshape(-1)

    flat_clique_indices = np.repeat(
        np.arange(
            number_of_cliques,
            dtype=np.uint32,
        ),
        edges_per_clique,
    )

    order = np.argsort(
        flat_edges,
        kind="stable",
    )

    cliques_grouped_by_edge = flat_clique_indices[order]

    cliques_per_edge = comb(
        n_vertices - 2,
        clique_size - 2,
    )

    edge_to_cliques = cliques_grouped_by_edge.reshape(
        number_of_edges,
        cliques_per_edge,
    )

    return edge_to_cliques


def _read_only(
    array: np.ndarray,
) -> np.ndarray:
    """
    Mark an owned NumPy array as immutable and return it.
    """
    array.flags.writeable = False
    return array


@dataclass(frozen=True, slots=True)
class RSubgraphIndex:
    """
    Precomputed occurrence tables for one forbidden clique size.
    """

    clique_size: int
    clique_edges: CliqueTable
    edge_to_cliques: EdgeToCliqueTable

    @property
    def clique_count(self) -> int:
        """
        Return the number of indexed clique occurrences.
        """
        return len(self.clique_edges)

    @property
    def edges_per_clique(self) -> int:
        """
        Return the number of edges in each indexed clique.
        """
        return self.clique_edges.shape[1]

    @property
    def cliques_per_edge(self) -> int:
        """
        Return the number of indexed cliques containing each edge.
        """
        return self.edge_to_cliques.shape[1]


class RGraph:
    """
    Complete host-graph topology required by one Ramsey problem.
    """

    def __init__(
        self,
        problem: RProblem,
    ) -> None:
        self._problem = problem

        edges = enumerate_edges(problem.n_vertices)

        self._edges = _read_only(edges)

        subgraph_indexes: dict[
            int,
            RSubgraphIndex,
        ] = {}

        for clique_size in problem.required_clique_sizes:
            clique_edges = enumerate_clique_edges(
                self._edges,
                n_vertices=problem.n_vertices,
                clique_size=clique_size,
            )

            edge_to_cliques = build_edge_to_cliques(
                clique_edges,
                n_vertices=problem.n_vertices,
                number_of_edges=problem.edge_count,
                clique_size=clique_size,
            )

            subgraph_indexes[clique_size] = RSubgraphIndex(
                clique_size=clique_size,
                clique_edges=_read_only(clique_edges),
                edge_to_cliques=_read_only(edge_to_cliques),
            )

        self._subgraph_indexes = MappingProxyType(subgraph_indexes)

    @property
    def problem(self) -> RProblem:
        """
        Return the mathematical problem represented by this graph.
        """
        return self._problem

    @property
    def edges(self) -> EdgeTable:
        """
        Return the canonical host-edge table.
        """
        return self._edges

    @property
    def number_of_edges(self) -> int:
        """
        Return the number of colorable host edges.
        """
        return len(self._edges)

    @property
    def subgraph_indexes(
        self,
    ) -> Mapping[
        int,
        RSubgraphIndex,
    ]:
        """
        Return the indexed forbidden clique sizes.
        """
        return self._subgraph_indexes

    def subgraph_index(
        self,
        clique_size: int,
    ) -> RSubgraphIndex:
        """
        Return precomputed tables for one required clique size.
        """
        try:
            return self._subgraph_indexes[clique_size]

        except KeyError as error:
            raise KeyError(
                f"Clique size {clique_size} " "was not indexed for this problem."
            ) from error
