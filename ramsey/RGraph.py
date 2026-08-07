"""Immutable host-graph topology and precomputed incidence indexes.

Defines :class:`RGraph`, the complete-graph host topology for one
:class:`ramsey.RProblem.RProblem`, together with the module-level
functions that enumerate its edges and build, for each required
forbidden clique size, the two lookup tables search and scoring code
depend on: which host edges belong to each clique
(:func:`enumerate_clique_edges`), and which cliques contain each host
edge (:func:`build_edge_to_cliques`). These tables are the incidence
structure that lets :class:`ramsey.RState.RSearchState` update clique
counts incrementally after a single-edge flip instead of rescanning
every clique.
"""

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
    """Enumerate every unordered edge of the complete graph K_n.

    Args:
        n_vertices (int): Number of vertices in the host graph; must be
            at least two and at most 256 (the ``uint8`` vertex-index
            limit).

    Returns:
        EdgeTable: ``uint8`` array of shape ``(comb(n_vertices, 2), 2)``
        listing every unordered vertex pair ``(i, j)`` with ``i < j``,
        in combinatorial (lexicographic) order. This order defines the
        canonical host-edge index used throughout the package.

    Raises:
        ValueError: If ``n_vertices`` is less than two, or greater than
            256.
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
    """Enumerate host-edge indexes belonging to every requested clique.

    Iterates every vertex subset of size ``clique_size`` in
    combinatorial order and, for each, looks up the host-edge index of
    every internal pair via a dictionary built from ``edges``.

    Args:
        edges (EdgeTable): Canonical host-edge table, as returned by
            :func:`enumerate_edges`.
        n_vertices (int): Number of vertices in the host graph.
        clique_size (int): Size of the vertex subsets to enumerate;
            must be between two and ``n_vertices``.

    Returns:
        CliqueTable: ``uint16`` array of shape
        ``(comb(n_vertices, clique_size), comb(clique_size, 2))``. Row
        ``c`` lists the host-edge indices of every edge within clique
        ``c``, in combinatorial vertex-pair order.

    Raises:
        ValueError: If ``edges`` does not have the shape implied by
            ``n_vertices``, if ``clique_size`` is out of range, or if
            the host graph has too many edges to index with ``uint16``.
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
    """List every requested clique containing each host-graph edge.

    Inverts ``clique_edges`` (clique to edges) into a per-edge table
    (edge to cliques). It flattens ``clique_edges`` and its parallel
    clique-index array, stable-sorts by edge index to group clique
    indices by the edge they belong to, then reshapes into a uniform
    ``(number_of_edges, cliques_per_edge)`` table. This relies on every
    host edge belonging to exactly the same number of indexed cliques,
    which holds because the host graph is complete.

    Args:
        clique_edges (CliqueTable): Clique-to-edges table, as returned
            by :func:`enumerate_clique_edges`.
        n_vertices (int): Number of vertices in the host graph.
        number_of_edges (int): Number of host-graph edges (rows of the
            result).
        clique_size (int): Size of the indexed cliques.

    Returns:
        EdgeToCliqueTable: ``uint32`` array of shape
        ``(number_of_edges, comb(n_vertices - 2, clique_size - 2))``.
        Row ``e`` lists the indices (into ``clique_edges``) of every
        clique containing host edge ``e``.

    Raises:
        ValueError: If ``clique_edges`` does not have the shape implied
            by ``n_vertices`` and ``clique_size``.
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
    """Mark an owned NumPy array as immutable and return it.

    Unlike similar helpers in other modules, this does not copy: the
    caller must already own ``array`` exclusively before calling this.

    Args:
        array (np.ndarray): Array to freeze in place.

    Returns:
        np.ndarray: The same array object, with ``flags.writeable`` set
        to ``False``.
    """
    array.flags.writeable = False
    return array


@dataclass(frozen=True, slots=True)
class RSubgraphIndex:
    """Precomputed occurrence tables for one forbidden clique size.

    Attributes:
        clique_size (int): Vertex-subset size these tables index.
        clique_edges (CliqueTable): ``uint16`` array of shape
            ``(clique_count, edges_per_clique)``; row ``c`` lists the
            host-edge indices belonging to clique ``c``.
        edge_to_cliques (EdgeToCliqueTable): ``uint32`` array of shape
            ``(number_of_edges, cliques_per_edge)``; row ``e`` lists
            the indices of every clique containing host edge ``e``.
    """

    clique_size: int
    clique_edges: CliqueTable
    edge_to_cliques: EdgeToCliqueTable

    @property
    def clique_count(self) -> int:
        """int: Number of indexed clique occurrences."""
        return len(self.clique_edges)

    @property
    def edges_per_clique(self) -> int:
        """int: Number of edges in each indexed clique."""
        return self.clique_edges.shape[1]

    @property
    def cliques_per_edge(self) -> int:
        """int: Number of indexed cliques containing each edge."""
        return self.edge_to_cliques.shape[1]


class RGraph:
    """Complete host-graph topology required by one Ramsey problem.

    Builds and owns the canonical edge table and, for every distinct
    clique size the problem requires (:attr:`RProblem.required_clique_sizes`),
    the precomputed :class:`RSubgraphIndex` occurrence tables used to
    maintain clique counts incrementally.
    """

    def __init__(
        self,
        problem: RProblem,
    ) -> None:
        """Build the host-graph edge table and per-clique-size indexes.

        Args:
            problem (RProblem): Ramsey problem specifying the vertex
                count and forbidden clique sizes to index.
        """
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
        """RProblem: Mathematical problem represented by this graph."""
        return self._problem

    @property
    def edges(self) -> EdgeTable:
        """EdgeTable: Canonical host-edge table (see :func:`enumerate_edges`)."""
        return self._edges

    @property
    def number_of_edges(self) -> int:
        """int: Number of colorable host edges."""
        return len(self._edges)

    @property
    def subgraph_indexes(
        self,
    ) -> Mapping[
        int,
        RSubgraphIndex,
    ]:
        """Mapping[int, RSubgraphIndex]: Indexed tables keyed by clique size.

        Covers every distinct clique size in
        ``problem.required_clique_sizes``. The mapping is read-only
        (backed by :class:`types.MappingProxyType`).
        """
        return self._subgraph_indexes

    def subgraph_index(
        self,
        clique_size: int,
    ) -> RSubgraphIndex:
        """Return precomputed tables for one required clique size.

        Args:
            clique_size (int): Clique size to look up.

        Returns:
            RSubgraphIndex: Precomputed occurrence tables for
            ``clique_size``.

        Raises:
            KeyError: If ``clique_size`` was not indexed for this
                graph's problem.
        """
        try:
            return self._subgraph_indexes[clique_size]

        except KeyError as error:
            raise KeyError(
                f"Clique size {clique_size} " "was not indexed for this problem."
            ) from error
