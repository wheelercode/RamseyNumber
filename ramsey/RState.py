"""Mutable incremental state for one active search attempt."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .RColoring import RColoring
from .RGraph import RGraph, RSubgraphIndex
from .RScoring import count_color_edges_per_clique


class RSearchState:
    """
    Incrementally maintained state for symmetric two-color search.

    RSearchState is the sole owner of mutable coloring, clique-count,
    histogram, and score data.
    """

    def __init__(
        self,
        coloring: RColoring,
    ) -> None:
        problem = coloring.graph.problem

        if problem.n_colors != 2 or not problem.is_symmetric:
            raise ValueError(
                "RSearchState currently requires a " "symmetric two-color problem."
            )

        self._graph = coloring.graph

        self._clique_size = problem.required_clique_sizes[0]

        self._index = self._graph.subgraph_index(self._clique_size)

        self._colors = coloring.mutable_copy()

        self._color_one_counts = count_color_edges_per_clique(
            coloring=coloring,
            clique_size=self._clique_size,
            color=1,
        )

        self._histogram = np.bincount(
            self._color_one_counts,
            minlength=self._index.edges_per_clique + 1,
        ).astype(np.int64)

        self._score = int(self._histogram[0] + self._histogram[-1])

        self._version = 0

    @property
    def graph(self) -> RGraph:
        """
        Return the immutable graph topology.
        """
        return self._graph

    @property
    def index(self) -> RSubgraphIndex:
        """
        Return the clique index maintained by this state.
        """
        return self._index

    @property
    def clique_size(self) -> int:
        """
        Return the forbidden clique size maintained by this state.
        """
        return self._clique_size

    @property
    def edges_per_clique(self) -> int:
        """
        Return the number of edges in each maintained clique.
        """
        return self._index.edges_per_clique

    @property
    def number_of_edges(self) -> int:
        """
        Return the number of available edge-flip actions.
        """
        return self._graph.number_of_edges

    @property
    def score(self) -> int:
        """
        Return the current monochromatic-clique score.
        """
        return self._score

    @property
    def version(self) -> int:
        """
        Return the number of mutations applied to this state.
        """
        return self._version

    @property
    def colors(
        self,
    ) -> NDArray[np.uint8]:
        """
        Return a read-only view of the current edge colors.
        """
        return self._read_only_view(self._colors)

    @property
    def color_one_counts(
        self,
    ) -> NDArray[np.uint8]:
        """
        Return a read-only view of per-clique color-one counts.
        """
        return self._read_only_view(self._color_one_counts)

    @property
    def histogram(
        self,
    ) -> NDArray[np.int64]:
        """
        Return a read-only view of the current count histogram.
        """
        return self._read_only_view(self._histogram)

    def coloring_snapshot(
        self,
    ) -> RColoring:
        """
        Return an immutable coloring containing the current state.
        """
        return RColoring(
            graph=self._graph,
            colors=self._colors,
        )

    def copy(
        self,
    ) -> "RSearchState":
        """
        Return an independent state with the same current coloring.

        The copied state begins at version zero because it has its own
        independent mutation history.
        """
        return RSearchState(self.coloring_snapshot())

    def apply_edge_flip(
        self,
        edge: int,
    ) -> int:
        """
        Flip one edge and return the exact score reduction.

        A positive result means the score improved.
        A negative result means the score became worse.
        """
        self._validate_edge(edge)

        old_score = self._score

        affected_cliques = self._index.edge_to_cliques[edge]

        old_counts = self._color_one_counts[affected_cliques].copy()

        if self._colors[edge] == 0:
            direction = 1
            self._colors[edge] = 1

        else:
            direction = -1
            self._colors[edge] = 0

        new_counts = (old_counts.astype(np.int16) + direction).astype(np.uint8)

        number_of_bins = self._index.edges_per_clique + 1

        self._histogram -= np.bincount(
            old_counts,
            minlength=number_of_bins,
        )

        self._histogram += np.bincount(
            new_counts,
            minlength=number_of_bins,
        )

        self._color_one_counts[affected_cliques] = new_counts

        self._score = int(self._histogram[0] + self._histogram[-1])

        self._version += 1

        return old_score - self._score

    def _validate_edge(
        self,
        edge: int,
    ) -> None:
        """
        Validate an encoded edge index.
        """
        if edge < 0 or edge >= self.number_of_edges:
            raise IndexError(f"Invalid edge index: {edge}")

    @staticmethod
    def _read_only_view(
        array: NDArray,
    ) -> NDArray:
        """
        Return a non-writeable view without copying array data.
        """
        view = array.view()
        view.flags.writeable = False

        return view
