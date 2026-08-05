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

        # Full greedy action analysis needs, for every host edge, the
        # number of incident cliques in each color-one-count bin.  The
        # cache is lazy so neural/random searches that use the fast
        # action path never pay to construct it.
        self._action_profiles: NDArray[np.uint16] | None = None

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

    @property
    def action_profiles(
        self,
    ) -> NDArray[np.uint16]:
        """
        Return incrementally maintained all-edge clique profiles.

        profiles[e, k] is the number of indexed cliques containing
        edge e that currently contain exactly k color-one edges.

        The profiles are constructed only on first use.  Once they
        exist, every edge flip updates only the profile contributions
        of cliques affected by that flip.
        """
        if self._action_profiles is None:
            self._action_profiles = self._build_action_profiles()

        return self._read_only_view(self._action_profiles)

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

        if self._action_profiles is not None:
            self._update_action_profiles(
                affected_cliques=affected_cliques,
                old_counts=old_counts,
                new_counts=new_counts,
            )

        self._score = int(self._histogram[0] + self._histogram[-1])

        self._version += 1

        return old_score - self._score

    def apply_edge_recoloring(
        self,
        edges: NDArray[np.integer],
        colors: NDArray[np.integer],
    ) -> int:
        """
        Recolor several distinct edges and return exact score reduction.

        Only edges whose requested color differs from the current color
        are mutated.

        Each changed edge goes through apply_edge_flip(), preserving all
        incrementally maintained state:

        - edge colors
        - K5 color counts
        - histogram
        - monochromatic score
        - action-profile cache
        - state version
        """
        edges = np.asarray(edges)
        colors = np.asarray(colors)

        if edges.ndim != 1 or colors.ndim != 1:
            raise ValueError(
                "edges and colors must be one-dimensional."
            )

        if len(edges) != len(colors):
            raise ValueError(
                "edges and colors must have equal length."
            )

        if not np.issubdtype(edges.dtype, np.integer):
            raise TypeError(
                "edges must contain integers."
            )

        if not np.issubdtype(colors.dtype, np.integer):
            raise TypeError(
                "colors must contain integers."
            )

        normalized_edges = edges.astype(
            np.int32,
            copy=False,
        )

        normalized_colors = colors.astype(
            np.int8,
            copy=False,
        )

        if (
            np.any(normalized_edges < 0)
            or np.any(
                normalized_edges
                >= self.number_of_edges
            )
        ):
            raise IndexError(
                "edges contains an invalid edge index."
            )

        if (
            len(np.unique(normalized_edges))
            != len(normalized_edges)
        ):
            raise ValueError(
                "edges must not contain duplicates."
            )

        if np.any(
            (normalized_colors != 0)
            & (normalized_colors != 1)
        ):
            raise ValueError(
                "colors must contain only zero or one."
            )

        old_score = self._score

        for edge, color in zip(
            normalized_edges,
            normalized_colors,
        ):
            edge = int(edge)

            if self._colors[edge] != color:
                self.apply_edge_flip(edge)

        return old_score - self._score
        
    def _build_action_profiles(
        self,
    ) -> NDArray[np.uint16]:
        """
        Construct the complete action-profile cache from current state.

        This is the expensive full reconstruction.  It occurs at most
        once per RSearchState; subsequent mutations use the incremental
        update path.
        """
        if self._index.cliques_per_edge > np.iinfo(np.uint16).max:
            raise ValueError(
                "The number of cliques per edge exceeds the "
                "uint16 action-profile capacity."
            )

        affected_counts = self._color_one_counts[
            self._index.edge_to_cliques
        ]

        number_of_bins = self.edges_per_clique + 1

        profiles = np.empty(
            (
                self.number_of_edges,
                number_of_bins,
            ),
            dtype=np.uint16,
        )

        for count in range(number_of_bins):
            profiles[:, count] = np.count_nonzero(
                affected_counts == count,
                axis=1,
            )

        return profiles

    def _update_action_profiles(
        self,
        *,
        affected_cliques: NDArray[np.uint32],
        old_counts: NDArray[np.uint8],
        new_counts: NDArray[np.uint8],
    ) -> None:
        """
        Update cached profiles after one edge flip.

        Each affected clique moves from one histogram bin to an
        adjacent bin.  That move changes the action profile of every
        edge contained in the clique.  Flattening (edge, bin) to one
        integer lets two small bincount operations accumulate all
        repeated updates efficiently in compiled NumPy code.
        """
        if self._action_profiles is None:
            return

        number_of_bins = self.edges_per_clique + 1

        clique_edges = self._index.clique_edges[
            affected_cliques
        ].reshape(-1).astype(
            np.int32,
            copy=False,
        )

        edges_per_clique = self._index.edges_per_clique

        old_bins = np.repeat(
            old_counts,
            edges_per_clique,
        ).astype(
            np.int32,
            copy=False,
        )

        new_bins = np.repeat(
            new_counts,
            edges_per_clique,
        ).astype(
            np.int32,
            copy=False,
        )

        old_indexes = clique_edges * number_of_bins + old_bins

        new_indexes = clique_edges * number_of_bins + new_bins

        profile_size = self._action_profiles.size

        profile_delta = np.bincount(
            new_indexes,
            minlength=profile_size,
        ) - np.bincount(
            old_indexes,
            minlength=profile_size,
        )

        updated_profiles = (
            self._action_profiles.astype(np.int32)
            + profile_delta.reshape(self._action_profiles.shape)
        )

        if np.any(updated_profiles < 0) or np.any(
            updated_profiles > np.iinfo(np.uint16).max
        ):
            raise RuntimeError(
                "Incremental action-profile update exceeded "
                "uint16 bounds."
            )

        self._action_profiles[:] = updated_profiles.astype(
            np.uint16,
        )

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