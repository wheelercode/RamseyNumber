"""Mutable incremental state for one active search attempt.

This module defines :class:`RSearchState`, the single mutable object a
search algorithm drives forward one edge flip at a time. Every mutation
touches only the cliques incident to the flipped edge, so the coloring,
per-clique color-one counts, global histogram, exact score, and (when in
use) the action-profile cache are all updated incrementally rather than
recomputed from scratch after each move.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .RColoring import RColoring
from .RGraph import RGraph, RSubgraphIndex
from .RScoring import count_color_edges_per_clique


class RSearchState:
    """Incrementally maintained state for symmetric two-color search.

    ``RSearchState`` is the sole owner of mutable coloring, clique-count,
    histogram, and score data for one search attempt. It currently
    supports only symmetric two-color problems (both colors forbid the
    same clique size), since the score is derived from a single
    color-one-count histogram whose two ends (``histogram[0]`` and
    ``histogram[-1]``) count the all-color-zero and all-color-one
    cliques respectively.

    Every exposed array property returns a read-only view that shares
    memory with the state's internal buffers; callers must not attempt
    to mutate them, and the view is only valid until the next mutating
    call, after which it reflects the new data (or in the case of a
    view into a *replaced* buffer, would no longer track it).
    """

    def __init__(
        self,
        coloring: RColoring,
    ) -> None:
        """Build a search state initialized to a starting coloring.

        Computes per-clique color-one counts, the color-one-count
        histogram, and the exact score from ``coloring``. The
        action-profile cache is left unbuilt (``None``) until first
        requested, so search paths that never need it avoid its
        construction cost.

        Args:
            coloring (RColoring): Starting coloring to copy into the
                new mutable state. The state takes an independent
                mutable copy of the colors; ``coloring`` itself is
                unaffected by later mutations.

        Raises:
            ValueError: If ``coloring.graph.problem`` is not a
                symmetric two-color problem.
        """
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
        """RGraph: Immutable host-graph topology this state colors."""
        return self._graph

    @property
    def index(self) -> RSubgraphIndex:
        """RSubgraphIndex: Precomputed clique-incidence index maintained by this state."""
        return self._index

    @property
    def clique_size(self) -> int:
        """int: Forbidden clique size maintained by this state."""
        return self._clique_size

    @property
    def edges_per_clique(self) -> int:
        """int: Number of edges in each maintained clique."""
        return self._index.edges_per_clique

    @property
    def number_of_edges(self) -> int:
        """int: Number of available edge-flip actions (host-graph edges)."""
        return self._graph.number_of_edges

    @property
    def score(self) -> int:
        """int: Current exact monochromatic-clique score."""
        return self._score

    @property
    def version(self) -> int:
        """int: Number of mutations applied to this state so far.

        Used by cached analyses (e.g. :class:`ramsey.RAction.RActionAnalysis`)
        to detect staleness after a state has been mutated.
        """
        return self._version

    @property
    def colors(
        self,
    ) -> NDArray[np.uint8]:
        """NDArray[np.uint8]: Read-only view of the current edge colors.

        Shape ``(number_of_edges,)``; entry ``e`` is the color (``0`` or
        ``1``) currently assigned to host edge ``e``. The view shares
        memory with this state's internal buffer and becomes stale after
        the next mutation.
        """
        return self._read_only_view(self._colors)

    @property
    def color_one_counts(
        self,
    ) -> NDArray[np.uint8]:
        """NDArray[np.uint8]: Read-only view of per-clique color-one counts.

        Shape ``(clique_count,)``; entry ``c`` is the number of
        color-one edges currently in indexed clique ``c``.
        """
        return self._read_only_view(self._color_one_counts)

    @property
    def histogram(
        self,
    ) -> NDArray[np.int64]:
        """NDArray[np.int64]: Read-only view of the current count histogram.

        Shape ``(edges_per_clique + 1,)``; bin ``k`` is the number of
        indexed cliques currently containing exactly ``k`` color-one
        edges. ``score`` equals ``histogram[0] + histogram[-1]``.
        """
        return self._read_only_view(self._histogram)

    @property
    def action_profiles(
        self,
    ) -> NDArray[np.uint16]:
        """NDArray[np.uint16]: Incrementally maintained all-edge clique profiles.

        ``profiles[e, k]`` is the number of indexed cliques containing
        edge ``e`` that currently contain exactly ``k`` color-one edges.
        Shape is ``(number_of_edges, edges_per_clique + 1)``.

        The profiles are constructed only on first use (via
        :meth:`_build_action_profiles`). Once built, every edge flip
        updates only the profile contributions of cliques affected by
        that flip (via :meth:`_update_action_profiles`), rather than
        rebuilding the table. The returned array is a read-only view
        that becomes stale after the next mutation.
        """
        if self._action_profiles is None:
            self._action_profiles = self._build_action_profiles()

        return self._read_only_view(self._action_profiles)

    def coloring_snapshot(
        self,
    ) -> RColoring:
        """Return an immutable coloring containing the current state.

        Returns:
            RColoring: New immutable coloring holding a copy of the
            current edge colors, decoupled from further mutations of
            this state.
        """
        return RColoring(
            graph=self._graph,
            colors=self._colors,
        )

    def copy(
        self,
    ) -> "RSearchState":
        """Return an independent state with the same current coloring.

        The copied state begins at version zero because it has its own
        independent mutation history; it is not comparable via
        :meth:`ramsey.RAction.RActionAnalysis.applies_to` to analyses
        built from ``self``.

        Returns:
            RSearchState: New state, independently mutable, initialized
            to a copy of this state's current coloring.
        """
        return RSearchState(self.coloring_snapshot())

    def apply_edge_flip(
        self,
        edge: int,
    ) -> int:
        """Flip one edge and return the exact score reduction.

        Recolors ``edge`` to the other color, then updates, for every
        clique incident to ``edge``, the color-one count, the global
        histogram, the exact score, and (if already built) the
        action-profile cache — all incrementally, without touching any
        clique not incident to ``edge``.

        Args:
            edge (int): Index of the host edge to flip.

        Returns:
            int: Exact score reduction from this flip
            (``old_score - new_score``). A positive result means the
            score improved (fewer monochromatic K5s); a negative result
            means it became worse.

        Raises:
            IndexError: If ``edge`` is out of range.
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
        """Recolor several distinct edges and return exact score reduction.

        Only edges whose requested color differs from the current color
        are mutated.

        Each changed edge goes through :meth:`apply_edge_flip`,
        preserving all incrementally maintained state:

        - edge colors
        - K5 color counts
        - histogram
        - monochromatic score
        - action-profile cache
        - state version

        Args:
            edges (NDArray[np.integer]): One-dimensional array of
                distinct host-edge indices to recolor.
            colors (NDArray[np.integer]): One-dimensional array, the
                same length as ``edges``, of requested colors (``0`` or
                ``1``) for each edge.

        Returns:
            int: Exact total score reduction from all resulting flips
            (``old_score - new_score``).

        Raises:
            ValueError: If ``edges`` or ``colors`` is not
                one-dimensional, if their lengths differ, if ``edges``
                contains duplicates, or if ``colors`` contains a value
                other than ``0`` or ``1``.
            TypeError: If ``edges`` or ``colors`` does not have an
                integer dtype.
            IndexError: If ``edges`` contains an out-of-range edge
                index.
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
        """Construct the complete action-profile cache from current state.

        This is the expensive full reconstruction: for every edge it
        counts, across all cliques incident to that edge, how many fall
        into each color-one-count bin. It occurs at most once per
        ``RSearchState`` instance; subsequent mutations use the
        incremental update path in :meth:`_update_action_profiles`.

        Returns:
            NDArray[np.uint16]: Owned array of shape
            ``(number_of_edges, edges_per_clique + 1)``.

        Raises:
            ValueError: If the number of cliques per edge exceeds the
                ``uint16`` capacity of the action-profile cache.
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
        """Update cached action profiles in place after one edge flip.

        Each affected clique moves from one histogram bin to an
        adjacent bin. That move changes the action profile of every
        edge contained in the clique. Flattening ``(edge, bin)`` to one
        integer lets two small ``bincount`` operations accumulate all
        repeated updates efficiently in compiled NumPy code, avoiding a
        Python-level loop over affected cliques.

        Args:
            affected_cliques (NDArray[np.uint32]): Indices of the
                cliques whose color-one count changed.
            old_counts (NDArray[np.uint8]): Color-one count of each
                affected clique before the flip.
            new_counts (NDArray[np.uint8]): Color-one count of each
                affected clique after the flip.

        Raises:
            RuntimeError: If the updated profile values would fall
                outside the valid ``uint16`` range, indicating a
                corrupted incremental-update invariant.
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
        """Validate an encoded edge index.

        Args:
            edge (int): Edge index to validate.

        Raises:
            IndexError: If ``edge`` is negative or not less than
                ``number_of_edges``.
        """
        if edge < 0 or edge >= self.number_of_edges:
            raise IndexError(f"Invalid edge index: {edge}")

    @staticmethod
    def _read_only_view(
        array: NDArray,
    ) -> NDArray:
        """Return a non-writeable view without copying array data.

        Args:
            array (NDArray): Owned array to expose read-only.

        Returns:
            NDArray: View sharing memory with ``array``, with
            ``flags.writeable`` set to ``False``.
        """
        view = array.view()
        view.flags.writeable = False

        return view