"""Exact single-edge-flip action analysis for a search state.

An "action" in this package is the flip of one host-graph edge from its
current color to the other color. Because the exact score is the count of
monochromatic K5 cliques (``histogram[0] + histogram[-1]`` of the color-one
edge-count histogram), flipping a single edge only changes the color-one
count of the cliques incident to that edge, and only those cliques can move
between histogram bins. This module derives, for every edge, the exact
resulting change to the global histogram and the exact score reduction
(reward) that flip would produce, without recomputing the score from
scratch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .RState import RSearchState

ActionProfiles = NDArray[np.uint16]
HistogramDeltas = NDArray[np.int32]


def _owned_read_only(
    array: NDArray,
) -> NDArray:
    """Copy an array, mark the copy read-only, and return it.

    Args:
        array (NDArray): Source array; may be any array-like value.

    Returns:
        NDArray: An independently owned copy with ``flags.writeable`` set
        to ``False``, safe to expose without risking aliasing the
        caller's mutable buffers.
    """
    owned = np.asarray(array).copy()

    owned.flags.writeable = False

    return owned


@dataclass(
    frozen=True,
    slots=True,
    eq=False,
)
class RActionAnalysis:
    """Exact consequences of every single-edge flip in one state.

    An instance is a snapshot: it is tied to the exact
    ``source_state``/``state_version`` pair it was built from, and
    :meth:`applies_to` must be used to confirm it is still valid before
    reusing it against a state that may since have been mutated.

    Attributes:
        source_state (RSearchState): Search state this analysis was
            computed from.
        state_version (int): ``source_state.version`` at the time of
            computation, used to detect staleness after later mutation.
        profiles (ActionProfiles): ``uint16`` array of shape
            ``(number_of_edges, edges_per_clique + 1)``. ``profiles[e, k]``
            is the number of indexed K5s containing edge ``e`` that
            currently have exactly ``k`` color-one edges.
        histogram_deltas (HistogramDeltas): ``int32`` array of the same
            shape as ``profiles``. Row ``e`` is the exact change to the
            global color-one-count histogram that would result from
            flipping edge ``e``.
        immediate_rewards (NDArray[np.int32]): Exact score reduction for
            flipping each edge; positive values improve the score
            (fewer monochromatic K5s), negative values worsen it.
        resulting_scores (NDArray[np.int32]): Exact score that would
            result from flipping each edge, equal to
            ``source_state.score - immediate_rewards``.
    """

    source_state: RSearchState
    state_version: int

    profiles: ActionProfiles
    histogram_deltas: HistogramDeltas

    immediate_rewards: NDArray[np.int32]
    resulting_scores: NDArray[np.int32]

    def __post_init__(self) -> None:
        """Validate array shapes and freeze all array fields.

        Raises:
            ValueError: If ``profiles``, ``histogram_deltas``, or
                ``resulting_scores`` does not describe the same number
                of actions as ``immediate_rewards``.
        """
        number_of_actions = len(self.immediate_rewards)

        if self.profiles.shape[0] != number_of_actions:
            raise ValueError(
                "profiles and immediate_rewards "
                "must describe the same number "
                "of actions."
            )

        if self.histogram_deltas.shape[0] != number_of_actions:
            raise ValueError(
                "histogram_deltas and "
                "immediate_rewards must describe "
                "the same number of actions."
            )

        if self.resulting_scores.shape != (number_of_actions,):
            raise ValueError("resulting_scores has the " "wrong shape.")

        object.__setattr__(
            self,
            "state_version",
            int(self.state_version),
        )

        object.__setattr__(
            self,
            "profiles",
            _owned_read_only(self.profiles),
        )

        object.__setattr__(
            self,
            "histogram_deltas",
            _owned_read_only(self.histogram_deltas),
        )

        object.__setattr__(
            self,
            "immediate_rewards",
            _owned_read_only(self.immediate_rewards),
        )

        object.__setattr__(
            self,
            "resulting_scores",
            _owned_read_only(self.resulting_scores),
        )

    @property
    def number_of_actions(
        self,
    ) -> int:
        """int: Number of analyzed edge-flip actions (one per host edge)."""
        return len(self.immediate_rewards)

    def applies_to(
        self,
        state: RSearchState,
    ) -> bool:
        """Return whether this analysis belongs to the current state.

        Both object identity and state version are checked. Two
        different states at version zero are not interchangeable.

        Args:
            state (RSearchState): Candidate state to check freshness
                against.

        Returns:
            bool: ``True`` if ``state`` is the exact object this
            analysis was computed from and has not been mutated since.
        """
        return self.source_state is state and self.state_version == state.version


def edge_clique_profiles(
    state: RSearchState,
) -> ActionProfiles:
    """Count affected cliques by current color-one edge count.

    ``profiles[e, k]`` is the number of indexed cliques containing
    edge ``e`` that currently have exactly ``k`` color-one edges.

    ``RSearchState`` constructs this information lazily and maintains it
    incrementally after each edge flip. This function returns an owned
    copy so it retains its original value-returning behavior even though
    ``state.action_profiles`` itself is a read-only view.

    Args:
        state (RSearchState): Search state to read the action-profile
            cache from.

    Returns:
        ActionProfiles: Owned ``uint16`` copy of shape
        ``(number_of_edges, edges_per_clique + 1)``.
    """
    return state.action_profiles.copy()


def all_histogram_deltas(
    state: RSearchState,
    profiles: ActionProfiles,
) -> HistogramDeltas:
    """Calculate the global histogram delta for every edge flip.

    For each edge ``e``, this computes how the global color-one-count
    histogram would change if ``e`` were flipped, given the current
    color of ``e`` and the clique-count profile of every edge. Flipping
    a color-zero edge to color one moves each incident clique's bin
    ``k`` up to ``k + 1``; flipping a color-one edge to color zero moves
    each incident clique's bin ``k`` down to ``k - 1``. Every other bin
    for that edge simply loses the cliques that moved out of it, which
    is why the deltas start as ``-profiles``.

    Args:
        state (RSearchState): Search state supplying the current edge
            colors and expected table dimensions.
        profiles (ActionProfiles): Per-edge clique-count profile, as
            returned by :func:`edge_clique_profiles`.

    Returns:
        HistogramDeltas: ``int32`` array of shape
        ``(number_of_edges, edges_per_clique + 1)``. Row ``e`` is the
        exact change to the global histogram if edge ``e`` is flipped.

    Raises:
        ValueError: If ``profiles`` does not have the shape
            ``(state.number_of_edges, state.edges_per_clique + 1)``.
    """
    expected_shape = (
        state.number_of_edges,
        state.edges_per_clique + 1,
    )

    if profiles.shape != expected_shape:
        raise ValueError(
            f"Expected profile shape "
            f"{expected_shape}, received "
            f"{profiles.shape}."
        )

    deltas = -profiles.astype(np.int32)

    red_edges = state.colors == 0

    blue_edges = state.colors == 1

    # Color zero to color one:
    # histogram bin k moves to k + 1.
    deltas[
        red_edges,
        1:,
    ] += profiles[
        red_edges,
        :-1,
    ]

    # Color one to color zero:
    # histogram bin k moves to k - 1.
    deltas[
        blue_edges,
        :-1,
    ] += profiles[
        blue_edges,
        1:,
    ]

    return deltas


def all_immediate_rewards(
    histogram_deltas: HistogramDeltas,
) -> NDArray[np.int32]:
    """Return the exact score reduction for every histogram delta.

    The exact score is ``histogram[0] + histogram[-1]`` (the number of
    all-color-zero plus all-color-one K5s), so the score change from a
    flip is the sum of its histogram delta's first and last bins; the
    reward is the negation of that change so a positive value means the
    score improved.

    Args:
        histogram_deltas (HistogramDeltas): Per-action histogram deltas,
            as returned by :func:`all_histogram_deltas`.

    Returns:
        NDArray[np.int32]: Exact score reduction for every action.

    Raises:
        ValueError: If ``histogram_deltas`` is not two-dimensional.
    """
    if histogram_deltas.ndim != 2:
        raise ValueError("histogram_deltas must be " "two-dimensional.")

    score_changes = histogram_deltas[:, 0] + histogram_deltas[:, -1]

    return -score_changes


def immediate_rewards_for_edges(
    state: RSearchState,
    edge_indices: NDArray[np.integer],
) -> NDArray[np.int32]:
    """Calculate exact score reductions for selected edge flips.

    This is the fast path used when only a subset of actions is
    needed: it avoids constructing the complete action-profile and
    histogram-delta tables and instead reads
    ``state.color_one_counts`` directly for just the cliques incident
    to the requested edges.

    For a color-zero (red) edge, flipping it to color one destroys
    every incident clique currently at count zero (all-red) and creates
    a new all-blue clique for every incident clique currently at count
    ``edges_per_clique - 1``. For a color-one (blue) edge, flipping it
    to color zero destroys every incident clique currently at count
    ``edges_per_clique`` (all-blue) and creates a new all-red clique for
    every incident clique currently at count one. The reward is
    destroyed-count minus created-count in each case.

    Args:
        state (RSearchState): Search state to read colors and per-clique
            color-one counts from.
        edge_indices (NDArray[np.integer]): One-dimensional array of
            host-edge indices to evaluate.

    Returns:
        NDArray[np.int32]: Exact score reduction for each requested
        edge, in the same order as ``edge_indices``.

    Raises:
        ValueError: If ``edge_indices`` is not one-dimensional.
        IndexError: If ``edge_indices`` contains an out-of-range edge.
    """
    edge_indices = np.asarray(
        edge_indices,
        dtype=np.int32,
    )

    if edge_indices.ndim != 1:
        raise ValueError("edge_indices must be " "one-dimensional.")

    if edge_indices.size == 0:
        return np.empty(
            0,
            dtype=np.int32,
        )

    if np.any(edge_indices < 0) or np.any(edge_indices >= state.number_of_edges):
        raise IndexError("edge_indices contains " "an invalid edge.")

    affected_counts = state.color_one_counts[state.index.edge_to_cliques[edge_indices]]

    selected_colors = state.colors[edge_indices]

    rewards = np.empty(
        len(edge_indices),
        dtype=np.int32,
    )

    red_edges = selected_colors == 0

    blue_edges = selected_colors == 1

    if np.any(red_edges):
        red_counts = affected_counts[red_edges]

        destroyed_red_cliques = np.count_nonzero(
            red_counts == 0,
            axis=1,
        )

        created_blue_cliques = np.count_nonzero(
            red_counts == state.edges_per_clique - 1,
            axis=1,
        )

        rewards[red_edges] = destroyed_red_cliques - created_blue_cliques

    if np.any(blue_edges):
        blue_counts = affected_counts[blue_edges]

        destroyed_blue_cliques = np.count_nonzero(
            blue_counts == state.edges_per_clique,
            axis=1,
        )

        created_red_cliques = np.count_nonzero(
            blue_counts == 1,
            axis=1,
        )

        rewards[blue_edges] = destroyed_blue_cliques - created_red_cliques

    return rewards


def analyze_actions(
    state: RSearchState,
) -> RActionAnalysis:
    """Calculate the exact consequences of every edge flip.

    This is the top-level entry point that assembles action profiles,
    histogram deltas, immediate rewards, and resulting scores into one
    :class:`RActionAnalysis` snapshot of ``state``.

    Args:
        state (RSearchState): Search state to analyze.

    Returns:
        RActionAnalysis: Exact per-edge consequences of flipping each
        edge of ``state``.
    """
    profiles = edge_clique_profiles(state)

    histogram_deltas = all_histogram_deltas(
        state,
        profiles,
    )

    immediate_rewards = all_immediate_rewards(histogram_deltas)

    resulting_scores = (state.score - immediate_rewards).astype(np.int32)

    return RActionAnalysis(
        source_state=state,
        state_version=state.version,
        profiles=profiles,
        histogram_deltas=histogram_deltas,
        immediate_rewards=immediate_rewards,
        resulting_scores=resulting_scores,
    )


def resulting_histograms(
    state: RSearchState,
    analysis: RActionAnalysis,
) -> NDArray[np.int64]:
    """Return the complete resulting histogram for every action.

    Args:
        state (RSearchState): Search state ``analysis`` must still
            describe.
        analysis (RActionAnalysis): Previously computed action analysis
            for ``state``.

    Returns:
        NDArray[np.int64]: Array of shape
        ``(number_of_actions, edges_per_clique + 1)`` where row ``e``
        is the complete color-one-count histogram that would result
        from flipping edge ``e``, computed as ``state.histogram +
        analysis.histogram_deltas``.

    Raises:
        ValueError: If ``analysis`` does not apply to ``state`` (see
            :meth:`RActionAnalysis.applies_to`).
    """
    if not analysis.applies_to(state):
        raise ValueError("Action analysis belongs to " "an earlier state version.")

    return (
        state.histogram[
            np.newaxis,
            :,
        ]
        + analysis.histogram_deltas
    )