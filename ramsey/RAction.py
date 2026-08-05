"""Consequences and exact score changes for candidate search actions."""

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
    """
    Copy an array, mark the copy read-only, and return it.
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
    """
    Exact consequences of every single-edge flip in one state.
    """

    source_state: RSearchState
    state_version: int

    profiles: ActionProfiles
    histogram_deltas: HistogramDeltas

    immediate_rewards: NDArray[np.int32]
    resulting_scores: NDArray[np.int32]

    def __post_init__(self) -> None:
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
        """
        Return the number of analyzed edge flips.
        """
        return len(self.immediate_rewards)

    def applies_to(
        self,
        state: RSearchState,
    ) -> bool:
        """
        Return whether this analysis belongs to the current state.

        Both object identity and state version are checked. Two
        different states at version zero are not interchangeable.
        """
        return self.source_state is state and self.state_version == state.version


def edge_clique_profiles(
    state: RSearchState,
) -> ActionProfiles:
    """
    Count affected cliques by current color-one edge count.

    profiles[e, k] is the number of indexed cliques containing
    edge e that currently have exactly k color-one edges.

    RSearchState constructs this information lazily and maintains it
    incrementally after each edge flip.  Return an owned copy so this
    function retains its original value-returning behavior.
    """
    return state.action_profiles.copy()


def all_histogram_deltas(
    state: RSearchState,
    profiles: ActionProfiles,
) -> HistogramDeltas:
    """
    Calculate the global histogram delta for every edge flip.
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
    """
    Return the exact score reduction for every histogram delta.
    """
    if histogram_deltas.ndim != 2:
        raise ValueError("histogram_deltas must be " "two-dimensional.")

    score_changes = histogram_deltas[:, 0] + histogram_deltas[:, -1]

    return -score_changes


def immediate_rewards_for_edges(
    state: RSearchState,
    edge_indices: NDArray[np.integer],
) -> NDArray[np.int32]:
    """
    Calculate exact score reductions for selected edge flips.

    This is the fast path. It avoids constructing complete profiles
    and histogram deltas for every action.
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
    """
    Calculate the exact consequences of every edge flip.
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
    """
    Return the complete resulting histogram for every action.
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