"""Read-only causal instrumentation for one two-edge Ramsey flip."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from .RAction import analyze_actions
from .REdgeFlipCausalAnalysis import (
    RMonochromaticCliqueChange,
    RMonochromaticParticipation,
    _monochromatic_color,
    _owned_read_only,
    monochromatic_participation,
)
from .RState import RSearchState


@dataclass(frozen=True, slots=True, eq=False)
class RTwoEdgeFlipCausalAnalysis:
    """Complete read-only causal footprint of two specified edge flips.

    Produced by :func:`analyze_two_edge_flip_causality`, which flips a
    copy of a search state at two distinct edges and records the state
    exactly before and after both flips, plus the exact set of
    monochromatic K5s created or destroyed by the pair. Unlike
    :class:`REdgeFlipCausalAnalysis.REdgeFlipCausalAnalysis`, this
    measures the *net* effect of the pair applied together: only the
    original and final states are compared, so events caused solely by
    the first flip and later undone or altered by the second are not
    reported individually — :attr:`interaction_reward` isolates the
    portion of the combined reward that is not explained by the two
    flips' independent, single-edge rewards.

    Attributes:
        edges (tuple[int, int]): Indices of the two flipped host-graph
            edges.
        endpoints (tuple[tuple[int, int], tuple[int, int]]): Vertex
            indices of each flipped edge.
        old_colors (tuple[int, int]): Edge colors before the flips, one
            per edge in ``edges``.
        new_colors (tuple[int, int]): Edge colors after the flips (the
            complement of each corresponding ``old_colors`` entry).
        score_before (int): Exact score of the state before either flip.
        score_after (int): Exact score of the state after both flips.
        participation_before (RMonochromaticParticipation): Exact
            monochromatic K5 participation counts before the flips.
        participation_after (RMonochromaticParticipation): Exact
            monochromatic K5 participation counts after the flips.
        clique_changes (tuple[RMonochromaticCliqueChange, ...]): Exact
            net event decomposition of every K5 created or destroyed
            comparing the original state directly to the final state.
        greedy_rewards_before (numpy.ndarray): Read-only ``int32`` array
            of the exact immediate reward for every edge, evaluated
            before either flip.
        greedy_rewards_after (numpy.ndarray): Read-only ``int32`` array
            of the exact immediate reward for every edge, evaluated
            after both flips.
        individual_rewards (tuple[int, int]): Exact immediate reward
            each edge would have produced flipped alone from the
            original state, one per edge in ``edges``.
    """

    edges: tuple[int, int]
    endpoints: tuple[tuple[int, int], tuple[int, int]]
    old_colors: tuple[int, int]
    new_colors: tuple[int, int]
    score_before: int
    score_after: int
    participation_before: RMonochromaticParticipation
    participation_after: RMonochromaticParticipation
    clique_changes: tuple[RMonochromaticCliqueChange, ...]
    greedy_rewards_before: NDArray[np.int32]
    greedy_rewards_after: NDArray[np.int32]
    individual_rewards: tuple[int, int]

    def __post_init__(self) -> None:
        """Validate and normalize all fields, then freeze the reward arrays.

        Raises:
            ValueError: If ``edges`` does not contain two distinct edge
                indices, if ``endpoints`` does not contain two vertex
                pairs, if ``old_colors``/``new_colors`` does not contain
                two binary colors, if a ``new_colors`` entry is not the
                complement of the matching ``old_colors`` entry, if
                ``individual_rewards`` does not contain two values, or
                if ``greedy_rewards_before`` and ``greedy_rewards_after``
                do not have equal shape.
        """
        edges = tuple(int(edge) for edge in self.edges)
        endpoints = tuple(
            tuple(int(vertex) for vertex in pair)
            for pair in self.endpoints
        )
        old_colors = tuple(int(color) for color in self.old_colors)
        new_colors = tuple(int(color) for color in self.new_colors)
        individual_rewards = tuple(
            int(reward)
            for reward in self.individual_rewards
        )

        if len(edges) != 2 or edges[0] == edges[1]:
            raise ValueError("edges must contain two distinct edge indices.")

        if len(endpoints) != 2 or any(len(pair) != 2 for pair in endpoints):
            raise ValueError("endpoints must contain two vertex pairs.")

        if len(old_colors) != 2 or any(color not in (0, 1) for color in old_colors):
            raise ValueError("old_colors must contain two binary colors.")

        if len(new_colors) != 2 or any(color not in (0, 1) for color in new_colors):
            raise ValueError("new_colors must contain two binary colors.")

        if any(new != 1 - old for old, new in zip(old_colors, new_colors)):
            raise ValueError("Each new color must be the complement of its old color.")

        if len(individual_rewards) != 2:
            raise ValueError("individual_rewards must contain two values.")

        before_rewards = _owned_read_only(
            self.greedy_rewards_before,
            dtype=np.int32,
        )
        after_rewards = _owned_read_only(
            self.greedy_rewards_after,
            dtype=np.int32,
        )

        if before_rewards.shape != after_rewards.shape:
            raise ValueError(
                "Before and after greedy reward arrays must have equal shape."
            )

        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "endpoints", endpoints)
        object.__setattr__(self, "old_colors", old_colors)
        object.__setattr__(self, "new_colors", new_colors)
        object.__setattr__(self, "score_before", int(self.score_before))
        object.__setattr__(self, "score_after", int(self.score_after))
        object.__setattr__(self, "individual_rewards", individual_rewards)
        object.__setattr__(self, "greedy_rewards_before", before_rewards)
        object.__setattr__(self, "greedy_rewards_after", after_rewards)

    @property
    def exact_reward(self) -> int:
        """int: Exact score improvement produced by flipping both edges together."""
        return self.score_before - self.score_after

    @property
    def interaction_reward(self) -> int:
        """int: Pair reward beyond the sum of the two independent single-edge rewards.

        Positive values mean the pair together does strictly better
        than the two flips would independently; negative values mean
        the flips interfere with each other (e.g. one flip destroys a
        K5 that the other flip would otherwise also have destroyed).
        """
        return self.exact_reward - sum(self.individual_rewards)

    @property
    def shared_vertex(self) -> bool:
        """bool: Whether the two host edges share an endpoint."""
        return bool(set(self.endpoints[0]) & set(self.endpoints[1]))

    @property
    def vertex_participation_delta(self) -> NDArray[np.int32]:
        """numpy.ndarray: Per-vertex, per-color change in monochromatic K5 participation."""
        return (
            self.participation_after.vertices
            - self.participation_before.vertices
        )

    @property
    def edge_participation_delta(self) -> NDArray[np.int32]:
        """numpy.ndarray: Per-edge, per-color change in monochromatic K5 participation."""
        return (
            self.participation_after.edges
            - self.participation_before.edges
        )

    @property
    def greedy_reward_delta(self) -> NDArray[np.int32]:
        """numpy.ndarray: Per-edge change in exact immediate reward caused by the pair."""
        return self.greedy_rewards_after - self.greedy_rewards_before

    @property
    def changed_vertices(self) -> NDArray[np.int32]:
        """numpy.ndarray: Indices of vertices whose monochromatic K5 participation changed."""
        changed = np.any(
            self.vertex_participation_delta != 0,
            axis=1,
        )
        return np.flatnonzero(changed).astype(np.int32)

    @property
    def changed_structure_edges(self) -> NDArray[np.int32]:
        """numpy.ndarray: Sorted unique indices of host edges touched by any clique change."""
        if not self.clique_changes:
            return np.empty(0, dtype=np.int32)

        return np.unique(
            np.concatenate(
                [change.edges for change in self.clique_changes]
            )
        ).astype(np.int32)

    @property
    def vertex_event_counts(self) -> NDArray[np.int32]:
        """numpy.ndarray: Number of net clique-change events touching each vertex.

        Counts every creation and destruction event in
        :attr:`clique_changes` regardless of color, so a vertex involved
        in both a destroyed red K5 and a created blue K5 scores two
        events even though its net delta may be zero.
        """
        result = np.zeros(
            len(self.participation_before.vertices),
            dtype=np.int32,
        )

        for change in self.clique_changes:
            result[change.vertices] += 1

        return result

    @property
    def edge_event_counts(self) -> NDArray[np.int32]:
        """numpy.ndarray: Number of net clique-change events touching each host edge.

        Counts events the same way as :attr:`vertex_event_counts`, but
        indexed by host edge rather than vertex.
        """
        result = np.zeros(
            len(self.participation_before.edges),
            dtype=np.int32,
        )

        for change in self.clique_changes:
            result[change.edges] += 1

        return result


def analyze_two_edge_flip_causality(
    state: RSearchState,
    first_edge: int,
    second_edge: int,
) -> RTwoEdgeFlipCausalAnalysis:
    """
    Analyze two simultaneous edge flips without mutating the supplied state.

    Copies ``state``, flips ``first_edge`` and then ``second_edge`` on
    the copy, and records the exact score, monochromatic participation,
    and per-edge immediate-reward landscape before either flip and
    after both. Clique changes are measured from the original state
    directly to the final state. Intermediate events caused by only the
    first flip are intentionally excluded, so the returned event
    decomposition is the exact net causal footprint of the pair (see
    :attr:`RTwoEdgeFlipCausalAnalysis.interaction_reward` for the
    departure from the sum of the two flips' independent rewards).

    Args:
        state (RSearchState): Search state to analyze. Not mutated;
            the flips are applied to an internal copy.
        first_edge (int): Index of the first host-graph edge to flip.
        second_edge (int): Index of the second host-graph edge to flip.
            Must differ from ``first_edge``.

    Returns:
        RTwoEdgeFlipCausalAnalysis: Complete net causal footprint of
        flipping both ``first_edge`` and ``second_edge`` from ``state``.

    Raises:
        TypeError: If ``first_edge`` or ``second_edge`` is not an
            integer.
        IndexError: If ``first_edge`` or ``second_edge`` is outside the
            host graph.
        ValueError: If ``first_edge`` and ``second_edge`` are equal.
        RuntimeError: If the internal event-decomposition consistency
            check fails (see :func:`_verify_event_decomposition`).
    """
    first_edge = _validated_edge(
        state,
        first_edge,
        name="first_edge",
    )
    second_edge = _validated_edge(
        state,
        second_edge,
        name="second_edge",
    )

    if first_edge == second_edge:
        raise ValueError("first_edge and second_edge must be distinct.")

    edges = (first_edge, second_edge)
    working_state = state.copy()
    index = working_state.index

    old_colors = tuple(
        int(working_state.colors[edge])
        for edge in edges
    )
    new_colors = tuple(1 - color for color in old_colors)
    endpoints = tuple(
        tuple(
            int(vertex)
            for vertex in working_state.graph.edges[edge]
        )
        for edge in edges
    )

    score_before = working_state.score
    participation_before = monochromatic_participation(
        working_state
    )
    greedy_rewards_before = analyze_actions(
        working_state
    ).immediate_rewards.copy()
    individual_rewards = tuple(
        int(greedy_rewards_before[edge])
        for edge in edges
    )

    affected_cliques = np.union1d(
        index.edge_to_cliques[first_edge],
        index.edge_to_cliques[second_edge],
    )
    counts_before = working_state.color_one_counts[
        affected_cliques
    ].copy()

    working_state.apply_edge_flip(first_edge)
    working_state.apply_edge_flip(second_edge)

    counts_after = working_state.color_one_counts[
        affected_cliques
    ]

    changes: list[RMonochromaticCliqueChange] = []

    for local_index, clique_index_value in enumerate(affected_cliques):
        clique_index = int(clique_index_value)
        before_count = int(counts_before[local_index])
        after_count = int(counts_after[local_index])

        before_color = _monochromatic_color(
            before_count,
            working_state.edges_per_clique,
        )
        after_color = _monochromatic_color(
            after_count,
            working_state.edges_per_clique,
        )

        if before_color == after_color:
            continue

        clique_edges = index.clique_edges[clique_index]
        clique_vertices = np.unique(
            working_state.graph.edges[
                clique_edges
            ].reshape(-1)
        ).astype(np.uint8)

        if before_color is not None:
            changes.append(
                RMonochromaticCliqueChange(
                    clique_index=clique_index,
                    color=before_color,
                    delta=-1,
                    vertices=clique_vertices,
                    edges=clique_edges,
                )
            )

        if after_color is not None:
            changes.append(
                RMonochromaticCliqueChange(
                    clique_index=clique_index,
                    color=after_color,
                    delta=1,
                    vertices=clique_vertices,
                    edges=clique_edges,
                )
            )

    participation_after = monochromatic_participation(
        working_state
    )
    greedy_rewards_after = analyze_actions(
        working_state
    ).immediate_rewards.copy()

    result = RTwoEdgeFlipCausalAnalysis(
        edges=edges,
        endpoints=endpoints,
        old_colors=old_colors,
        new_colors=new_colors,
        score_before=score_before,
        score_after=working_state.score,
        participation_before=participation_before,
        participation_after=participation_after,
        clique_changes=tuple(changes),
        greedy_rewards_before=greedy_rewards_before,
        greedy_rewards_after=greedy_rewards_after,
        individual_rewards=individual_rewards,
    )

    _verify_event_decomposition(result)

    return result


def _validated_edge(
    state: RSearchState,
    edge: int,
    *,
    name: str,
) -> int:
    """Validate that ``edge`` is an in-range integer edge index.

    Args:
        state (RSearchState): Search state defining the valid edge
            range.
        edge (int): Candidate edge index to validate.
        name (str): Parameter name to use in error messages.

    Returns:
        int: ``edge`` coerced to a plain ``int``.

    Raises:
        TypeError: If ``edge`` is not an integer (or is a ``bool``).
        IndexError: If ``edge`` is outside the host graph.
    """
    if isinstance(edge, bool) or not isinstance(edge, Integral):
        raise TypeError(f"{name} must be an integer.")

    edge = int(edge)

    if edge < 0 or edge >= state.number_of_edges:
        raise IndexError(f"{name} is outside the host graph.")

    return edge


def _verify_event_decomposition(
    analysis: RTwoEdgeFlipCausalAnalysis,
) -> None:
    """Verify that pair events exactly reconstruct both load deltas.

    Replays every recorded :class:`RMonochromaticCliqueChange` event
    and checks that the resulting per-vertex and per-edge deltas match
    :attr:`RTwoEdgeFlipCausalAnalysis.vertex_participation_delta` and
    :attr:`RTwoEdgeFlipCausalAnalysis.edge_participation_delta` exactly.

    Args:
        analysis (RTwoEdgeFlipCausalAnalysis): Analysis whose event
            decomposition should be checked for consistency.

    Raises:
        RuntimeError: If replaying the clique-change events does not
            exactly reconstruct either the vertex or the edge
            participation delta.
    """
    vertex_delta = np.zeros_like(
        analysis.vertex_participation_delta
    )
    edge_delta = np.zeros_like(
        analysis.edge_participation_delta
    )

    for change in analysis.clique_changes:
        vertex_delta[
            change.vertices,
            change.color,
        ] += change.delta
        edge_delta[
            change.edges,
            change.color,
        ] += change.delta

    if not np.array_equal(
        vertex_delta,
        analysis.vertex_participation_delta,
    ):
        raise RuntimeError(
            "Causal K5 events do not reconstruct the vertex delta."
        )

    if not np.array_equal(
        edge_delta,
        analysis.edge_participation_delta,
    ):
        raise RuntimeError(
            "Causal K5 events do not reconstruct the edge delta."
        )
