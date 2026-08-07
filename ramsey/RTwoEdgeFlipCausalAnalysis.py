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
    """Complete read-only causal footprint of two specified edge flips."""

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
        """Score improvement produced by flipping both edges together."""
        return self.score_before - self.score_after

    @property
    def interaction_reward(self) -> int:
        """Pair reward beyond the sum of the two independent rewards."""
        return self.exact_reward - sum(self.individual_rewards)

    @property
    def shared_vertex(self) -> bool:
        """Whether the two host edges share an endpoint."""
        return bool(set(self.endpoints[0]) & set(self.endpoints[1]))

    @property
    def vertex_participation_delta(self) -> NDArray[np.int32]:
        return (
            self.participation_after.vertices
            - self.participation_before.vertices
        )

    @property
    def edge_participation_delta(self) -> NDArray[np.int32]:
        return (
            self.participation_after.edges
            - self.participation_before.edges
        )

    @property
    def greedy_reward_delta(self) -> NDArray[np.int32]:
        return self.greedy_rewards_after - self.greedy_rewards_before

    @property
    def changed_vertices(self) -> NDArray[np.int32]:
        changed = np.any(
            self.vertex_participation_delta != 0,
            axis=1,
        )
        return np.flatnonzero(changed).astype(np.int32)

    @property
    def changed_structure_edges(self) -> NDArray[np.int32]:
        if not self.clique_changes:
            return np.empty(0, dtype=np.int32)

        return np.unique(
            np.concatenate(
                [change.edges for change in self.clique_changes]
            )
        ).astype(np.int32)

    @property
    def vertex_event_counts(self) -> NDArray[np.int32]:
        result = np.zeros(
            len(self.participation_before.vertices),
            dtype=np.int32,
        )

        for change in self.clique_changes:
            result[change.vertices] += 1

        return result

    @property
    def edge_event_counts(self) -> NDArray[np.int32]:
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

    Clique changes are measured from the original state directly to the final
    state. Intermediate events caused by only the first flip are intentionally
    excluded, so the returned event decomposition is the exact net causal
    footprint of the pair.
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
    if isinstance(edge, bool) or not isinstance(edge, Integral):
        raise TypeError(f"{name} must be an integer.")

    edge = int(edge)

    if edge < 0 or edge >= state.number_of_edges:
        raise IndexError(f"{name} is outside the host graph.")

    return edge


def _verify_event_decomposition(
    analysis: RTwoEdgeFlipCausalAnalysis,
) -> None:
    """Verify that pair events exactly reconstruct both load deltas."""
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
