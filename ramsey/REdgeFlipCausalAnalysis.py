"""Read-only causal instrumentation for one single-edge Ramsey flip."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from .RAction import analyze_actions
from .RState import RSearchState


def _owned_read_only(
    array: NDArray,
    *,
    dtype=None,
) -> NDArray:
    """Return an owned immutable NumPy array."""
    result = np.asarray(
        array,
        dtype=dtype,
    ).copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True, eq=False)
class RMonochromaticParticipation:
    """
    Monochromatic-clique participation by vertex and host edge.

    The final axis is the edge color:

        [..., 0] = color-zero / red K5 participation
        [..., 1] = color-one / blue K5 participation
    """

    vertices: NDArray[np.int32]
    edges: NDArray[np.int32]

    def __post_init__(self) -> None:
        vertices = _owned_read_only(
            self.vertices,
            dtype=np.int32,
        )
        edges = _owned_read_only(
            self.edges,
            dtype=np.int32,
        )

        if vertices.ndim != 2 or vertices.shape[1] != 2:
            raise ValueError("vertices must have shape (n_vertices, 2).")

        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("edges must have shape (n_edges, 2).")

        if np.any(vertices < 0) or np.any(edges < 0):
            raise ValueError("Participation counts cannot be negative.")

        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "edges", edges)

    @property
    def red_vertices(self) -> NDArray[np.int32]:
        return self.vertices[:, 0]

    @property
    def blue_vertices(self) -> NDArray[np.int32]:
        return self.vertices[:, 1]

    @property
    def red_edges(self) -> NDArray[np.int32]:
        return self.edges[:, 0]

    @property
    def blue_edges(self) -> NDArray[np.int32]:
        return self.edges[:, 1]


@dataclass(frozen=True, slots=True, eq=False)
class RMonochromaticCliqueChange:
    """One monochromatic K5 created or destroyed by the edge flip."""

    clique_index: int
    color: int
    delta: int
    vertices: NDArray[np.uint8]
    edges: NDArray[np.uint16]

    def __post_init__(self) -> None:
        if self.color not in (0, 1):
            raise ValueError("color must be zero or one.")

        if self.delta not in (-1, 1):
            raise ValueError("delta must be -1 or +1.")

        vertices = _owned_read_only(
            self.vertices,
            dtype=np.uint8,
        )
        edges = _owned_read_only(
            self.edges,
            dtype=np.uint16,
        )

        if vertices.ndim != 1:
            raise ValueError("vertices must be one-dimensional.")

        if edges.ndim != 1:
            raise ValueError("edges must be one-dimensional.")

        object.__setattr__(self, "clique_index", int(self.clique_index))
        object.__setattr__(self, "color", int(self.color))
        object.__setattr__(self, "delta", int(self.delta))
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "edges", edges)

    @property
    def created(self) -> bool:
        return self.delta == 1

    @property
    def destroyed(self) -> bool:
        return self.delta == -1


@dataclass(frozen=True, slots=True, eq=False)
class REdgeFlipCausalAnalysis:
    """Complete read-only causal footprint of one specified edge flip."""

    edge: int
    endpoints: tuple[int, int]
    old_color: int
    new_color: int
    score_before: int
    score_after: int
    participation_before: RMonochromaticParticipation
    participation_after: RMonochromaticParticipation
    clique_changes: tuple[RMonochromaticCliqueChange, ...]
    greedy_rewards_before: NDArray[np.int32]
    greedy_rewards_after: NDArray[np.int32]

    def __post_init__(self) -> None:
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

        object.__setattr__(self, "edge", int(self.edge))
        object.__setattr__(self, "old_color", int(self.old_color))
        object.__setattr__(self, "new_color", int(self.new_color))
        object.__setattr__(self, "score_before", int(self.score_before))
        object.__setattr__(self, "score_after", int(self.score_after))
        object.__setattr__(self, "greedy_rewards_before", before_rewards)
        object.__setattr__(self, "greedy_rewards_after", after_rewards)

    @property
    def exact_reward(self) -> int:
        return self.score_before - self.score_after

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


def monochromatic_participation(
    state: RSearchState,
) -> RMonochromaticParticipation:
    """Count red and blue monochromatic K5 participation exactly."""
    index = state.index
    counts = state.color_one_counts

    red_cliques = index.clique_edges[
        counts == 0
    ]
    blue_cliques = index.clique_edges[
        counts == state.edges_per_clique
    ]

    edge_participation = np.zeros(
        (state.number_of_edges, 2),
        dtype=np.int32,
    )

    if len(red_cliques):
        edge_participation[:, 0] = np.bincount(
            red_cliques.reshape(-1),
            minlength=state.number_of_edges,
        )

    if len(blue_cliques):
        edge_participation[:, 1] = np.bincount(
            blue_cliques.reshape(-1),
            minlength=state.number_of_edges,
        )

    vertex_participation = np.zeros(
        (state.graph.problem.n_vertices, 2),
        dtype=np.int32,
    )

    # Every K5 containing a vertex contributes to exactly four of the
    # host edges incident to that vertex.  Summing edge participation
    # at each endpoint and dividing by four therefore recovers the
    # exact number of monochromatic K5s containing the vertex.
    for color in (0, 1):
        loads = edge_participation[:, color]

        np.add.at(
            vertex_participation[:, color],
            state.graph.edges[:, 0],
            loads,
        )
        np.add.at(
            vertex_participation[:, color],
            state.graph.edges[:, 1],
            loads,
        )

    if np.any(vertex_participation % (state.clique_size - 1) != 0):
        raise RuntimeError(
            "Vertex participation incidence did not divide exactly."
        )

    vertex_participation //= state.clique_size - 1

    return RMonochromaticParticipation(
        vertices=vertex_participation,
        edges=edge_participation,
    )


def analyze_edge_flip_causality(
    state: RSearchState,
    edge: int,
) -> REdgeFlipCausalAnalysis:
    """
    Analyze one edge flip without mutating the supplied search state.

    The returned K5 changes form an exact event decomposition: replaying
    their +/-1 color contributions reconstructs all vertex and edge
    monochromatic-participation deltas.
    """
    if isinstance(edge, bool) or not isinstance(edge, Integral):
        raise TypeError("edge must be an integer.")

    edge = int(edge)

    if edge < 0 or edge >= state.number_of_edges:
        raise IndexError("edge is outside the host graph.")

    working_state = state.copy()
    index = working_state.index

    old_color = int(working_state.colors[edge])
    new_color = 1 - old_color
    endpoints = tuple(
        int(vertex)
        for vertex in working_state.graph.edges[edge]
    )

    score_before = working_state.score
    participation_before = monochromatic_participation(
        working_state
    )
    greedy_rewards_before = analyze_actions(
        working_state
    ).immediate_rewards.copy()

    affected_cliques = index.edge_to_cliques[edge]
    counts_before = working_state.color_one_counts[
        affected_cliques
    ].copy()

    working_state.apply_edge_flip(edge)

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

    result = REdgeFlipCausalAnalysis(
        edge=edge,
        endpoints=endpoints,
        old_color=old_color,
        new_color=new_color,
        score_before=score_before,
        score_after=working_state.score,
        participation_before=participation_before,
        participation_after=participation_after,
        clique_changes=tuple(changes),
        greedy_rewards_before=greedy_rewards_before,
        greedy_rewards_after=greedy_rewards_after,
    )

    _verify_event_decomposition(result)

    return result


def _monochromatic_color(
    color_one_count: int,
    edges_per_clique: int,
) -> int | None:
    if color_one_count == 0:
        return 0
    if color_one_count == edges_per_clique:
        return 1
    return None


def _verify_event_decomposition(
    analysis: REdgeFlipCausalAnalysis,
) -> None:
    """Verify that causal events exactly reconstruct both load deltas."""
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