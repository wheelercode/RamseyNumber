"""Pairwise interaction measurements for single-edge Ramsey actions."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np
from numpy.typing import NDArray

from .RAction import immediate_rewards_for_edges
from .RState import RSearchState


def _read_only_copy(array: NDArray) -> NDArray:
    """Return an owned, read-only copy of an array."""
    result = np.asarray(array).copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True, eq=False)
class REdgePairInteraction:
    """Exact mixed effects for an oriented collection of edge pairs.

    For a pair ``(e1, e2)``, ``reward_interactions`` contains

    ``reward(e2 after flipping e1) - reward(e2 before flipping e1)``.

    A positive value means flipping ``e1`` makes ``e2`` a better
    immediate score action. A negative value means the first flip makes
    the second action worse. ``score_interactions`` is the same mixed
    finite difference expressed as a score change and therefore has the
    opposite sign.

    Attributes:
        edge_pairs: Oriented edge-index pairs with shape ``(m, 2)``.
        reward_interactions: Change in the second edge's exact reward.
        shared_vertices: Number of endpoints shared by each edge pair.
        common_cliques: Number of indexed forbidden cliques containing
            both edges.
    """

    edge_pairs: NDArray[np.int32]
    reward_interactions: NDArray[np.int32]
    shared_vertices: NDArray[np.uint8]
    common_cliques: NDArray[np.uint32]

    def __post_init__(self) -> None:
        """Validate shapes and make all stored arrays immutable."""
        edge_pairs = np.asarray(self.edge_pairs, dtype=np.int32)

        if edge_pairs.ndim != 2 or edge_pairs.shape[1] != 2:
            raise ValueError("edge_pairs must have shape (m, 2).")

        number_of_pairs = len(edge_pairs)

        for name, dtype in (
            ("reward_interactions", np.int32),
            ("shared_vertices", np.uint8),
            ("common_cliques", np.uint32),
        ):
            value = np.asarray(getattr(self, name), dtype=dtype)

            if value.shape != (number_of_pairs,):
                raise ValueError(
                    f"{name} must have shape ({number_of_pairs},)."
                )

            object.__setattr__(self, name, _read_only_copy(value))

        object.__setattr__(self, "edge_pairs", _read_only_copy(edge_pairs))

    @property
    def score_interactions(self) -> NDArray[np.int32]:
        """Return the mixed finite difference of Ramsey score."""
        result = -self.reward_interactions.copy()
        result.flags.writeable = False
        return result

    @property
    def adjacent_mask(self) -> NDArray[np.bool_]:
        """Return a mask selecting edge pairs sharing one vertex."""
        result = self.shared_vertices == 1
        result.flags.writeable = False
        return result

    @property
    def disjoint_mask(self) -> NDArray[np.bool_]:
        """Return a mask selecting edge pairs sharing no vertices."""
        result = self.shared_vertices == 0
        result.flags.writeable = False
        return result

    def summary(self) -> dict[str, float | int]:
        """Return compact statistics suitable for experiment reports."""
        values = self.reward_interactions.astype(np.float64)
        adjacent = values[self.adjacent_mask]
        disjoint = values[self.disjoint_mask]

        return {
            "interaction_pairs": int(len(values)),
            "interaction_mean": float(values.mean()) if values.size else 0.0,
            "interaction_std": float(values.std()) if values.size else 0.0,
            "interaction_mean_abs": (
                float(np.abs(values).mean()) if values.size else 0.0
            ),
            "interaction_min": int(values.min()) if values.size else 0,
            "interaction_max": int(values.max()) if values.size else 0,
            "adjacent_interaction_mean": (
                float(adjacent.mean()) if adjacent.size else 0.0
            ),
            "adjacent_interaction_mean_abs": (
                float(np.abs(adjacent).mean()) if adjacent.size else 0.0
            ),
            "disjoint_interaction_mean": (
                float(disjoint.mean()) if disjoint.size else 0.0
            ),
            "disjoint_interaction_mean_abs": (
                float(np.abs(disjoint).mean()) if disjoint.size else 0.0
            ),
        }


def calculate_edge_pair_interactions(
    state: RSearchState,
    edge_pairs: NDArray[np.integer],
) -> REdgePairInteraction:
    """Calculate exact pairwise interaction without mutating ``state``.

    The calculation copies the source state once. For each distinct
    first edge it temporarily flips that edge in the working copy,
    evaluates the requested second edges, and immediately restores the
    first edge. This avoids reconstructing a K5 state for every pair.

    Args:
        state: Complete Ramsey coloring to analyze.
        edge_pairs: Oriented edge-index pairs with shape ``(m, 2)``.

    Returns:
        REdgePairInteraction: Exact reward interactions plus simple
        geometric descriptors of every requested pair.

    Raises:
        ValueError: If the pair table has the wrong shape or contains
            the same edge twice in one pair.
        IndexError: If an edge index is outside the host graph.
    """
    pairs = np.asarray(edge_pairs, dtype=np.int32)

    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("edge_pairs must have shape (m, 2).")

    if pairs.size and (
        np.any(pairs < 0)
        or np.any(pairs >= state.number_of_edges)
    ):
        raise IndexError("edge_pairs contains an invalid edge index.")

    if np.any(pairs[:, 0] == pairs[:, 1]):
        raise ValueError("An interaction pair must contain two edges.")

    if len(pairs) == 0:
        return REdgePairInteraction(
            edge_pairs=pairs,
            reward_interactions=np.empty(0, dtype=np.int32),
            shared_vertices=np.empty(0, dtype=np.uint8),
            common_cliques=np.empty(0, dtype=np.uint32),
        )

    working = state.copy()
    all_edges = np.arange(state.number_of_edges, dtype=np.int32)
    base_rewards = immediate_rewards_for_edges(working, all_edges)
    interactions = np.empty(len(pairs), dtype=np.int32)

    for first_edge in np.unique(pairs[:, 0]):
        pair_rows = np.flatnonzero(pairs[:, 0] == first_edge)
        second_edges = pairs[pair_rows, 1]

        working.apply_edge_flip(int(first_edge))
        changed_rewards = immediate_rewards_for_edges(
            working,
            second_edges,
        )
        working.apply_edge_flip(int(first_edge))

        interactions[pair_rows] = (
            changed_rewards - base_rewards[second_edges]
        )

    edge_table = state.graph.edges
    first_endpoints = edge_table[pairs[:, 0]]
    second_endpoints = edge_table[pairs[:, 1]]

    shared_vertices = (
        (first_endpoints[:, 0, None] == second_endpoints).sum(axis=1)
        + (first_endpoints[:, 1, None] == second_endpoints).sum(axis=1)
    ).astype(np.uint8)

    vertices_spanned = 4 - shared_vertices.astype(np.int32)
    n_vertices = state.graph.problem.n_vertices
    clique_size = state.clique_size
    common_cliques = np.asarray(
        [
            comb(n_vertices - int(span), clique_size - int(span))
            if int(span) <= clique_size
            else 0
            for span in vertices_spanned
        ],
        dtype=np.uint32,
    )

    return REdgePairInteraction(
        edge_pairs=pairs,
        reward_interactions=interactions,
        shared_vertices=shared_vertices,
        common_cliques=common_cliques,
    )
