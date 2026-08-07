"""Structural measurements that distinguish equal-score Ramsey colorings."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from itertools import combinations
from math import comb
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .RAction import analyze_actions
from .REdgeFlipCausalAnalysis import monochromatic_participation
from .RState import RSearchState


METRIC_VERSION = 1


@dataclass(frozen=True, slots=True)
class RMetricSnapshot:
    """A reusable structural fingerprint of one complete coloring."""

    metric_version: int
    score: int
    red_violations: int
    blue_violations: int

    # Index d contains the number of K5s requiring d edge recolorings
    # to become monochromatic in the named color.
    red_deficit_histogram: tuple[int, ...]
    blue_deficit_histogram: tuple[int, ...]

    red_vertex_degrees: tuple[int, ...]
    blue_vertex_degrees: tuple[int, ...]

    red_vertex_violation_load: tuple[int, ...]
    blue_vertex_violation_load: tuple[int, ...]
    red_edge_violation_load: tuple[int, ...]
    blue_edge_violation_load: tuple[int, ...]

    red_vertex_deficit_one_load: tuple[int, ...]
    blue_vertex_deficit_one_load: tuple[int, ...]
    red_edge_deficit_one_load: tuple[int, ...]
    blue_edge_deficit_one_load: tuple[int, ...]

    # Index t counts pairs of violating K5s sharing exactly t vertices.
    # Distinct K5s can share at most four vertices.
    red_red_overlap: tuple[int, ...]
    blue_blue_overlap: tuple[int, ...]
    red_blue_overlap: tuple[int, ...]

    # Exact score reward for flipping each host-graph edge.
    single_flip_rewards: tuple[int, ...]

    @property
    def red_deficit_one(self) -> int:
        return self.red_deficit_histogram[1]

    @property
    def blue_deficit_one(self) -> int:
        return self.blue_deficit_histogram[1]

    @property
    def best_single_flip_reward(self) -> int:
        return max(self.single_flip_rewards)

    @property
    def worst_single_flip_reward(self) -> int:
        return min(self.single_flip_rewards)

    @property
    def improving_single_flips(self) -> int:
        return sum(reward > 0 for reward in self.single_flip_rewards)

    @property
    def neutral_single_flips(self) -> int:
        return sum(reward == 0 for reward in self.single_flip_rewards)

    @property
    def worsening_single_flips(self) -> int:
        return sum(reward < 0 for reward in self.single_flip_rewards)

    @property
    def maximum_vertex_violation_load(self) -> int:
        return max(
            red + blue
            for red, blue in zip(
                self.red_vertex_violation_load,
                self.blue_vertex_violation_load,
            )
        )

    @property
    def maximum_edge_violation_load(self) -> int:
        return max(
            red + blue
            for red, blue in zip(
                self.red_edge_violation_load,
                self.blue_edge_violation_load,
            )
        )

    @property
    def same_color_shared_edge_violation_pairs(self) -> int:
        # Two K5s share at least one host edge exactly when they share
        # at least two vertices.
        return sum(self.red_red_overlap[2:]) + sum(
            self.blue_blue_overlap[2:]
        )

    def summary(self) -> dict[str, int | float]:
        """Return compact scalar columns useful for tables and plots."""
        rewards = np.asarray(
            self.single_flip_rewards,
            dtype=np.float64,
        )

        return {
            "score": self.score,
            "red_violations": self.red_violations,
            "blue_violations": self.blue_violations,
            "red_deficit_1": self.red_deficit_one,
            "blue_deficit_1": self.blue_deficit_one,
            "max_vertex_violation_load": self.maximum_vertex_violation_load,
            "max_edge_violation_load": self.maximum_edge_violation_load,
            "shared_edge_violation_pairs": (
                self.same_color_shared_edge_violation_pairs
            ),
            "best_single_flip_reward": self.best_single_flip_reward,
            "improving_single_flips": self.improving_single_flips,
            "neutral_single_flips": self.neutral_single_flips,
            "worsening_single_flips": self.worsening_single_flips,
            "mean_single_flip_reward": float(rewards.mean()),
            "std_single_flip_reward": float(rewards.std()),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        values: dict[str, Any],
    ) -> "RMetricSnapshot":
        """Restore a snapshot from its JSON-compatible representation."""
        normalized = dict(values)

        tuple_fields = (
            "red_deficit_histogram",
            "blue_deficit_histogram",
            "red_vertex_degrees",
            "blue_vertex_degrees",
            "red_vertex_violation_load",
            "blue_vertex_violation_load",
            "red_edge_violation_load",
            "blue_edge_violation_load",
            "red_vertex_deficit_one_load",
            "blue_vertex_deficit_one_load",
            "red_edge_deficit_one_load",
            "blue_edge_deficit_one_load",
            "red_red_overlap",
            "blue_blue_overlap",
            "red_blue_overlap",
            "single_flip_rewards",
        )

        for field_name in tuple_fields:
            normalized[field_name] = tuple(
                int(value)
                for value in normalized[field_name]
            )

        return cls(**normalized)


def calculate_metrics(
    state: RSearchState,
) -> RMetricSnapshot:
    """Calculate the versioned structural fingerprint of one state."""
    if state.graph.problem.n_colors != 2 or not state.graph.problem.is_symmetric:
        raise ValueError(
            "calculate_metrics currently requires a symmetric two-color problem."
        )

    edges_per_clique = state.edges_per_clique
    histogram = state.histogram

    # color_one_counts is literally the red completion deficit: every
    # color-one/blue edge must become red.  The blue deficit is the
    # complementary number of red edges and therefore uses the reversed
    # histogram.  Keeping both names makes the two directions explicit.
    red_deficit_histogram = tuple(
        int(value)
        for value in histogram
    )
    blue_deficit_histogram = tuple(
        int(value)
        for value in histogram[::-1]
    )

    participation = monochromatic_participation(state)

    red_vertex_degrees, blue_vertex_degrees = _vertex_color_degrees(
        state
    )

    (
        red_vertex_deficit_one,
        red_edge_deficit_one,
    ) = _participation_for_clique_mask(
        state,
        state.color_one_counts == 1,
    )

    (
        blue_vertex_deficit_one,
        blue_edge_deficit_one,
    ) = _participation_for_clique_mask(
        state,
        state.color_one_counts == edges_per_clique - 1,
    )

    (
        red_red_overlap,
        blue_blue_overlap,
        red_blue_overlap,
    ) = _violation_overlap_histograms(state)

    action_rewards = analyze_actions(
        state
    ).immediate_rewards

    return RMetricSnapshot(
        metric_version=METRIC_VERSION,
        score=state.score,
        red_violations=int(histogram[0]),
        blue_violations=int(histogram[-1]),
        red_deficit_histogram=red_deficit_histogram,
        blue_deficit_histogram=blue_deficit_histogram,
        red_vertex_degrees=_integer_tuple(red_vertex_degrees),
        blue_vertex_degrees=_integer_tuple(blue_vertex_degrees),
        red_vertex_violation_load=_integer_tuple(
            participation.red_vertices
        ),
        blue_vertex_violation_load=_integer_tuple(
            participation.blue_vertices
        ),
        red_edge_violation_load=_integer_tuple(
            participation.red_edges
        ),
        blue_edge_violation_load=_integer_tuple(
            participation.blue_edges
        ),
        red_vertex_deficit_one_load=_integer_tuple(
            red_vertex_deficit_one
        ),
        blue_vertex_deficit_one_load=_integer_tuple(
            blue_vertex_deficit_one
        ),
        red_edge_deficit_one_load=_integer_tuple(
            red_edge_deficit_one
        ),
        blue_edge_deficit_one_load=_integer_tuple(
            blue_edge_deficit_one
        ),
        red_red_overlap=red_red_overlap,
        blue_blue_overlap=blue_blue_overlap,
        red_blue_overlap=red_blue_overlap,
        single_flip_rewards=_integer_tuple(action_rewards),
    )


def _vertex_color_degrees(
    state: RSearchState,
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    number_of_vertices = state.graph.problem.n_vertices
    red = np.zeros(number_of_vertices, dtype=np.int32)
    blue = np.zeros(number_of_vertices, dtype=np.int32)

    endpoints = state.graph.edges
    red_edges = state.colors == 0
    blue_edges = state.colors == 1

    np.add.at(red, endpoints[red_edges, 0], 1)
    np.add.at(red, endpoints[red_edges, 1], 1)
    np.add.at(blue, endpoints[blue_edges, 0], 1)
    np.add.at(blue, endpoints[blue_edges, 1], 1)

    return red, blue


def _participation_for_clique_mask(
    state: RSearchState,
    clique_mask: NDArray[np.bool_],
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    clique_edges = state.index.clique_edges[clique_mask]

    edge_load = np.zeros(
        state.number_of_edges,
        dtype=np.int32,
    )
    vertex_load = np.zeros(
        state.graph.problem.n_vertices,
        dtype=np.int32,
    )

    if len(clique_edges) == 0:
        return vertex_load, edge_load

    edge_load[:] = np.bincount(
        clique_edges.reshape(-1),
        minlength=state.number_of_edges,
    )

    # Each clique contributes through clique_size - 1 incident edges at
    # each of its vertices, just as in monochromatic_participation().
    np.add.at(
        vertex_load,
        state.graph.edges[:, 0],
        edge_load,
    )
    np.add.at(
        vertex_load,
        state.graph.edges[:, 1],
        edge_load,
    )

    divisor = state.clique_size - 1

    if np.any(vertex_load % divisor != 0):
        raise RuntimeError(
            "Clique participation incidence did not divide exactly."
        )

    vertex_load //= divisor

    return vertex_load, edge_load


def _violation_overlap_histograms(
    state: RSearchState,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Count exact red/red, blue/blue, and red/blue K5 intersections."""
    counts = state.color_one_counts
    edges_per_clique = state.edges_per_clique

    red_indexes = np.flatnonzero(counts == 0)
    blue_indexes = np.flatnonzero(counts == edges_per_clique)

    red_subset_counts = _violation_subset_counts(
        state,
        red_indexes,
    )
    blue_subset_counts = _violation_subset_counts(
        state,
        blue_indexes,
    )

    red_at_least = _same_color_subset_pair_counts(
        red_subset_counts
    )
    blue_at_least = _same_color_subset_pair_counts(
        blue_subset_counts
    )
    mixed_at_least = _mixed_subset_pair_counts(
        red_subset_counts,
        blue_subset_counts,
    )

    red_exact = _exact_intersections_from_subset_counts(
        red_at_least,
        total_pairs=comb(len(red_indexes), 2),
    )
    blue_exact = _exact_intersections_from_subset_counts(
        blue_at_least,
        total_pairs=comb(len(blue_indexes), 2),
    )
    mixed_exact = _exact_intersections_from_subset_counts(
        mixed_at_least,
        total_pairs=len(red_indexes) * len(blue_indexes),
    )

    return red_exact, blue_exact, mixed_exact


def _violation_subset_counts(
    state: RSearchState,
    clique_indexes: NDArray[np.int64],
) -> dict[int, Counter[tuple[int, ...]]]:
    result = {
        size: Counter()
        for size in range(1, state.clique_size)
    }

    for clique_index_value in clique_indexes:
        clique_edges = state.index.clique_edges[
            int(clique_index_value)
        ]
        vertices = tuple(
            int(vertex)
            for vertex in np.unique(
                state.graph.edges[clique_edges].reshape(-1)
            )
        )

        if len(vertices) != state.clique_size:
            raise RuntimeError(
                "Indexed clique did not contain the expected vertices."
            )

        for subset_size, counter in result.items():
            counter.update(
                combinations(vertices, subset_size)
            )

    return result


def _same_color_subset_pair_counts(
    subset_counts: dict[int, Counter[tuple[int, ...]]],
) -> dict[int, int]:
    return {
        size: sum(
            comb(count, 2)
            for count in counter.values()
            if count >= 2
        )
        for size, counter in subset_counts.items()
    }


def _mixed_subset_pair_counts(
    first: dict[int, Counter[tuple[int, ...]]],
    second: dict[int, Counter[tuple[int, ...]]],
) -> dict[int, int]:
    result: dict[int, int] = {}

    for size in first:
        first_counter = first[size]
        second_counter = second[size]
        result[size] = sum(
            count * second_counter.get(subset, 0)
            for subset, count in first_counter.items()
        )

    return result


def _exact_intersections_from_subset_counts(
    at_least: dict[int, int],
    *,
    total_pairs: int,
) -> tuple[int, ...]:
    """Invert subset-incidence counts into exact intersection sizes."""
    maximum_shared = max(at_least, default=0)
    exact = [0] * (maximum_shared + 1)

    for shared in range(maximum_shared, 0, -1):
        value = at_least[shared]

        for larger in range(shared + 1, maximum_shared + 1):
            value -= comb(larger, shared) * exact[larger]

        exact[shared] = value

    exact[0] = total_pairs - sum(exact[1:])

    if any(value < 0 for value in exact):
        raise RuntimeError("Violation overlap inversion became negative.")

    return tuple(int(value) for value in exact)


def _integer_tuple(
    values: NDArray,
) -> tuple[int, ...]:
    return tuple(int(value) for value in values)