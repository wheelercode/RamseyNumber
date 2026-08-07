"""Structural measurements that distinguish equal-score Ramsey colorings.

Two colorings with the same exact score (monochromatic K5 count) can
differ substantially in how their violations are distributed and how
close they are to further violations. :data:`METRIC_VERSION` and
:func:`calculate_metrics` together define a versioned schema of such
structural measurements, captured in an :class:`RMetricSnapshot`. The
version number is bumped whenever the schema of computed fields changes,
so stored snapshots (see :mod:`ramsey.RMetricStore`) can be recognized as
stale and recomputed rather than silently misinterpreted.
"""

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
    """A reusable structural fingerprint of one complete coloring.

    All array-shaped fields are stored as plain tuples of Python ``int``
    so instances are hashable, JSON-serializable (via :meth:`to_dict` /
    :meth:`from_dict`), and safe to persist (see
    :class:`ramsey.RMetricStore.RMetricStore`).

    Attributes:
        metric_version (int): Schema version this snapshot was computed
            under; should equal :data:`METRIC_VERSION` for a
            freshly-computed snapshot.
        score (int): Exact monochromatic score of the coloring (total
            count of monochromatic K5s).
        red_violations (int): Number of K5s entirely color zero
            (``histogram[0]``).
        blue_violations (int): Number of K5s entirely color one
            (``histogram[-1]``).
        red_deficit_histogram (tuple[int, ...]): Index ``d`` holds the
            number of K5s that would need exactly ``d`` edges recolored
            to red to become monochromatic red (equivalently, the number
            of K5s with exactly ``d`` color-one edges). This is the raw
            color-one-count histogram.
        blue_deficit_histogram (tuple[int, ...]): Index ``d`` holds the
            number of K5s that would need exactly ``d`` edges recolored
            to blue to become monochromatic blue. This is
            ``red_deficit_histogram`` reversed.
        red_vertex_degrees (tuple[int, ...]): Per-vertex count of
            incident red (color zero) edges.
        blue_vertex_degrees (tuple[int, ...]): Per-vertex count of
            incident blue (color one) edges.
        red_vertex_violation_load (tuple[int, ...]): Per-vertex count of
            monochromatic-red K5s the vertex participates in.
        blue_vertex_violation_load (tuple[int, ...]): Per-vertex count of
            monochromatic-blue K5s the vertex participates in.
        red_edge_violation_load (tuple[int, ...]): Per-edge count of
            monochromatic-red K5s the edge participates in.
        blue_edge_violation_load (tuple[int, ...]): Per-edge count of
            monochromatic-blue K5s the edge participates in.
        red_vertex_deficit_one_load (tuple[int, ...]): Per-vertex count
            of participation in K5s with red deficit exactly one (K5s one
            edge recoloring away from becoming monochromatic red).
        blue_vertex_deficit_one_load (tuple[int, ...]): Per-vertex count
            of participation in K5s with blue deficit exactly one.
        red_edge_deficit_one_load (tuple[int, ...]): Per-edge count of
            participation in K5s with red deficit exactly one.
        blue_edge_deficit_one_load (tuple[int, ...]): Per-edge count of
            participation in K5s with blue deficit exactly one.
        red_red_overlap (tuple[int, ...]): Index ``t`` counts pairs of
            distinct monochromatic-red K5s that share exactly ``t``
            vertices. Distinct K5s can share at most four vertices.
        blue_blue_overlap (tuple[int, ...]): Index ``t`` counts pairs of
            distinct monochromatic-blue K5s that share exactly ``t``
            vertices.
        red_blue_overlap (tuple[int, ...]): Index ``t`` counts
            (monochromatic-red, monochromatic-blue) K5 pairs that share
            exactly ``t`` vertices.
        single_flip_rewards (tuple[int, ...]): Exact score reduction that
            would result from flipping each host-graph edge, indexed by
            edge index (as computed by
            :func:`ramsey.RAction.analyze_actions`).
    """

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
        """int: Number of K5s one edge recoloring away from monochromatic red."""
        return self.red_deficit_histogram[1]

    @property
    def blue_deficit_one(self) -> int:
        """int: Number of K5s one edge recoloring away from monochromatic blue."""
        return self.blue_deficit_histogram[1]

    @property
    def best_single_flip_reward(self) -> int:
        """int: Largest exact score reduction achievable by any single edge flip."""
        return max(self.single_flip_rewards)

    @property
    def worst_single_flip_reward(self) -> int:
        """int: Largest exact score increase (worst outcome) any single edge flip could cause."""
        return min(self.single_flip_rewards)

    @property
    def improving_single_flips(self) -> int:
        """int: Number of host edges whose flip would strictly reduce the score."""
        return sum(reward > 0 for reward in self.single_flip_rewards)

    @property
    def neutral_single_flips(self) -> int:
        """int: Number of host edges whose flip would leave the score unchanged."""
        return sum(reward == 0 for reward in self.single_flip_rewards)

    @property
    def worsening_single_flips(self) -> int:
        """int: Number of host edges whose flip would strictly increase the score."""
        return sum(reward < 0 for reward in self.single_flip_rewards)

    @property
    def maximum_vertex_violation_load(self) -> int:
        """int: Largest combined red-plus-blue violation load at any single vertex."""
        return max(
            red + blue
            for red, blue in zip(
                self.red_vertex_violation_load,
                self.blue_vertex_violation_load,
            )
        )

    @property
    def maximum_edge_violation_load(self) -> int:
        """int: Largest combined red-plus-blue violation load at any single edge."""
        return max(
            red + blue
            for red, blue in zip(
                self.red_edge_violation_load,
                self.blue_edge_violation_load,
            )
        )

    @property
    def same_color_shared_edge_violation_pairs(self) -> int:
        """int: Number of same-color monochromatic K5 pairs that share a host edge.

        Two K5s share at least one host edge exactly when they share at
        least two vertices, so this sums ``red_red_overlap`` and
        ``blue_blue_overlap`` from index 2 onward.
        """
        # Two K5s share at least one host edge exactly when they share
        # at least two vertices.
        return sum(self.red_red_overlap[2:]) + sum(
            self.blue_blue_overlap[2:]
        )

    def summary(self) -> dict[str, int | float]:
        """Return compact scalar columns useful for tables and plots.

        Reduces the snapshot's array-shaped fields to a handful of
        headline scalars (score, violation counts, deficit-one counts,
        maximum vertex/edge violation loads, shared-edge violation
        pairs, and single-flip reward statistics), suitable for a
        dataframe row or a summary print.

        Returns:
            dict[str, int | float]: Mapping of column name to value.
        """
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
        """Return a JSON-compatible representation.

        Returns:
            dict[str, Any]: Every field, keyed by name, with tuple fields
            preserved as JSON-serializable lists via ``dataclasses.asdict``.
        """
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        values: dict[str, Any],
    ) -> "RMetricSnapshot":
        """Restore a snapshot from its JSON-compatible representation.

        Args:
            values (dict[str, Any]): A mapping produced by :meth:`to_dict`
                (or an equivalent JSON-decoded object), with array-shaped
                fields as lists rather than tuples.

        Returns:
            RMetricSnapshot: A snapshot equal in value to the one
            originally serialized, with array-shaped fields restored to
            tuples of ``int``.
        """
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
    """Calculate the versioned structural fingerprint of one state.

    Computes every field of :class:`RMetricSnapshot` from the current
    coloring: violation counts and deficit histograms directly from the
    state's histogram, vertex/edge participation loads via
    :func:`ramsey.REdgeFlipCausalAnalysis.monochromatic_participation` and
    the deficit-one-specific helper below, exact pairwise K5 overlap
    counts via inclusion-exclusion, and single-flip rewards via
    :func:`ramsey.RAction.analyze_actions`.

    Args:
        state (RSearchState): Search state to compute metrics for. Must
            use a symmetric two-color problem.

    Returns:
        RMetricSnapshot: The complete structural fingerprint, tagged with
        :data:`METRIC_VERSION`.

    Raises:
        ValueError: If the state's problem is not a symmetric two-color
            problem.
    """
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
    """Count each vertex's incident edges by color.

    Args:
        state (RSearchState): Search state to inspect.

    Returns:
        tuple[NDArray[np.int32], NDArray[np.int32]]: ``(red, blue)``
        arrays of shape ``(n_vertices,)``, where each entry is the
        vertex's degree within the color-zero (red) or color-one (blue)
        edge subgraph, respectively.
    """
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
    """Count vertex and edge participation across a selected subset of cliques.

    Used to compute deficit-one participation loads: ``clique_mask``
    selects the K5s with a given color-one count (e.g. exactly one, for
    red deficit-one), and this returns how many of those selected K5s
    each vertex and edge belongs to.

    Args:
        state (RSearchState): Search state to inspect.
        clique_mask (NDArray[np.bool_]): Boolean mask over all cliques
            selecting the subset to count participation in.

    Returns:
        tuple[NDArray[np.int32], NDArray[np.int32]]: ``(vertex_load,
        edge_load)``, shaped ``(n_vertices,)`` and ``(number_of_edges,)``
        respectively, giving the count of selected cliques each vertex or
        edge participates in. Both are all zero if ``clique_mask``
        selects no cliques.

    Raises:
        RuntimeError: If accumulated vertex incidence does not divide
            evenly by ``clique_size - 1``, which would indicate the
            clique index is inconsistent.
    """
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
    """Count exact red/red, blue/blue, and red/blue K5 intersections.

    Uses inclusion-exclusion over vertex-subset incidence: for each pair
    of same- or different-colored monochromatic K5s, the number of shared
    vertices (0 to 4) is recovered exactly, rather than merely bounded.

    Args:
        state (RSearchState): Search state to inspect.

    Returns:
        tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        ``(red_exact, blue_exact, mixed_exact)``, the exact
        shared-vertex-count histograms for red/red, blue/blue, and
        red/blue monochromatic K5 pairs respectively.
    """
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
    """Count how many listed cliques contain each vertex subset.

    For every subset size from 1 up to ``clique_size - 1``, builds a
    counter mapping each vertex-subset tuple to the number of cliques in
    ``clique_indexes`` whose vertex set contains that subset. These are
    "at least" incidence counts, later inverted into exact intersection
    sizes by :func:`_exact_intersections_from_subset_counts`.

    Args:
        state (RSearchState): Search state to inspect.
        clique_indexes (NDArray[np.int64]): Indices of the monochromatic
            cliques to accumulate subset counts over (e.g. all red or all
            blue violating K5s).

    Returns:
        dict[int, Counter[tuple[int, ...]]]: Mapping from subset size to
        a ``Counter`` of vertex-subset tuple to incidence count.

    Raises:
        RuntimeError: If an indexed clique's edges do not span exactly
            ``state.clique_size`` distinct vertices, indicating the
            clique index is inconsistent.
    """
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
    """Count same-color clique pairs sharing at least each subset size.

    For every subset size, sums ``comb(count, 2)`` over vertex subsets
    that appear in two or more cliques, giving the number of clique pairs
    that share at least that subset of vertices.

    Args:
        subset_counts (dict[int, Counter[tuple[int, ...]]]): Per-subset-size
            incidence counters, as produced by
            :func:`_violation_subset_counts` for one color.

    Returns:
        dict[int, int]: Mapping from subset size to the number of clique
        pairs (of the same color) sharing at least that many vertices.
    """
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
    """Count cross-color clique pairs sharing at least each subset size.

    For every subset size, sums, over vertex subsets appearing in
    ``first``, the product of that subset's count in ``first`` and its
    count in ``second``, giving the number of (first-color, second-color)
    clique pairs that share at least that subset of vertices.

    Args:
        first (dict[int, Counter[tuple[int, ...]]]): Per-subset-size
            incidence counters for one color (e.g. red).
        second (dict[int, Counter[tuple[int, ...]]]): Per-subset-size
            incidence counters for the other color (e.g. blue).

    Returns:
        dict[int, int]: Mapping from subset size to the number of
        cross-color clique pairs sharing at least that many vertices.
    """
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
    """Invert subset-incidence counts into exact intersection sizes.

    Applies inclusion-exclusion from the largest subset size downward:
    the "at least ``shared``" count includes every pair that actually
    shares more than ``shared`` vertices, so those contributions
    (weighted by ``comb(larger, shared)``, the number of ``shared``-sized
    subsets of a ``larger``-sized intersection) are subtracted out to
    leave the count of pairs sharing exactly ``shared`` vertices. The
    zero-overlap count is the remainder after all positive counts are
    accounted for against ``total_pairs``.

    Args:
        at_least (dict[int, int]): Mapping from subset size to the number
            of clique pairs sharing at least that many vertices, as
            produced by :func:`_same_color_subset_pair_counts` or
            :func:`_mixed_subset_pair_counts`.
        total_pairs (int): Total number of clique pairs being
            categorized (e.g. ``comb(len(red_indexes), 2)``).

    Returns:
        tuple[int, ...]: Index ``t`` holds the number of clique pairs
        sharing exactly ``t`` vertices.

    Raises:
        RuntimeError: If the inversion produces a negative count,
            indicating an inconsistency in the input incidence counts.
    """
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
    """Convert an array to a plain tuple of Python ``int``.

    Args:
        values (NDArray): Source array of numeric values.

    Returns:
        tuple[int, ...]: The array's values as a tuple of ``int``,
        suitable for a hashable, JSON-serializable dataclass field.
    """
    return tuple(int(value) for value in values)