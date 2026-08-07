"""Objective-neutral analysis and application of K5 recoloring actions.

A K5 recoloring action (or "K5 pattern macro action") replaces the colors
of all ten edges of one currently monochromatic K5 at once, choosing among
the 1,022 nonmonochromatic ten-edge patterns. This module computes the
exact score consequences of every such pattern for one target K5 and
applies a chosen pattern to a search state. It is objective-neutral: it
reports exact score reductions only and leaves any objective-specific
reward shaping (such as danger-weighted rewards) to :mod:`ramsey.RK5Policy`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .RState import RSearchState


K5_EDGE_COUNT = 10
K5_PATTERN_COUNT = 1 << K5_EDGE_COUNT

# Pattern IDs are ten-bit integers. Bit j is the requested color of
# target edge j in the canonical clique-edge ordering. IDs zero and
# 1023 are monochromatic and are intentionally excluded.
K5_ALLOWED_PATTERN_IDS = np.arange(
    1,
    K5_PATTERN_COUNT - 1,
    dtype=np.uint16,
)

K5_PATTERN_COLORS = (
    (
        K5_ALLOWED_PATTERN_IDS[:, np.newaxis]
        >> np.arange(K5_EDGE_COUNT, dtype=np.uint16)
    )
    & np.uint16(1)
).astype(np.uint8)

K5_POPCOUNT = np.asarray(
    [
        pattern.bit_count()
        for pattern in range(K5_PATTERN_COUNT)
    ],
    dtype=np.uint8,
)

K5_ALLOWED_PATTERN_IDS.flags.writeable = False
K5_PATTERN_COLORS.flags.writeable = False
K5_POPCOUNT.flags.writeable = False


def _read_only_copy(
    values: NDArray,
) -> NDArray:
    """Copy an array-like value and mark the copy read-only.

    Args:
        values (NDArray): Source array or array-like value.

    Returns:
        NDArray: An independently owned copy with ``flags.writeable`` set
        to ``False``, safe to expose on a frozen dataclass without risking
        aliasing the caller's mutable buffers.
    """
    result = np.asarray(values).copy()

    result.flags.writeable = False

    return result


@dataclass(
    frozen=True,
    slots=True,
    eq=False,
)
class RK5PatternAnalysis:
    """Exact consequences of all 1,022 allowed patterns for one bad K5.

    Attributes:
        source_state (RSearchState): Search state this analysis was
            computed against; used together with ``state_version`` by
            :meth:`applies_to` to detect staleness.
        state_version (int): Value of ``source_state.version`` at the time
            of analysis.
        target_clique (int): Index of the analyzed monochromatic K5.
        target_edges (NDArray[np.uint16]): The clique's ten host-graph
            edge indices, in the canonical clique-edge ordering used to
            interpret pattern IDs.
        old_pattern_id (int): The clique's current pattern ID (0 if
            entirely color zero, 1023 if entirely color one).
        pattern_ids (NDArray[np.uint16]): The 1,022 nonmonochromatic
            pattern IDs that were evaluated, i.e. ``K5_ALLOWED_PATTERN_IDS``.
        histogram_deltas (NDArray[np.int32]): Shape ``(1022, 11)``. Row
            ``i`` is the net change to every bin of the global color-one
            edge-count histogram that would result from recoloring the
            target clique to ``pattern_ids[i]``.
        exact_rewards (NDArray[np.int32]): Shape ``(1022,)``. Exact score
            reduction for applying each candidate pattern, derived from
            the change to the two monochromatic histogram bins.
        resulting_scores (NDArray[np.int32]): Shape ``(1022,)``. The
            exact score the state would have after applying each
            candidate pattern.
        changed_edge_counts (NDArray[np.uint8]): Shape ``(1022,)``.
            Number of the ten target edges whose color differs between
            ``old_pattern_id`` and each candidate pattern.
        affected_clique_count (int): Number of distinct K5s (including the
            target) that share at least one edge with the target clique
            and were therefore considered when aggregating histogram
            effects.
    """

    source_state: RSearchState
    state_version: int

    target_clique: int
    target_edges: NDArray[np.uint16]
    old_pattern_id: int

    pattern_ids: NDArray[np.uint16]
    histogram_deltas: NDArray[np.int32]
    exact_rewards: NDArray[np.int32]
    resulting_scores: NDArray[np.int32]
    changed_edge_counts: NDArray[np.uint8]

    affected_clique_count: int

    def __post_init__(self) -> None:
        """Validate array shapes and freeze all fields to owned, read-only copies.

        Raises:
            ValueError: If ``histogram_deltas``, ``exact_rewards``,
                ``resulting_scores``, ``changed_edge_counts``, or
                ``target_edges`` does not have the shape implied by the
                number of evaluated patterns and ``K5_EDGE_COUNT``.
        """
        pattern_ids = _read_only_copy(
            np.asarray(self.pattern_ids, dtype=np.uint16)
        )

        exact_rewards = _read_only_copy(
            np.asarray(self.exact_rewards, dtype=np.int32)
        )

        histogram_deltas = _read_only_copy(
            np.asarray(self.histogram_deltas, dtype=np.int32)
        )

        resulting_scores = _read_only_copy(
            np.asarray(self.resulting_scores, dtype=np.int32)
        )

        changed_edge_counts = _read_only_copy(
            np.asarray(self.changed_edge_counts, dtype=np.uint8)
        )

        action_shape = (len(pattern_ids),)

        expected_delta_shape = (
            len(pattern_ids),
            K5_EDGE_COUNT + 1,
        )

        if histogram_deltas.shape != expected_delta_shape:
            raise ValueError("histogram_deltas has the wrong shape.")

        for name, values in (
            ("exact_rewards", exact_rewards),
            ("resulting_scores", resulting_scores),
            ("changed_edge_counts", changed_edge_counts),
        ):
            if values.shape != action_shape:
                raise ValueError(f"{name} has the wrong shape.")

        target_edges = _read_only_copy(
            np.asarray(self.target_edges, dtype=np.uint16)
        )

        if target_edges.shape != (K5_EDGE_COUNT,):
            raise ValueError("target_edges must contain ten edges.")

        object.__setattr__(self, "state_version", int(self.state_version))
        object.__setattr__(self, "target_clique", int(self.target_clique))
        object.__setattr__(self, "target_edges", target_edges)
        object.__setattr__(self, "old_pattern_id", int(self.old_pattern_id))
        object.__setattr__(self, "pattern_ids", pattern_ids)
        object.__setattr__(self, "histogram_deltas", histogram_deltas)
        object.__setattr__(self, "exact_rewards", exact_rewards)
        object.__setattr__(self, "resulting_scores", resulting_scores)
        object.__setattr__(self, "changed_edge_counts", changed_edge_counts)
        object.__setattr__(
            self,
            "affected_clique_count",
            int(self.affected_clique_count),
        )

    def applies_to(
        self,
        state: RSearchState,
    ) -> bool:
        """Return whether this analysis still describes the given state.

        Both object identity and state version are checked, so a state
        that has been mutated since analysis, or a different state
        entirely, is correctly rejected.

        Args:
            state (RSearchState): Candidate search state to check.

        Returns:
            bool: ``True`` if ``state`` is the same object this analysis
            was computed from and has not since been mutated.
        """
        return self.source_state is state and self.state_version == state.version


def _require_k5_state(
    state: RSearchState,
) -> None:
    """Validate that a search state is configured for K5 forbidden cliques.

    Args:
        state (RSearchState): Search state to check.

    Raises:
        ValueError: If the state's clique size is not 5 or its edge count
            per clique is not ``K5_EDGE_COUNT``.
    """
    if state.clique_size != 5 or state.edges_per_clique != K5_EDGE_COUNT:
        raise ValueError("K5 pattern actions require a K5 search state.")


def pattern_colors(
    pattern_id: int,
) -> NDArray[np.uint8]:
    """Return the ten edge colors encoded by one nonmonochromatic pattern.

    Bit ``j`` of ``pattern_id`` is the requested color (0 or 1) of target
    edge ``j`` in the canonical clique-edge ordering.

    Args:
        pattern_id (int): A nonmonochromatic ten-bit pattern ID between 1
            and 1022 inclusive (0 and 1023 are excluded because they are
            monochromatic).

    Returns:
        NDArray[np.uint8]: Shape ``(10,)`` array of edge colors (0 or 1)
            decoded from ``pattern_id``.

    Raises:
        TypeError: If ``pattern_id`` is not an integer.
        ValueError: If ``pattern_id`` is outside ``[1, 1022]``.
    """
    if isinstance(pattern_id, bool) or not isinstance(
        pattern_id,
        (int, np.integer),
    ):
        raise TypeError("pattern_id must be an integer.")

    pattern_id = int(pattern_id)

    if pattern_id <= 0 or pattern_id >= K5_PATTERN_COUNT - 1:
        raise ValueError("pattern_id must be between 1 and 1022.")

    return (
        (
            np.uint16(pattern_id)
            >> np.arange(K5_EDGE_COUNT, dtype=np.uint16)
        )
        & np.uint16(1)
    ).astype(np.uint8)


def monochromatic_k5_indices(
    state: RSearchState,
) -> NDArray[np.uint32]:
    """Return indexes of every currently monochromatic K5.

    A K5 is monochromatic (a "bad" K5, contributing to the exact score)
    when its color-one edge count is either zero (all color zero) or
    ``state.edges_per_clique`` (all color one).

    Args:
        state (RSearchState): Search state to inspect.

    Returns:
        NDArray[np.uint32]: Indices of every currently monochromatic K5,
        in ascending clique-index order.

    Raises:
        ValueError: If ``state`` is not configured for K5 forbidden
            cliques.
    """
    _require_k5_state(state)

    counts = state.color_one_counts

    return np.flatnonzero(
        (counts == 0)
        | (counts == state.edges_per_clique)
    ).astype(np.uint32)


def analyze_k5_patterns(
    state: RSearchState,
    target_clique: int,
) -> RK5PatternAnalysis:
    """
    Evaluate every nonmonochromatic recoloring of one bad K5 exactly.

    Neighboring K5s are aggregated by two sufficient statistics:

    - which subset of the target's ten edges they share; and
    - how many blue edges they have outside the target.

    A K5 can share only a K2, K3, K4, or K5 with the target, so only a
    small number of shared-edge masks actually occur.  This reduces the
    1,022-pattern calculation from tens of millions of clique-by-pattern
    operations to a small grouped table without losing information.

    Args:
        state (RSearchState): Search state to analyze. Must be configured
            for K5 forbidden cliques.
        target_clique (int): Index of a currently monochromatic K5 to
            evaluate replacement patterns for.

    Returns:
        RK5PatternAnalysis: The exact score, histogram, and changed-edge
        consequences of every one of the 1,022 nonmonochromatic patterns
        for ``target_clique``.

    Raises:
        ValueError: If ``state`` is not configured for K5 forbidden
            cliques, or if ``target_clique`` is not currently
            monochromatic.
        TypeError: If ``target_clique`` is not an integer.
        IndexError: If ``target_clique`` is out of range.
    """
    _require_k5_state(state)

    if isinstance(target_clique, bool) or not isinstance(
        target_clique,
        (int, np.integer),
    ):
        raise TypeError("target_clique must be an integer.")

    target_clique = int(target_clique)

    if target_clique < 0 or target_clique >= state.index.clique_count:
        raise IndexError("target_clique is out of range.")

    target_count = int(state.color_one_counts[target_clique])

    if target_count not in (0, K5_EDGE_COUNT):
        raise ValueError("target_clique must currently be monochromatic.")

    target_edges = state.index.clique_edges[target_clique]

    target_colors = state.colors[target_edges]

    old_pattern_id = int(
        np.sum(
            target_colors.astype(np.uint16)
            << np.arange(K5_EDGE_COUNT, dtype=np.uint16),
            dtype=np.uint16,
        )
    )

    affected_cliques = np.unique(
        state.index.edge_to_cliques[target_edges].reshape(-1)
    )

    edge_bits = np.zeros(
        state.number_of_edges,
        dtype=np.uint16,
    )

    edge_bits[target_edges] = np.left_shift(
        np.uint16(1),
        np.arange(K5_EDGE_COUNT, dtype=np.uint16),
    )

    neighbor_edges = state.index.clique_edges[affected_cliques]

    shared_masks = np.bitwise_or.reduce(
        edge_bits[neighbor_edges],
        axis=1,
    )

    old_shared_blue = K5_POPCOUNT[
        np.bitwise_and(
            shared_masks,
            np.uint16(old_pattern_id),
        )
    ]

    outside_blue = (
        state.color_one_counts[affected_cliques].astype(np.int16)
        - old_shared_blue.astype(np.int16)
    )

    number_of_bins = K5_EDGE_COUNT + 1

    group_indexes = (
        shared_masks.astype(np.int32) * number_of_bins
        + outside_blue.astype(np.int32)
    )

    group_counts = np.bincount(
        group_indexes,
        minlength=K5_PATTERN_COUNT * number_of_bins,
    ).reshape(
        K5_PATTERN_COUNT,
        number_of_bins,
    )

    active_masks, active_outside = np.nonzero(group_counts)

    active_counts = group_counts[
        active_masks,
        active_outside,
    ].astype(np.int64)

    active_masks = active_masks.astype(np.uint16)

    shared_blue = K5_POPCOUNT[
        np.bitwise_and(
            K5_ALLOWED_PATTERN_IDS[:, np.newaxis],
            active_masks[np.newaxis, :],
        )
    ].astype(np.int16)

    new_blue_counts = (
        shared_blue
        + active_outside[np.newaxis, :]
    )

    current_counts = state.color_one_counts[affected_cliques]

    current_histogram = np.bincount(
        current_counts,
        minlength=number_of_bins,
    )

    new_histograms = np.stack(
        [
            (new_blue_counts == count) @ active_counts
            for count in range(number_of_bins)
        ],
        axis=1,
    ).astype(np.int32)

    histogram_deltas = (
        new_histograms
        - current_histogram[np.newaxis, :]
    ).astype(np.int32)

    # Exact score reward is simply the negative change in the two
    # monochromatic histogram bins. This is the same structural rule
    # used by the ordinary single-edge action analyzer.
    exact_rewards = -(
        histogram_deltas[:, 0]
        + histogram_deltas[:, -1]
    )

    resulting_scores = (
        state.score - exact_rewards
    ).astype(np.int32)

    changed_edge_counts = K5_POPCOUNT[
        np.bitwise_xor(
            K5_ALLOWED_PATTERN_IDS,
            np.uint16(old_pattern_id),
        )
    ]

    return RK5PatternAnalysis(
        source_state=state,
        state_version=state.version,
        target_clique=target_clique,
        target_edges=target_edges,
        old_pattern_id=old_pattern_id,
        pattern_ids=K5_ALLOWED_PATTERN_IDS,
        histogram_deltas=histogram_deltas,
        exact_rewards=exact_rewards,
        resulting_scores=resulting_scores,
        changed_edge_counts=changed_edge_counts,
        affected_clique_count=len(affected_cliques),
    )


def apply_k5_pattern(
    state: RSearchState,
    target_clique: int,
    pattern_id: int,
) -> int:
    """Apply one nonmonochromatic ten-edge K5 pattern exactly.

    Recolors the target clique's ten edges to the colors encoded by
    ``pattern_id``, via :meth:`RSearchState.apply_edge_recoloring`, which
    mutates ``state`` in place and keeps all incrementally maintained
    data (histogram, score, action-profile cache, version) consistent.

    Args:
        state (RSearchState): Search state to mutate. Must be configured
            for K5 forbidden cliques.
        target_clique (int): Index of the K5 to recolor.
        pattern_id (int): Nonmonochromatic pattern ID between 1 and 1022,
            as decoded by :func:`pattern_colors`.

    Returns:
        int: The exact score reduction produced by the recoloring; a
        positive value means the score improved.

    Raises:
        ValueError: If ``state`` is not configured for K5 forbidden
            cliques.
        TypeError: If ``target_clique`` is not an integer, or if
            ``pattern_id`` is not an integer (raised by
            :func:`pattern_colors`).
        IndexError: If ``target_clique`` is out of range.
    """
    _require_k5_state(state)

    if isinstance(target_clique, bool) or not isinstance(
        target_clique,
        (int, np.integer),
    ):
        raise TypeError("target_clique must be an integer.")

    target_clique = int(target_clique)

    if target_clique < 0 or target_clique >= state.index.clique_count:
        raise IndexError("target_clique is out of range.")

    colors = pattern_colors(pattern_id)

    target_edges = state.index.clique_edges[target_clique]

    return state.apply_edge_recoloring(
        target_edges,
        colors,
    )