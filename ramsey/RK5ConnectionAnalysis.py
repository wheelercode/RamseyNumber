"""Connection-class imbalance analysis for individual K5 subgraphs."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np
from numpy.typing import NDArray

from .RState import RSearchState


@dataclass(frozen=True, slots=True)
class RK5ConnectionAnalysis:
    """Store signed K5 imbalance and exact overlap-class summaries.

    Attributes:
        clique_vertices: Vertex indexes for every K5, in graph-index order.
        local_imbalances: Raw signed color imbalance ``blue - red`` for every
            K5. Negative values are red-biased and positive values are
            blue-biased.
        class_imbalance_sums: Sum of neighbor signed imbalances for the exact
            O1 through O4 connection classes. Shape is ``(number_of_k5s, 4)``.
        class_mean_imbalances: Mean signed imbalance in O1 through O4.
        class_sizes: Number of neighbors in O1 through O4.
        class_weights: Weights used to combine the normalized class means.
        weighted_connection_imbalances: Weighted mean of the raw O1 through
            O4 neighborhood color imbalances. This value is independent of the
            target K5's own color imbalance.
    """

    clique_vertices: NDArray[np.uint8]
    local_imbalances: NDArray[np.int8]
    class_imbalance_sums: NDArray[np.int64]
    class_mean_imbalances: NDArray[np.float64]
    class_sizes: NDArray[np.int64]
    class_weights: NDArray[np.float64]
    weighted_connection_imbalances: NDArray[np.float64]

    def connection_histograms(
        self,
        state: RSearchState,
        clique_index: int,
    ) -> NDArray[np.int64]:
        """Return exact H0-H10 distributions for O1 through O4.

        Each neighboring K5 is counted exactly once, according to its exact
        number of vertices shared with the target K5. A neighbor in O3 is not
        also counted in O1 or O2.

        Args:
            state: Coloring state used to calculate K5 color counts.
            clique_index: Target K5 index.

        Returns:
            Array of shape ``(4, 11)``. Row zero is O1 and row three is O4.

        Raises:
            IndexError: If ``clique_index`` is outside the K5 index.
            ValueError: If ``state`` does not describe the analyzed graph.
        """
        if state.clique_size != 5:
            raise ValueError("Connection analysis requires K5 state data.")

        if not 0 <= clique_index < len(self.clique_vertices):
            raise IndexError("clique_index is outside the K5 index.")

        if len(state.color_one_counts) != len(self.clique_vertices):
            raise ValueError("State and connection analysis sizes disagree.")

        target = self.clique_vertices[clique_index]

        shared_counts = np.zeros(
            len(self.clique_vertices),
            dtype=np.uint8,
        )

        for vertex in target:
            shared_counts += np.any(
                self.clique_vertices == vertex,
                axis=1,
            )

        histograms = np.zeros((4, 11), dtype=np.int64)

        for overlap_class in range(1, 5):
            mask = shared_counts == overlap_class
            histograms[overlap_class - 1] = np.bincount(
                state.color_one_counts[mask],
                minlength=11,
            )

        return histograms


def calculate_k5_connection_analysis(
    state: RSearchState,
    *,
    class_weights: tuple[float, float, float, float] = (1.0, 2.0, 3.0, 4.0),
) -> RK5ConnectionAnalysis:
    """Calculate signed O1-O4 imbalance summaries for every K5.

    The calculation uses exact intersection classes. A neighboring K5 that
    shares three vertices with a target contributes to O3 only. Subset
    incidence followed by a descending inversion makes it possible to analyze
    every K5 without constructing the roughly trillion-entry K5-pair matrix.

    Args:
        state: Search state containing the K5 color counts.
        class_weights: Relative weights assigned to normalized O1, O2, O3,
            and O4 mean imbalances when calculating the sortable summary.

    Returns:
        Complete per-K5 connection-class analysis.

    Raises:
        ValueError: If the state is not a K5 problem or the weights are invalid.
    """
    if state.clique_size != 5 or state.edges_per_clique != 10:
        raise ValueError("Connection analysis currently requires K5s.")

    weights = np.asarray(class_weights, dtype=np.float64)

    if weights.shape != (4,):
        raise ValueError("class_weights must contain O1 through O4 weights.")

    if np.any(weights < 0.0) or not np.any(weights > 0.0):
        raise ValueError("class_weights must be nonnegative with a positive sum.")

    n_vertices = state.graph.problem.n_vertices
    number_of_cliques = len(state.color_one_counts)

    clique_vertices = np.asarray(
        list(combinations(range(n_vertices), 5)),
        dtype=np.uint8,
    )

    if len(clique_vertices) != number_of_cliques:
        raise RuntimeError("K5 vertex enumeration and graph index disagree.")

    # Raw color imbalance is blue minus red. For a K5 with h blue edges,
    # blue - red = h - (10 - h) = 2h - 10.
    local_imbalances = (
        2 * state.color_one_counts.astype(np.int16) - 10
    ).astype(np.int8)

    # Y_s(K) is the total signed imbalance of all other K5s containing
    # each s-vertex subset of K, summed over K's C(5,s) such subsets.
    # A neighbor in exact class Ot occurs C(t,s) times in Y_s.
    at_least_class_sums: dict[int, NDArray[np.int64]] = {}
    choose_table = np.zeros((5, n_vertices), dtype=np.int32)

    for order in range(1, 5):
        choose_table[order] = np.asarray(
            [comb(vertex, order) for vertex in range(n_vertices)],
            dtype=np.int32,
        )

    for subset_size in range(1, 5):
        position_sets = tuple(combinations(range(5), subset_size))
        subset_ranks = np.empty(
            (number_of_cliques, len(position_sets)),
            dtype=np.int32,
        )

        for column, positions in enumerate(position_sets):
            subset = clique_vertices[:, positions]
            ranks = np.zeros(number_of_cliques, dtype=np.int32)

            # Combinadic/colex rank: unique and dense in [0, C(n,s)).
            for order, position in enumerate(range(subset_size), start=1):
                values = subset[:, position]
                ranks += choose_table[order, values]

            subset_ranks[:, column] = ranks

        flattened_ranks = subset_ranks.reshape(-1)
        repeated_imbalances = np.repeat(
            local_imbalances.astype(np.int64),
            len(position_sets),
        )

        subset_loads = np.bincount(
            flattened_ranks,
            weights=repeated_imbalances,
            minlength=comb(n_vertices, subset_size),
        ).astype(np.int64)

        totals = subset_loads[subset_ranks].sum(axis=1, dtype=np.int64)

        # Remove the target itself, which appeared once for every subset.
        totals -= comb(5, subset_size) * local_imbalances.astype(np.int64)
        at_least_class_sums[subset_size] = totals

    exact_sums: dict[int, NDArray[np.int64]] = {}

    for overlap_class in range(4, 0, -1):
        values = at_least_class_sums[overlap_class].copy()

        for larger_class in range(overlap_class + 1, 5):
            values -= (
                comb(larger_class, overlap_class)
                * exact_sums[larger_class]
            )

        exact_sums[overlap_class] = values

    class_imbalance_sums = np.column_stack(
        [exact_sums[overlap_class] for overlap_class in range(1, 5)]
    )

    class_sizes = np.asarray(
        [
            comb(5, overlap_class)
            * comb(n_vertices - 5, 5 - overlap_class)
            for overlap_class in range(1, 5)
        ],
        dtype=np.int64,
    )

    class_mean_imbalances = (
        class_imbalance_sums / class_sizes[np.newaxis, :]
    )

    weighted_connection_imbalances = np.average(
        class_mean_imbalances,
        axis=1,
        weights=weights,
    )

    return RK5ConnectionAnalysis(
        clique_vertices=clique_vertices,
        local_imbalances=local_imbalances,
        class_imbalance_sums=class_imbalance_sums,
        class_mean_imbalances=class_mean_imbalances,
        class_sizes=class_sizes,
        class_weights=weights.copy(),
        weighted_connection_imbalances=weighted_connection_imbalances,
    )