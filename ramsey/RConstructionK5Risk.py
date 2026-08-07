"""Incremental K5 construction guided by overlapping partial-K5 risk.

Builds a symmetric two-color K5-scored coloring by repeatedly selecting a
constrained, partially colored K5, choosing a full coloring for its
remaining edges (preferring the twelve balanced C5/complement-C5
patterns when compatible with existing colors), and committing the
candidate that minimizes expected future monochromatic K5 risk summed
over every partial K5 touched by the new edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, permutations
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from .RColoring import RColoring
from .RConstruction import RConstruction
from .RGraph import RGraph


def _balanced_k5_cycle_patterns() -> NDArray[np.uint8]:
    """
    Return the 12 labeled red/blue C5-complement-C5 patterns.

    Color one is the five-cycle.  Color zero is its complement, which
    is also a five-cycle.  Consequently neither color contains a
    triangle and every pattern contains exactly five edges of each
    color.

    Each pattern is a length-10 array of 0/1 colors, one entry per edge
    of the 5-vertex clique in the fixed order produced by
    ``itertools.combinations(range(5), 2)``, describing one way to
    2-color K5 as a 5-cycle plus its complementary 5-cycle. Vertex 0 is
    fixed as the cycle's first vertex to remove rotational duplicates;
    reversing a cycle yields the same undirected edge set and is
    removed automatically by the ``set`` deduplication.

    Returns:
        NDArray[np.uint8]: A read-only array of shape ``(12, 10)``
        listing every distinct labeled balanced pattern, sorted in
        ascending lexicographic order.

    Raises:
        RuntimeError: If the computed pattern count or shape is not
            exactly ``(12, 10)`` (an internal consistency check, not
            expected to trigger under normal use).
    """
    local_edges = tuple(
        combinations(
            range(5),
            2,
        )
    )
    edge_lookup = {
        edge: index
        for index, edge in enumerate(local_edges)
    }

    patterns: set[tuple[int, ...]] = set()

    # Fix vertex zero as the first cycle vertex. Rotations disappear;
    # reversing a cycle produces the same undirected edge set and is
    # removed automatically by the set.
    for tail in permutations(range(1, 5)):
        cycle = (0, *tail)
        cycle_edges = {
            tuple(sorted((cycle[i], cycle[(i + 1) % 5])))
            for i in range(5)
        }

        pattern = [0] * len(local_edges)

        for edge in cycle_edges:
            pattern[edge_lookup[edge]] = 1

        patterns.add(tuple(pattern))

    result = np.asarray(
        sorted(patterns),
        dtype=np.uint8,
    )

    if result.shape != (12, 10):
        raise RuntimeError(
            "Expected exactly 12 labeled balanced K5 cycle patterns."
        )

    result.flags.writeable = False
    return result


# Precomputed once at import time: the 12 labeled balanced C5/complement
# -C5 colorings of K5, used to bias new K5 completions toward this
# monochromatic-clique-free structure whenever compatible.
K5_BALANCED_CYCLE_PATTERNS = _balanced_k5_cycle_patterns()


@dataclass(frozen=True, slots=True)
class RK5RiskConstructionReport:
    """
    Diagnostics from one completed K5-risk construction.

    Attributes:
        macro_steps (int): Number of target-K5 completion steps taken
            (each step colors every remaining edge of one selected K5).
        balanced_pattern_steps (int): Subset of ``macro_steps`` where the
            selected K5 was compatible with one of the 12 balanced
            C5/complement-C5 patterns.
        fallback_pattern_steps (int): Subset of ``macro_steps`` where no
            balanced pattern was compatible, so all nonmonochromatic
            completions were considered instead.
        risk_ties (int): Number of steps where more than one candidate
            pattern tied for minimum expected risk, requiring the
            balance/random tie-break.
        red_edges (int): Total edges assigned color 0 (red).
        blue_edges (int): Total edges assigned color 1 (blue).
    """

    macro_steps: int
    balanced_pattern_steps: int
    fallback_pattern_steps: int
    risk_ties: int
    red_edges: int
    blue_edges: int

    @property
    def total_edges(self) -> int:
        """
        int: Total number of colored edges (``red_edges + blue_edges``).
        """
        return self.red_edges + self.blue_edges


@dataclass(slots=True)
class RK5RiskConstruction(RConstruction):
    """
    Build a coloring by completing carefully selected K5s.

    The constructor prefers incomplete K5s having at least
    ``preferred_new_edges`` uncolored edges, while choosing the most
    constrained member of that group.  With the default value four,
    the construction naturally grows from one completed K5 into K5s
    sharing four vertices with already colored structure.

    Whenever the existing assignments are compatible with one of the
    12 balanced C5/complement-C5 K5 colorings, only those patterns are
    considered.  Otherwise all compatible nonmonochromatic assignments
    of the target K5's remaining edges are considered.

    Each candidate is scored jointly over every partial K5 containing
    at least one newly colored edge.  Lower expected future
    monochromatic risk wins. Ties are broken by preferring the candidate
    that keeps the overall red/blue edge count closest to balanced,
    with any remaining tie broken uniformly at random.

    Requires a symmetric two-color Ramsey problem scored by K5 cliques.

    Attributes:
        rng (numpy.random.Generator): Source of randomness used to
            select among tied target cliques and tied candidate
            patterns.
        minimum_pressure_edges (int): Minimum same-color edge count a
            partial K5 must reach, with no opposite-colored edges yet
            assigned, before it contributes to risk.
        preferred_new_edges (int): Minimum number of uncolored edges a
            target K5 should still have to be preferred as the next
            completion target.
        risk_epsilon (float): Maximum absolute risk difference from the
            minimum for a candidate pattern to be treated as tied for
            best.
    """

    rng: np.random.Generator
    minimum_pressure_edges: int = 4
    preferred_new_edges: int = 4
    risk_epsilon: float = 1.0e-12

    _last_report: RK5RiskConstructionReport | None = field(
        init=False,
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        """
        Validate ``rng``, ``minimum_pressure_edges``,
        ``preferred_new_edges``, and ``risk_epsilon``.

        Raises:
            TypeError: If ``rng`` is not a NumPy ``Generator``, or if
                ``minimum_pressure_edges`` or ``preferred_new_edges``
                is not an integer.
            ValueError: If ``minimum_pressure_edges`` or
                ``preferred_new_edges`` is less than 1, or if
                ``risk_epsilon`` is negative.
        """
        if not isinstance(self.rng, np.random.Generator):
            raise TypeError("rng must be a NumPy Generator.")

        for name in (
            "minimum_pressure_edges",
            "preferred_new_edges",
        ):
            value = getattr(self, name)

            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer.")

            value = int(value)

            if value < 1:
                raise ValueError(f"{name} must be positive.")

            object.__setattr__(self, name, value)

        if self.risk_epsilon < 0:
            raise ValueError("risk_epsilon cannot be negative.")

        self.risk_epsilon = float(self.risk_epsilon)

    @property
    def name(self) -> str:
        """
        str: Name encoding ``minimum_pressure_edges`` and
        ``preferred_new_edges``, e.g. ``"k5-risk-pressure-4-new-4"``.
        """
        return (
            "k5-risk-"
            f"pressure-{self.minimum_pressure_edges}-"
            f"new-{self.preferred_new_edges}"
        )

    @property
    def last_report(
        self,
    ) -> RK5RiskConstructionReport | None:
        """
        RK5RiskConstructionReport | None: Diagnostics from the most
        recent call to :meth:`construct`, or ``None`` beforehand.
        """
        return self._last_report

    def construct(
        self,
        graph: RGraph,
    ) -> RColoring:
        """
        Color ``graph`` by repeatedly completing constrained K5s.

        Repeatedly selects a target K5 (see :meth:`_select_target_clique`),
        enumerates candidate colorings for its remaining edges (see
        :meth:`_candidate_patterns` and :meth:`_fallback_patterns`),
        scores each candidate by summed expected risk over every
        affected partial K5, and commits the lowest-risk candidate
        (breaking ties toward color balance and then at random) until
        every edge is colored.

        Args:
            graph (RGraph): Host graph to color. Its problem must use
                two symmetric colors and be scored by K5 cliques.

        Returns:
            RColoring: The completed coloring. Also records diagnostics
            retrievable via :attr:`last_report`.

        Raises:
            ValueError: If ``graph.problem`` is not a symmetric
                two-color problem, if it is not scored by K5 cliques, or
                if ``minimum_pressure_edges`` or ``preferred_new_edges``
                exceeds the number of edges in a K5 (ten).
            RuntimeError: If a selected target K5 has no uncolored
                edges (an internal consistency check on the selection
                logic, not expected to trigger under normal use).
        """
        problem = graph.problem

        if problem.n_colors != 2 or not problem.is_symmetric:
            raise ValueError(
                "RK5RiskConstruction requires a symmetric two-color "
                "Ramsey problem."
            )

        if problem.required_clique_sizes != (5,):
            raise ValueError(
                "RK5RiskConstruction currently requires K5 scoring."
            )

        index = graph.subgraph_index(5)
        edges_per_clique = index.edges_per_clique

        if self.minimum_pressure_edges > edges_per_clique:
            raise ValueError(
                "minimum_pressure_edges cannot exceed ten."
            )

        if self.preferred_new_edges > edges_per_clique:
            raise ValueError(
                "preferred_new_edges cannot exceed ten."
            )

        risk_weights = self._risk_weights(edges_per_clique)

        # -1 is uncolored; zero and one are the two final colors.
        colors = np.full(
            graph.number_of_edges,
            -1,
            dtype=np.int8,
        )

        red_counts = np.zeros(
            index.clique_count,
            dtype=np.uint8,
        )
        blue_counts = np.zeros_like(red_counts)
        assigned_counts = np.zeros_like(red_counts)

        macro_steps = 0
        balanced_pattern_steps = 0
        fallback_pattern_steps = 0
        risk_ties = 0

        while np.any(colors < 0):
            target = self._select_target_clique(
                assigned_counts=assigned_counts,
                edges_per_clique=edges_per_clique,
            )

            target_edges = index.clique_edges[target]
            target_colors = colors[target_edges]
            uncolored_positions = np.flatnonzero(
                target_colors < 0
            ).astype(np.int32)

            if uncolored_positions.size == 0:
                raise RuntimeError(
                    "Selected target K5 contains no uncolored edges."
                )

            candidate_patterns = self._candidate_patterns(
                target_colors
            )

            used_balanced_patterns = (
                len(candidate_patterns) > 0
            )

            if not used_balanced_patterns:
                candidate_patterns = self._fallback_patterns(
                    target_colors
                )

            new_edges = target_edges[uncolored_positions]
            affected_cliques = np.unique(
                index.edge_to_cliques[
                    new_edges
                ].reshape(-1)
            )

            local_positions = tuple(
                np.searchsorted(
                    affected_cliques,
                    index.edge_to_cliques[int(edge)],
                )
                for edge in new_edges
            )

            risks = np.empty(
                len(candidate_patterns),
                dtype=np.float64,
            )

            base_red = red_counts[affected_cliques]
            base_blue = blue_counts[affected_cliques]

            for candidate_index, pattern in enumerate(
                candidate_patterns
            ):
                candidate_red = base_red.copy()
                candidate_blue = base_blue.copy()

                for new_index, local_position in enumerate(
                    uncolored_positions
                ):
                    color = int(pattern[local_position])
                    affected_positions = local_positions[new_index]

                    if color == 0:
                        candidate_red[affected_positions] += 1
                    else:
                        candidate_blue[affected_positions] += 1

                risks[candidate_index] = self._partial_risk(
                    candidate_red,
                    candidate_blue,
                    risk_weights,
                )

            minimum_risk = float(risks.min())
            best_candidates = np.flatnonzero(
                np.abs(risks - minimum_risk)
                <= self.risk_epsilon
            )

            if len(best_candidates) > 1:
                risk_ties += 1

                # Among equal-risk patterns, prefer the assignment
                # keeping total graph color density closest to 50/50.
                current_red = int(np.count_nonzero(colors == 0))
                current_blue = int(np.count_nonzero(colors == 1))

                balance_costs = np.empty(
                    len(best_candidates),
                    dtype=np.int32,
                )

                for index_in_best, candidate_index in enumerate(
                    best_candidates
                ):
                    pattern = candidate_patterns[candidate_index]
                    new_colors = pattern[uncolored_positions]
                    added_red = int(np.count_nonzero(new_colors == 0))
                    added_blue = len(new_colors) - added_red
                    balance_costs[index_in_best] = abs(
                        (current_red + added_red)
                        - (current_blue + added_blue)
                    )

                minimum_balance_cost = int(balance_costs.min())
                balanced_best = best_candidates[
                    balance_costs == minimum_balance_cost
                ]
                selected_candidate = int(
                    self.rng.choice(balanced_best)
                )
            else:
                selected_candidate = int(best_candidates[0])

            selected_pattern = candidate_patterns[
                selected_candidate
            ]

            for local_position in uncolored_positions:
                edge = int(target_edges[local_position])
                color = int(selected_pattern[local_position])
                edge_cliques = index.edge_to_cliques[edge]

                colors[edge] = color
                assigned_counts[edge_cliques] += 1

                if color == 0:
                    red_counts[edge_cliques] += 1
                else:
                    blue_counts[edge_cliques] += 1

            macro_steps += 1

            if used_balanced_patterns:
                balanced_pattern_steps += 1
            else:
                fallback_pattern_steps += 1

        final_colors = colors.astype(np.uint8)
        red_edges = int(np.count_nonzero(final_colors == 0))
        blue_edges = graph.number_of_edges - red_edges

        self._last_report = RK5RiskConstructionReport(
            macro_steps=macro_steps,
            balanced_pattern_steps=balanced_pattern_steps,
            fallback_pattern_steps=fallback_pattern_steps,
            risk_ties=risk_ties,
            red_edges=red_edges,
            blue_edges=blue_edges,
        )

        return RColoring(
            graph,
            final_colors,
        )

    def _select_target_clique(
        self,
        *,
        assigned_counts: NDArray[np.uint8],
        edges_per_clique: int,
    ) -> int:
        """
        Choose the next K5 to complete.

        Prefers cliques with at least ``preferred_new_edges`` uncolored
        edges, breaking ties toward the most already-assigned (most
        constrained) such clique; when no clique meets that preference,
        falls back to the most-assigned incomplete clique instead. Any
        remaining tie is broken uniformly at random.

        Args:
            assigned_counts (NDArray[np.uint8]): Number of colored edges
                in every indexed clique, indexed by clique id.
            edges_per_clique (int): Number of edges in each K5 (ten).

        Returns:
            int: Index of the selected target clique.

        Raises:
            RuntimeError: If every clique is already fully colored.
        """
        uncolored_counts = (
            edges_per_clique
            - assigned_counts.astype(np.int16)
        )

        preferred = uncolored_counts >= self.preferred_new_edges

        if np.any(preferred):
            maximum_assigned = int(
                assigned_counts[preferred].max()
            )
            candidates = np.flatnonzero(
                preferred
                & (assigned_counts == maximum_assigned)
            )
        else:
            incomplete = uncolored_counts > 0

            if not np.any(incomplete):
                raise RuntimeError("No incomplete K5 remains.")

            maximum_assigned = int(
                assigned_counts[incomplete].max()
            )
            candidates = np.flatnonzero(
                incomplete
                & (assigned_counts == maximum_assigned)
            )

        return int(self.rng.choice(candidates))

    @staticmethod
    def _candidate_patterns(
        target_colors: NDArray[np.int8],
    ) -> NDArray[np.uint8]:
        """
        Return balanced C5/complement-C5 patterns compatible with
        ``target_colors``.

        Args:
            target_colors (NDArray[np.int8]): Length-10 array of colors
                already assigned to the target K5's edges, in the fixed
                combination order; ``-1`` marks an uncolored edge.

        Returns:
            NDArray[np.uint8]: The subset of
            :data:`K5_BALANCED_CYCLE_PATTERNS` that agree with every
            already-colored edge in ``target_colors``. Empty if no
            balanced pattern is compatible; equal to
            :data:`K5_BALANCED_CYCLE_PATTERNS` unchanged if no edge is
            yet colored.
        """
        colored = target_colors >= 0

        if not np.any(colored):
            return K5_BALANCED_CYCLE_PATTERNS

        compatible = np.all(
            K5_BALANCED_CYCLE_PATTERNS[:, colored]
            == target_colors[colored],
            axis=1,
        )

        return K5_BALANCED_CYCLE_PATTERNS[compatible]

    @staticmethod
    def _fallback_patterns(
        target_colors: NDArray[np.int8],
    ) -> NDArray[np.uint8]:
        """
        Enumerate every nonmonochromatic completion of the target K5.

        Used when no balanced C5/complement-C5 pattern is compatible
        with the target's existing colors. Enumerates every 0/1
        assignment of the remaining uncolored edges and keeps those that
        do not leave the K5 entirely one color, so this construction
        never deliberately creates a new monochromatic K5 when a
        nonmonochromatic completion exists.

        Args:
            target_colors (NDArray[np.int8]): Length-10 array of colors
                already assigned to the target K5's edges, in the fixed
                combination order; ``-1`` marks an uncolored edge.

        Returns:
            NDArray[np.uint8]: Every completion of ``target_colors``
            that avoids a monochromatic result, or, if every
            completion would be monochromatic (the target is already
            forced), every possible completion instead so the
            construction can make progress.
        """
        uncolored_positions = np.flatnonzero(
            target_colors < 0
        )
        number_of_unknown = len(uncolored_positions)

        patterns: list[np.ndarray] = []

        for pattern_id in range(1 << number_of_unknown):
            pattern = target_colors.copy()

            for bit, local_position in enumerate(
                uncolored_positions
            ):
                pattern[local_position] = (
                    pattern_id >> bit
                ) & 1

            # Never deliberately complete the target as monochromatic
            # when a nonmonochromatic completion is available.
            if np.all(pattern == 0) or np.all(pattern == 1):
                continue

            patterns.append(
                pattern.astype(np.uint8)
            )

        if not patterns:
            # This can only occur if previous assignments have already
            # forced the target K5 monochromatic.  Preserve progress and
            # enumerate the remaining assignments rather than deadlock.
            for pattern_id in range(1 << number_of_unknown):
                pattern = target_colors.copy()

                for bit, local_position in enumerate(
                    uncolored_positions
                ):
                    pattern[local_position] = (
                        pattern_id >> bit
                    ) & 1

                patterns.append(
                    pattern.astype(np.uint8)
                )

        return np.asarray(
            patterns,
            dtype=np.uint8,
        )

    def _risk_weights(
        self,
        edges_per_clique: int,
    ) -> NDArray[np.float64]:
        """
        Build the risk weight table indexed by same-color edge count.

        Weight ``k`` (for ``k >= minimum_pressure_edges``) is
        ``2 ** -(edges_per_clique - k)``, the probability that a K5
        already showing ``k`` edges of one color and none of the
        opposite color would become monochromatic if its remaining
        edges were colored uniformly at random. Weights below
        ``minimum_pressure_edges`` are zero, so cliques under that
        pressure threshold contribute no risk.

        Args:
            edges_per_clique (int): Number of edges in each K5 (ten).

        Returns:
            NDArray[np.float64]: Array of length ``edges_per_clique +
            1`` mapping a same-color edge count to its risk weight.
        """
        weights = np.zeros(
            edges_per_clique + 1,
            dtype=np.float64,
        )

        for same_color in range(
            self.minimum_pressure_edges,
            edges_per_clique + 1,
        ):
            uncolored = edges_per_clique - same_color
            weights[same_color] = 2.0 ** (-uncolored)

        return weights

    @staticmethod
    def _partial_risk(
        red_counts: NDArray[np.uint8],
        blue_counts: NDArray[np.uint8],
        risk_weights: NDArray[np.float64],
    ) -> float:
        """
        Sum expected monochromatic-completion risk over a set of cliques.

        For each clique that is entirely red so far (``blue_counts ==
        0``), looks up the risk weight for its red edge count; likewise
        for cliques that are entirely blue so far. Cliques already
        containing both colors contribute zero, since they can no
        longer become monochromatic.

        Args:
            red_counts (NDArray[np.uint8]): Red edge count in each
                considered clique.
            blue_counts (NDArray[np.uint8]): Blue edge count in each
                considered clique, aligned by index with ``red_counts``.
            risk_weights (NDArray[np.float64]): Risk weight table from
                :meth:`_risk_weights`, indexed by same-color edge count.

        Returns:
            float: Total expected future monochromatic-clique risk
            summed over every considered clique.
        """
        red_only = blue_counts == 0
        blue_only = red_counts == 0

        return float(
            risk_weights[red_counts[red_only]].sum()
            + risk_weights[blue_counts[blue_only]].sum()
        )
