"""Incremental edge construction guided by partial-K5 future risk.

Builds one coloring by assigning edges to a symmetric two-color K5 problem
in random order, one edge at a time. Each edge normally receives the
globally balanced alternating color, but is overridden whenever the
partial K5s touching that edge show a clear expected-risk advantage for
the opposite color, in an attempt to steer away from future monochromatic
K5 completions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Callable

import numpy as np

from .RColoring import RColoring
from .RConstruction import RConstruction
from .RGraph import RGraph


@dataclass(frozen=True, slots=True)
class REdgeRiskConstructionReport:
    """
    Diagnostics from one completed edge-risk construction.

    Attributes:
        alternating_decisions (int): Number of edges colored by the
            globally balanced alternating rule (risk was tied within
            ``risk_epsilon``).
        risk_decisions (int): Number of edges colored by comparing
            expected future monochromatic risk between the two colors.
        risk_overrides (int): Subset of ``risk_decisions`` where the
            risk-driven color differed from the alternating proposal.
        first_risk_step (int | None): Zero-based edge-processing step at
            which risk first broke a tie, or ``None`` if risk never
            differed from the alternating proposal.
        red_edges (int): Total edges assigned color 0 (red).
        blue_edges (int): Total edges assigned color 1 (blue).
    """

    alternating_decisions: int
    risk_decisions: int
    risk_overrides: int
    first_risk_step: int | None
    red_edges: int
    blue_edges: int

    @property
    def total_edges(self) -> int:
        """
        int: Total number of colored edges (``red_edges + blue_edges``).
        """
        return self.red_edges + self.blue_edges


@dataclass(frozen=True, slots=True)
class REdgeRiskConstructionProgress:
    """
    One observable edge-assignment event during construction.

    Passed to the construction's ``observer`` callback (if any) after
    each edge is colored, giving a full account of that step's decision.

    Attributes:
        step_number (int): One-based index of this edge within the
            random processing order.
        total_edges (int): Total number of edges being colored.
        edge (int): Index of the edge that was just colored.
        endpoints (tuple[int, int]): Vertex pair joined by ``edge``.
        color (int): Color actually assigned to ``edge`` (0 or 1).
        alternating_color (int): Color the globally balanced alternating
            rule would have proposed for this step.
        risk_red (float): Expected future monochromatic risk summed over
            partial K5s containing ``edge`` if it were colored red.
        risk_blue (float): Expected future monochromatic risk summed
            over partial K5s containing ``edge`` if it were colored
            blue.
        red_completions (int): Number of partial K5s containing
            ``edge`` that would become a monochromatic red K5 if
            ``edge`` were colored red.
        blue_completions (int): Number of partial K5s containing
            ``edge`` that would become a monochromatic blue K5 if
            ``edge`` were colored blue.
        risk_driven (bool): Whether ``risk_red`` and ``risk_blue``
            differed by more than ``risk_epsilon``, making this a
            risk-driven decision rather than an alternating one.
        overrode_alternating (bool): Whether the color actually chosen
            differs from ``alternating_color``.
        red_edges (int): Cumulative red edges assigned through this
            step, inclusive.
        blue_edges (int): Cumulative blue edges assigned through this
            step, inclusive.
        alternating_decisions (int): Cumulative alternating-rule
            decisions through this step, inclusive.
        risk_decisions (int): Cumulative risk-driven decisions through
            this step, inclusive.
        risk_overrides (int): Cumulative risk-driven decisions that
            overrode the alternating proposal, through this step,
            inclusive.
    """

    step_number: int
    total_edges: int
    edge: int
    endpoints: tuple[int, int]
    color: int
    alternating_color: int
    risk_red: float
    risk_blue: float
    red_completions: int
    blue_completions: int
    risk_driven: bool
    overrode_alternating: bool
    red_edges: int
    blue_edges: int
    alternating_decisions: int
    risk_decisions: int
    risk_overrides: int

    @property
    def completed(self) -> bool:
        """
        bool: Whether this step colored the final edge of the graph.
        """
        return self.step_number == self.total_edges

    @property
    def unavoidable(self) -> bool:
        """
        bool: Whether coloring ``edge`` either color would complete at
        least one monochromatic K5 (a forced trade-off between colors).
        """
        return self.red_completions > 0 and self.blue_completions > 0

    @property
    def created_monochromatic_k5s(self) -> int:
        """
        int: Number of monochromatic K5s completed by the color
        actually assigned to ``edge`` (``red_completions`` or
        ``blue_completions``, matching ``color``).
        """
        if self.color == 0:
            return self.red_completions

        return self.blue_completions


@dataclass(slots=True)
class REdgeRiskConstruction(RConstruction):
    """
    Color the host graph one edge at a time while minimizing K5 risk.

    Requires a symmetric two-color Ramsey problem scored by a single
    forbidden clique size (e.g. K5). Edges are processed in one random
    order. Step parity proposes the globally balanced red/blue
    alternating color. If the partial K5s containing the current edge
    give one candidate color strictly less expected future monochromatic
    risk (beyond ``risk_epsilon``), that color overrides the alternating
    proposal.

    A partial K5 contributes risk only after at least
    ``minimum_pressure_edges`` edges of one color have been assigned
    while none of the opposite color have been assigned; its risk weight
    is ``2 ** -(uncolored edges remaining in that K5)``, the probability
    that the remaining edges would all randomly complete the
    monochromatic clique.

    Attributes:
        rng (numpy.random.Generator): Source of randomness used to
            choose the edge processing order.
        minimum_pressure_edges (int): Minimum same-color edge count a
            partial K5 must reach, with no opposite-colored edges yet
            assigned, before it contributes to risk.
        risk_epsilon (float): Minimum absolute difference between
            ``risk_red`` and ``risk_blue`` required to treat a decision
            as risk-driven rather than alternating.
        observer (Callable[[REdgeRiskConstructionProgress], None] | None):
            Optional callback invoked with a
            :class:`REdgeRiskConstructionProgress` after each edge is
            colored, for diagnostics or visualization.
    """

    rng: np.random.Generator
    minimum_pressure_edges: int = 4
    risk_epsilon: float = 1.0e-12
    observer: Callable[[REdgeRiskConstructionProgress], None] | None = None

    _last_report: REdgeRiskConstructionReport | None = field(
        init=False,
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        """
        Validate ``rng``, ``minimum_pressure_edges``, ``risk_epsilon``,
        and ``observer``.

        Raises:
            TypeError: If ``rng`` is not a NumPy ``Generator``,
                ``minimum_pressure_edges`` is not an integer, or
                ``observer`` is neither ``None`` nor callable.
            ValueError: If ``minimum_pressure_edges`` is less than 1 or
                ``risk_epsilon`` is negative.
        """
        if not isinstance(self.rng, np.random.Generator):
            raise TypeError("rng must be a NumPy Generator.")

        if (
            isinstance(self.minimum_pressure_edges, bool)
            or not isinstance(self.minimum_pressure_edges, Integral)
        ):
            raise TypeError("minimum_pressure_edges must be an integer.")

        self.minimum_pressure_edges = int(
            self.minimum_pressure_edges
        )

        if self.minimum_pressure_edges < 1:
            raise ValueError(
                "minimum_pressure_edges must be positive."
            )

        if self.risk_epsilon < 0:
            raise ValueError("risk_epsilon cannot be negative.")

        self.risk_epsilon = float(self.risk_epsilon)

        if self.observer is not None and not callable(self.observer):
            raise TypeError("observer must be callable or None.")

    @property
    def name(self) -> str:
        """
        str: Name encoding ``minimum_pressure_edges``, e.g.
        ``"edge-risk-pressure-4"``.
        """
        return (
            "edge-risk-"
            f"pressure-{self.minimum_pressure_edges}"
        )

    @property
    def last_report(
        self,
    ) -> REdgeRiskConstructionReport | None:
        """
        REdgeRiskConstructionReport | None: Diagnostics from the most
        recent call to :meth:`construct`, or ``None`` beforehand.
        """
        return self._last_report

    def construct(
        self,
        graph: RGraph,
    ) -> RColoring:
        """
        Color every edge of ``graph`` one at a time, minimizing K5 risk.

        Args:
            graph (RGraph): Host graph to color. Its problem must use
                two symmetric colors and a single forbidden clique
                size.

        Returns:
            RColoring: The completed coloring. Also records diagnostics
            retrievable via :attr:`last_report`, and, if ``observer`` is
            set, reports one :class:`REdgeRiskConstructionProgress`
            event per colored edge.

        Raises:
            ValueError: If ``graph.problem`` is not a symmetric
                two-color problem, or if ``minimum_pressure_edges``
                exceeds the number of edges in the forbidden clique.
        """
        problem = graph.problem

        if problem.n_colors != 2 or not problem.is_symmetric:
            raise ValueError(
                "REdgeRiskConstruction requires a symmetric "
                "two-color Ramsey problem."
            )

        clique_size = problem.required_clique_sizes[0]
        index = graph.subgraph_index(clique_size)
        edges_per_clique = index.edges_per_clique

        if self.minimum_pressure_edges > edges_per_clique:
            raise ValueError(
                "minimum_pressure_edges cannot exceed the number "
                "of edges in the forbidden clique."
            )

        # weights[k] is the probability that a K5 containing k edges
        # of one color and no opposite-colored edges becomes
        # monochromatic if all remaining edges are random.
        risk_weights = np.zeros(
            edges_per_clique + 1,
            dtype=np.float64,
        )

        for assigned_same_color in range(
            self.minimum_pressure_edges,
            edges_per_clique + 1,
        ):
            uncolored = (
                edges_per_clique
                - assigned_same_color
            )
            risk_weights[assigned_same_color] = 2.0 ** (-uncolored)

        red_counts = np.zeros(
            index.clique_count,
            dtype=np.uint8,
        )
        blue_counts = np.zeros_like(red_counts)

        colors = np.empty(
            graph.number_of_edges,
            dtype=np.uint8,
        )

        edge_order = self.rng.permutation(
            graph.number_of_edges
        )

        alternating_decisions = 0
        risk_decisions = 0
        risk_overrides = 0
        first_risk_step: int | None = None
        red_edges = 0
        blue_edges = 0

        for step, edge_value in enumerate(edge_order):
            edge = int(edge_value)
            affected = index.edge_to_cliques[edge]

            affected_red = red_counts[affected]
            affected_blue = blue_counts[affected]

            red_completions = int(
                np.count_nonzero(
                    (affected_red == edges_per_clique - 1)
                    & (affected_blue == 0)
                )
            )
            blue_completions = int(
                np.count_nonzero(
                    (affected_blue == edges_per_clique - 1)
                    & (affected_red == 0)
                )
            )

            red_after = (
                affected_red.astype(np.int16) + 1
            )
            blue_after = (
                affected_blue.astype(np.int16) + 1
            )

            risk_red = float(
                risk_weights[red_after][
                    affected_blue == 0
                ].sum()
            )
            risk_blue = float(
                risk_weights[blue_after][
                    affected_red == 0
                ].sum()
            )

            alternating_color = step % 2
            risk_difference = risk_red - risk_blue

            if abs(risk_difference) <= self.risk_epsilon:
                color = alternating_color
                alternating_decisions += 1
            else:
                color = 0 if risk_red < risk_blue else 1
                risk_decisions += 1

                if first_risk_step is None:
                    first_risk_step = step

                if color != alternating_color:
                    risk_overrides += 1

            colors[edge] = color

            if color == 0:
                red_counts[affected] += 1
                red_edges += 1
            else:
                blue_counts[affected] += 1
                blue_edges += 1

            if self.observer is not None:
                endpoints = graph.edges[edge]

                self.observer(
                    REdgeRiskConstructionProgress(
                        step_number=step + 1,
                        total_edges=graph.number_of_edges,
                        edge=edge,
                        endpoints=(
                            int(endpoints[0]),
                            int(endpoints[1]),
                        ),
                        color=color,
                        alternating_color=alternating_color,
                        risk_red=risk_red,
                        risk_blue=risk_blue,
                        red_completions=red_completions,
                        blue_completions=blue_completions,
                        risk_driven=(
                            abs(risk_difference)
                            > self.risk_epsilon
                        ),
                        overrode_alternating=(
                            color != alternating_color
                        ),
                        red_edges=red_edges,
                        blue_edges=blue_edges,
                        alternating_decisions=alternating_decisions,
                        risk_decisions=risk_decisions,
                        risk_overrides=risk_overrides,
                    )
                )

        self._last_report = REdgeRiskConstructionReport(
            alternating_decisions=alternating_decisions,
            risk_decisions=risk_decisions,
            risk_overrides=risk_overrides,
            first_risk_step=first_risk_step,
            red_edges=red_edges,
            blue_edges=blue_edges,
        )

        return RColoring(
            graph,
            colors,
        )