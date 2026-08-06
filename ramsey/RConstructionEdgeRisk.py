"""Incremental edge construction guided by partial-K5 future risk."""

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
    """Diagnostics from one completed edge-risk construction."""

    alternating_decisions: int
    risk_decisions: int
    risk_overrides: int
    first_risk_step: int | None
    red_edges: int
    blue_edges: int

    @property
    def total_edges(self) -> int:
        return self.red_edges + self.blue_edges


@dataclass(frozen=True, slots=True)
class REdgeRiskConstructionProgress:
    """One observable edge-assignment event during construction."""

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
        return self.step_number == self.total_edges

    @property
    def unavoidable(self) -> bool:
        """Return whether either color completes a monochromatic K5."""
        return self.red_completions > 0 and self.blue_completions > 0

    @property
    def created_monochromatic_k5s(self) -> int:
        """Return the K5s completed by the color actually chosen."""
        if self.color == 0:
            return self.red_completions

        return self.blue_completions


@dataclass(slots=True)
class REdgeRiskConstruction(RConstruction):
    """
    Color K43 one edge at a time while minimizing partial-K5 risk.

    Edges are processed in one random order.  Step parity proposes the
    globally balanced red/blue alternating color.  If the partial K5s
    containing the current edge give one candidate color strictly less
    expected future monochromatic risk, that color overrides the
    alternating proposal.

    A partial K5 contributes risk only after at least
    ``minimum_pressure_edges`` edges of one color have been assigned
    while none of the opposite color have been assigned.
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
        return (
            "edge-risk-"
            f"pressure-{self.minimum_pressure_edges}"
        )

    @property
    def last_report(
        self,
    ) -> REdgeRiskConstructionReport | None:
        return self._last_report

    def construct(
        self,
        graph: RGraph,
    ) -> RColoring:
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