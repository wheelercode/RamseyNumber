"""Exact score, histogram, and monochromatic-subgraph calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .RColoring import RColoring


@dataclass(
    frozen=True,
    slots=True,
    eq=False,
)
class RScoreReport:
    """
    Exact monochromatic score and per-color supporting data.
    """

    total: int
    by_color: tuple[int, ...]
    histograms: tuple[
        NDArray[np.int64],
        ...,
    ]

    def __post_init__(self) -> None:
        if len(self.by_color) != len(self.histograms):
            raise ValueError("by_color and histograms must " "have equal lengths.")

        normalized_histograms: list[NDArray[np.int64]] = []

        for histogram in self.histograms:
            normalized = np.asarray(
                histogram,
                dtype=np.int64,
            ).copy()

            normalized.flags.writeable = False

            normalized_histograms.append(normalized)

        normalized_by_color = tuple(int(count) for count in self.by_color)

        if self.total != sum(normalized_by_color):
            raise ValueError("total must equal the sum " "of by_color counts.")

        object.__setattr__(
            self,
            "total",
            int(self.total),
        )

        object.__setattr__(
            self,
            "by_color",
            normalized_by_color,
        )

        object.__setattr__(
            self,
            "histograms",
            tuple(normalized_histograms),
        )

    def histogram_for_color(
        self,
        color: int,
    ) -> NDArray[np.int64]:
        """
        Return the edge-count histogram calculated for one color.
        """
        if color < 0 or color >= len(self.histograms):
            raise IndexError(f"Invalid color index: {color}")

        return self.histograms[color]


def count_color_edges_per_clique(
    coloring: RColoring,
    clique_size: int,
    color: int,
) -> NDArray[np.uint8]:
    """
    Count selected-color edges in every indexed clique.
    """
    if color < 0 or color >= coloring.graph.problem.n_colors:
        raise IndexError(f"Invalid color index: {color}")

    index = coloring.graph.subgraph_index(clique_size)

    return np.count_nonzero(
        (coloring.colors[index.clique_edges] == color),
        axis=1,
    ).astype(np.uint8)


def clique_histogram(
    coloring: RColoring,
    clique_size: int,
    color: int,
) -> NDArray[np.int64]:
    """
    Group indexed cliques by their selected-color edge count.
    """
    counts = count_color_edges_per_clique(
        coloring=coloring,
        clique_size=clique_size,
        color=color,
    )

    edges_per_clique = coloring.graph.problem.edges_per_clique(clique_size)

    return np.bincount(
        counts,
        minlength=edges_per_clique + 1,
    ).astype(np.int64)


def binary_histogram(
    coloring: RColoring,
    edge_color: int = 1,
) -> NDArray[np.int64]:
    """
    Return the legacy histogram for a symmetric two-color problem.
    """
    problem = coloring.graph.problem

    if problem.n_colors != 2 or not problem.is_symmetric:
        raise ValueError("binary_histogram requires a " "symmetric two-color problem.")

    clique_size = problem.required_clique_sizes[0]

    return clique_histogram(
        coloring=coloring,
        clique_size=clique_size,
        color=edge_color,
    )


def evaluate_coloring(
    coloring: RColoring,
) -> RScoreReport:
    """
    Calculate every forbidden monochromatic count exactly.
    """
    problem = coloring.graph.problem

    # A symmetric two-color problem needs only one histogram.
    # The other color's histogram is its reversal.
    if problem.n_colors == 2 and problem.is_symmetric:
        clique_size = problem.required_clique_sizes[0]

        color_one_histogram = clique_histogram(
            coloring=coloring,
            clique_size=clique_size,
            color=1,
        )

        color_zero_histogram = color_one_histogram[::-1].copy()

        by_color = (
            int(color_one_histogram[0]),
            int(color_one_histogram[-1]),
        )

        return RScoreReport(
            total=sum(by_color),
            by_color=by_color,
            histograms=(
                color_zero_histogram,
                color_one_histogram,
            ),
        )

    histograms: list[NDArray[np.int64]] = []

    by_color: list[int] = []

    for (
        color,
        clique_size,
    ) in enumerate(problem.forbidden_clique_sizes):
        histogram = clique_histogram(
            coloring=coloring,
            clique_size=clique_size,
            color=color,
        )

        histograms.append(histogram)

        by_color.append(int(histogram[-1]))

    return RScoreReport(
        total=sum(by_color),
        by_color=tuple(by_color),
        histograms=tuple(histograms),
    )


def score_coloring(
    coloring: RColoring,
) -> int:
    """
    Return the total number of forbidden monochromatic cliques.
    """
    return evaluate_coloring(coloring).total
