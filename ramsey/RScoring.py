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

    Immutable snapshot produced by :func:`evaluate_coloring`: the total
    number of forbidden monochromatic cliques, the count attributed to
    each color, and each color's full clique edge-count histogram.

    Attributes:
        total (int): Total number of forbidden monochromatic cliques,
            summed over all colors.
        by_color (tuple[int, ...]): Number of forbidden monochromatic
            cliques found for each color index.
        histograms (tuple[NDArray[np.int64], ...]): Per-color histogram
            of clique counts, one entry per color. ``histograms[c][k]``
            is the number of indexed cliques of color ``c``'s forbidden
            size that contain exactly ``k`` edges of color ``c``. Each
            array is an owned, read-only copy.
    """

    total: int
    by_color: tuple[int, ...]
    histograms: tuple[
        NDArray[np.int64],
        ...,
    ]

    def __post_init__(self) -> None:
        """
        Validate array lengths, normalize dtypes, and freeze all fields.

        Raises:
            ValueError: If ``by_color`` and ``histograms`` have
                different lengths, or if ``total`` does not equal the
                sum of ``by_color``.
        """
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

        Args:
            color (int): Color index to look up.

        Returns:
            NDArray[np.int64]: Read-only histogram where entry ``k`` is
            the number of indexed cliques of that color containing
            exactly ``k`` edges of that color.

        Raises:
            IndexError: If ``color`` is outside the range of stored
                histograms.
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

    Args:
        coloring (RColoring): Coloring to inspect.
        clique_size (int): Clique size to look up in the coloring's
            graph's subgraph index.
        color (int): Color index whose edges are counted within each
            clique.

    Returns:
        NDArray[np.uint8]: One count per indexed clique of size
        ``clique_size``, equal to the number of that clique's edges
        currently colored ``color``.

    Raises:
        IndexError: If ``color`` is not a valid color index for the
            coloring's problem.
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

    Args:
        coloring (RColoring): Coloring to inspect.
        clique_size (int): Clique size to look up in the coloring's
            graph's subgraph index.
        color (int): Color index whose edges are counted within each
            clique.

    Returns:
        NDArray[np.int64]: Histogram of length ``edges_per_clique + 1``
        where entry ``k`` is the number of indexed cliques containing
        exactly ``k`` edges of ``color``.
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

    Args:
        coloring (RColoring): Coloring to inspect; must belong to a
            symmetric, two-color problem.
        edge_color (int): Color index whose edges are counted within
            each clique. Defaults to ``1``.

    Returns:
        NDArray[np.int64]: Histogram of clique counts by ``edge_color``
        edge count, computed over the problem's single required clique
        size.

    Raises:
        ValueError: If the coloring's problem is not a symmetric
            two-color problem.
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

    For a symmetric two-color problem, only one color's histogram is
    computed directly; the other color's histogram is its exact
    reversal, since exchanging color roles reverses the meaning of the
    all-color-zero and all-color-one histogram bins. For problems with
    more than two colors, or asymmetric problems, each color's
    histogram is computed independently over that color's own
    forbidden clique size.

    Args:
        coloring (RColoring): Coloring to score.

    Returns:
        RScoreReport: Exact total score, per-color counts, and
        per-color histograms for ``coloring``.
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

    Args:
        coloring (RColoring): Coloring to score.

    Returns:
        int: Total count of monochromatic forbidden cliques across all
        colors, equal to ``evaluate_coloring(coloring).total``.
    """
    return evaluate_coloring(coloring).total
