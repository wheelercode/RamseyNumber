"""Independent verification of colorings and incremental search state.

Recomputes exact scores directly from a coloring's edge colors, bypassing
any incrementally maintained bookkeeping, so that results produced by
faster incremental code paths (such as :class:`~ramsey.RState.RSearchState`)
can be cross-checked for correctness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .RColoring import RColoring
from .RScoring import (
    RScoreReport,
    count_color_edges_per_clique,
    evaluate_coloring,
)
from .RState import RSearchState


@dataclass(
    frozen=True,
    slots=True,
)
class RColoringVerification:
    """
    Independent exact result for one immutable coloring.

    Attributes:
        coloring_hash (str): Exact content hash of the verified coloring,
            as returned by :meth:`~ramsey.RColoring.RColoring.exact_hash`.
        score_report (RScoreReport): Exact monochromatic score, freshly
            computed by :func:`~ramsey.RScoring.evaluate_coloring`.
    """

    coloring_hash: str
    score_report: RScoreReport

    @property
    def ramsey_free(self) -> bool:
        """
        bool: Whether no forbidden monochromatic clique exists, i.e.
        ``score_report.total == 0``.
        """
        return self.score_report.total == 0


@dataclass(
    frozen=True,
    slots=True,
)
class RStateVerification:
    """
    Independent consistency result for incremental search state.

    Attributes:
        coloring (RColoringVerification): Independent verification of
            the state's current coloring snapshot.
        errors (tuple[str, ...]): Human-readable descriptions of every
            incremental field found to disagree with an independently
            recomputed value; empty when the state is fully consistent.
    """

    coloring: RColoringVerification
    errors: tuple[str, ...]

    @property
    def consistent(self) -> bool:
        """
        bool: Whether every incremental field matched its independently
        recomputed value, i.e. ``errors`` is empty.
        """
        return not self.errors

    @property
    def ramsey_free(self) -> bool:
        """
        bool: The independently calculated Ramsey-free result, from
        ``coloring.ramsey_free``.
        """
        return self.coloring.ramsey_free

    def require_consistent(
        self,
    ) -> None:
        """
        Raise an error when any state inconsistency was found.

        Raises:
            RuntimeError: If ``errors`` is nonempty, with a message
                listing every detected inconsistency.
        """
        if self.errors:
            details = "; ".join(self.errors)

            raise RuntimeError(
                "Incremental search state " f"is inconsistent: {details}"
            )


def verify_coloring(
    coloring: RColoring,
) -> RColoringVerification:
    """
    Independently score one immutable coloring.

    Args:
        coloring (RColoring): Coloring to verify.

    Returns:
        RColoringVerification: The coloring's exact content hash paired
        with a freshly computed exact score report.
    """
    return RColoringVerification(
        coloring_hash=coloring.exact_hash(),
        score_report=evaluate_coloring(coloring),
    )


def verify_search_state(
    state: RSearchState,
) -> RStateVerification:
    """
    Recompute every incremental field from the current colors.

    This function does not trust the state's stored clique counts,
    histogram, or score. It takes an immutable snapshot of the state's
    current coloring, independently recomputes the color-one edge count
    per clique, the resulting histogram, and the total score, and
    compares each against the state's incrementally maintained values.

    Args:
        state (RSearchState): Incremental search state to verify.

    Returns:
        RStateVerification: The independent coloring verification
        together with a description of every incremental field found
        to disagree with the recomputed values (empty when consistent).
    """
    snapshot = state.coloring_snapshot()

    coloring_verification = verify_coloring(snapshot)

    expected_counts = count_color_edges_per_clique(
        coloring=snapshot,
        clique_size=state.clique_size,
        color=1,
    )

    expected_histogram = np.bincount(
        expected_counts,
        minlength=state.edges_per_clique + 1,
    ).astype(np.int64)

    expected_score = int(expected_histogram[0] + expected_histogram[-1])

    errors: list[str] = []

    if not np.array_equal(
        state.color_one_counts,
        expected_counts,
    ):
        errors.append("color-one clique counts " "are incorrect")

    if not np.array_equal(
        state.histogram,
        expected_histogram,
    ):
        errors.append("histogram is incorrect")

    if state.score != expected_score:
        errors.append(f"score is {state.score}; " f"expected {expected_score}")

    if coloring_verification.score_report.total != expected_score:
        errors.append(
            "independent score report " "disagrees with the recomputed " "binary score"
        )

    return RStateVerification(
        coloring=coloring_verification,
        errors=tuple(errors),
    )
