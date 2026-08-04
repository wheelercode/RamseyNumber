"""Independent verification of colorings and incremental state."""

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
    """

    coloring_hash: str
    score_report: RScoreReport

    @property
    def ramsey_free(self) -> bool:
        """
        Return whether no forbidden monochromatic clique exists.
        """
        return self.score_report.total == 0


@dataclass(
    frozen=True,
    slots=True,
)
class RStateVerification:
    """
    Independent consistency result for incremental search state.
    """

    coloring: RColoringVerification
    errors: tuple[str, ...]

    @property
    def consistent(self) -> bool:
        """
        Return whether every incremental field was correct.
        """
        return not self.errors

    @property
    def ramsey_free(self) -> bool:
        """
        Return the independently calculated Ramsey-free result.
        """
        return self.coloring.ramsey_free

    def require_consistent(
        self,
    ) -> None:
        """
        Raise RuntimeError when any state inconsistency exists.
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
    histogram, or score.
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
