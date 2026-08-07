"""Optimization objectives and reward shaping derived from action data.

An objective assigns comparable numeric values to search states and to
candidate actions on those states, so policies can be built independently
of what "better" means. Two concrete objectives are provided:
:class:`RMonochromaticObjective`, which uses the exact monochromatic-K5
score directly, and :class:`RDangerObjective`, a heuristic that instead
uses a graded, distance-weighted "danger energy" over the color-one-count
histogram so that near-monochromatic (almost-violating) cliques also
contribute, not just fully monochromatic ones.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .RAction import RActionAnalysis
from .RState import RSearchState


def _validate_decay(
    decay: float,
) -> float:
    """
    Return a validated danger-decay value.

    Args:
        decay (float): Candidate decay rate.

    Returns:
        float: ``decay`` coerced to ``float``.

    Raises:
        ValueError: If ``decay`` is not between zero and one, inclusive.
    """
    decay = float(decay)

    if not 0.0 <= decay <= 1.0:
        raise ValueError("decay must be between zero and one.")

    return decay


def minority_histogram(
    histogram: NDArray[np.integer],
) -> NDArray[np.int64]:
    """
    Combine color-reversed histogram bins by minority count.

    For an eleven-bin K5 histogram, the result contains:

        monochromatic,
        one minority edge,
        two minority edges,
        three minority edges,
        four minority edges,
        balanced five-versus-five.

    Bin ``k`` and bin ``last_bin - k`` (the color-reversed count) are
    symmetric: both describe cliques with ``k`` edges of the minority
    color, just with red and blue swapped. Summing each such pair
    collapses a general clique_size-edge histogram into
    ``last_bin // 2 + 1`` minority-count bins, with the central,
    self-symmetric bin (if ``last_bin`` is even) left unsummed.

    Args:
        histogram (NDArray[np.integer]): One-dimensional histogram of
            clique counts by color-one edge count, with at least two
            bins.

    Returns:
        NDArray[np.int64]: The minority-count histogram, of length
        ``(len(histogram) - 1) // 2 + 1``.

    Raises:
        ValueError: If ``histogram`` is not one-dimensional or has fewer
            than two bins.
    """
    histogram = np.asarray(histogram)

    if histogram.ndim != 1 or histogram.size < 2:
        raise ValueError(
            "histogram must be a " "one-dimensional array with " "at least two bins."
        )

    last_bin = len(histogram) - 1

    minority_bins = last_bin // 2 + 1

    result = np.empty(
        minority_bins,
        dtype=np.int64,
    )

    for minority_count in range(minority_bins):
        opposite = last_bin - minority_count

        if minority_count == opposite:
            result[minority_count] = histogram[minority_count]

        else:
            result[minority_count] = histogram[minority_count] + histogram[opposite]

    return result


def danger_weights(
    number_of_bins: int,
    decay: float = 0.25,
) -> NDArray[np.float64]:
    """
    Return symmetric histogram weights for graded danger.

    Bin ``p`` receives weight ``decay ** distance``, where ``distance``
    is ``p``'s distance to the nearer monochromatic extreme (bin 0 or bin
    ``number_of_bins - 1``). The two monochromatic bins therefore always
    receive weight 1, and weights fall off geometrically at rate
    ``decay`` moving toward the balanced center bin, symmetrically from
    both ends.

    Args:
        number_of_bins (int): Number of histogram bins (``edges_per_clique
            + 1`` for a clique's color-one-count histogram).
        decay (float): Geometric decay rate in ``[0, 1]`` applied per unit
            of distance from a monochromatic extreme. Defaults to 0.25.

    Returns:
        NDArray[np.float64]: Weight for each bin, shape
        ``(number_of_bins,)``.

    Raises:
        ValueError: If ``number_of_bins`` is less than two, or if
            ``decay`` is outside ``[0, 1]``.
    """
    decay = _validate_decay(decay)

    if number_of_bins < 2:
        raise ValueError("number_of_bins must be at least two.")

    positions = np.arange(
        number_of_bins,
        dtype=np.int32,
    )

    distances = np.minimum(
        positions,
        (number_of_bins - 1 - positions),
    )

    return np.power(
        decay,
        distances,
        dtype=np.float64,
    )


def danger_energy(
    histogram: NDArray[np.integer],
    decay: float = 0.25,
) -> float:
    """
    Return the graded global danger energy of one histogram.

    Lower energy is better. Computed as the dot product of the histogram
    counts with :func:`danger_weights`, so cliques closer to
    monochromatic (in either color) contribute more energy than cliques
    near the balanced center, and fully monochromatic cliques contribute
    the most.

    Args:
        histogram (NDArray[np.integer]): One-dimensional histogram of
            clique counts by color-one edge count, with at least two
            bins.
        decay (float): Geometric decay rate in ``[0, 1]`` passed to
            :func:`danger_weights`. Defaults to 0.25.

    Returns:
        float: The weighted sum of histogram counts.

    Raises:
        ValueError: If ``histogram`` is not one-dimensional or has fewer
            than two bins.
    """
    histogram = np.asarray(histogram)

    if histogram.ndim != 1 or histogram.size < 2:
        raise ValueError(
            "histogram must be a " "one-dimensional array with " "at least two bins."
        )

    weights = danger_weights(
        len(histogram),
        decay=decay,
    )

    return float(histogram @ weights)


def all_danger_rewards(
    histogram_deltas: NDArray[np.integer],
    decay: float = 0.25,
) -> NDArray[np.float64]:
    """
    Return the danger-energy reduction for every action.

    Positive reward means the action reduces danger energy.

    Args:
        histogram_deltas (NDArray[np.integer]): Two-dimensional array of
            per-action histogram deltas, shape ``(number_of_actions,
            number_of_bins)``, as produced for example by
            :func:`ramsey.RAction.all_histogram_deltas` or
            :attr:`ramsey.RK5Action.RK5PatternAnalysis.histogram_deltas`.
        decay (float): Geometric decay rate in ``[0, 1]`` passed to
            :func:`danger_weights`. Defaults to 0.25.

    Returns:
        NDArray[np.float64]: Danger-energy reduction for each action,
        shape ``(number_of_actions,)``. Positive values indicate reduced
        danger (an improving action); negative values indicate increased
        danger.

    Raises:
        ValueError: If ``histogram_deltas`` is not two-dimensional or has
            fewer than two bins.
    """
    histogram_deltas = np.asarray(histogram_deltas)

    if histogram_deltas.ndim != 2 or histogram_deltas.shape[1] < 2:
        raise ValueError(
            "histogram_deltas must be a "
            "two-dimensional array with "
            "at least two bins."
        )

    weights = danger_weights(
        histogram_deltas.shape[1],
        decay=decay,
    )

    energy_changes = histogram_deltas @ weights

    return -energy_changes


class RObjective(ABC):
    """
    Assign comparable values to states and candidate actions.

    Every objective follows the same convention:

        lower state energy is better;
        positive action reward is better.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the stable objective name.

        Returns:
            str: A short, stable identifier for this objective (e.g. for
            logging or reporting which objective produced a result).
        """
        ...

    @abstractmethod
    def energy(
        self,
        state: RSearchState,
    ) -> float:
        """
        Return a state energy where lower values are better.

        Args:
            state (RSearchState): Search state to evaluate.

        Returns:
            float: The state's energy under this objective.
        """
        ...

    @abstractmethod
    def action_rewards(
        self,
        state: RSearchState,
        analysis: RActionAnalysis,
    ) -> NDArray[np.float64]:
        """
        Return the energy reduction produced by every action.

        Args:
            state (RSearchState): Search state the actions would be
                applied to.
            analysis (RActionAnalysis): Precomputed exact action analysis
                for ``state``; must currently describe ``state`` (see
                :meth:`ramsey.RAction.RActionAnalysis.applies_to`).

        Returns:
            NDArray[np.float64]: Reward for each action, where a positive
            value means the action reduces this objective's energy.
        """
        ...

    @staticmethod
    def _require_current_analysis(
        state: RSearchState,
        analysis: RActionAnalysis,
    ) -> None:
        """
        Reject analysis that does not describe the supplied state.

        Args:
            state (RSearchState): Search state the analysis should
                describe.
            analysis (RActionAnalysis): Action analysis to validate.

        Raises:
            ValueError: If ``analysis`` does not currently describe
                ``state`` (stale version or different state).
        """
        if not analysis.applies_to(state):
            raise ValueError("Action analysis does not describe " "the current state.")


class RMonochromaticObjective(RObjective):
    """
    Use the exact monochromatic-clique score as the objective.

    Energy is exactly the state's monochromatic score (the count of
    fully monochromatic K5s); action rewards are the exact score
    reductions already computed by action analysis, with no additional
    shaping.
    """

    @property
    def name(self) -> str:
        """str: The objective name, ``"monochromatic"``."""
        return "monochromatic"

    def energy(
        self,
        state: RSearchState,
    ) -> float:
        """Return the state's exact monochromatic score.

        Args:
            state (RSearchState): Search state to evaluate.

        Returns:
            float: ``state.score`` as a float; zero means a valid
            (violation-free) coloring has been found.
        """
        return float(state.score)

    def action_rewards(
        self,
        state: RSearchState,
        analysis: RActionAnalysis,
    ) -> NDArray[np.float64]:
        """Return each action's exact score reduction.

        Args:
            state (RSearchState): Search state the actions would be
                applied to.
            analysis (RActionAnalysis): Precomputed exact action analysis
                for ``state``; must currently describe ``state``.

        Returns:
            NDArray[np.float64]: A float64 copy of
            ``analysis.immediate_rewards``.

        Raises:
            ValueError: If ``analysis`` does not currently describe
                ``state``.
        """
        self._require_current_analysis(
            state,
            analysis,
        )

        return analysis.immediate_rewards.astype(
            np.float64,
            copy=True,
        )


@dataclass(
    frozen=True,
    slots=True,
)
class RDangerObjective(RObjective):
    """
    Use graded distance from monochromatic cliques as energy.

    Unlike :class:`RMonochromaticObjective`, this is a heuristic
    objective: it does not count only exact monochromatic violations but
    also weighs near-monochromatic cliques, using
    :func:`danger_energy`/:func:`all_danger_rewards` over the state's
    color-one-count histogram. This can guide a policy through plateaus
    where the exact score is unchanged but the coloring is moving cliques
    closer to (or further from) monochromatic.

    Attributes:
        decay (float): Geometric decay rate in ``[0, 1]`` controlling how
            quickly per-bin danger weight falls off moving away from the
            monochromatic extremes toward the balanced center. Defaults
            to 0.25.
    """

    decay: float = 0.25

    def __post_init__(
        self,
    ) -> None:
        """Validate and normalize ``decay``.

        Raises:
            ValueError: If ``decay`` is outside ``[0, 1]``.
        """
        object.__setattr__(
            self,
            "decay",
            _validate_decay(self.decay),
        )

    @property
    def name(self) -> str:
        """str: The objective name, ``"danger"``."""
        return "danger"

    def energy(
        self,
        state: RSearchState,
    ) -> float:
        """Return the state's graded danger energy.

        Args:
            state (RSearchState): Search state to evaluate.

        Returns:
            float: ``danger_energy(state.histogram, decay=self.decay)``;
            lower values indicate fewer and less-nearly-monochromatic
            cliques.
        """
        return danger_energy(
            state.histogram,
            decay=self.decay,
        )

    def action_rewards(
        self,
        state: RSearchState,
        analysis: RActionAnalysis,
    ) -> NDArray[np.float64]:
        """Return each action's danger-energy reduction.

        Args:
            state (RSearchState): Search state the actions would be
                applied to.
            analysis (RActionAnalysis): Precomputed exact action analysis
                for ``state``; must currently describe ``state``.

        Returns:
            NDArray[np.float64]: Danger-energy reduction for each action,
            from :func:`all_danger_rewards` applied to
            ``analysis.histogram_deltas``.

        Raises:
            ValueError: If ``analysis`` does not currently describe
                ``state``.
        """
        self._require_current_analysis(
            state,
            analysis,
        )

        return all_danger_rewards(
            analysis.histogram_deltas,
            decay=self.decay,
        )
