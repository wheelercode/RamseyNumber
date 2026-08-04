"""Optimization objectives and reward shaping derived from action data."""

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

    Lower energy is better.
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
        """
        ...

    @abstractmethod
    def energy(
        self,
        state: RSearchState,
    ) -> float:
        """
        Return a state energy where lower values are better.
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
        """
        ...

    @staticmethod
    def _require_current_analysis(
        state: RSearchState,
        analysis: RActionAnalysis,
    ) -> None:
        """
        Reject analysis that does not describe the supplied state.
        """
        if not analysis.applies_to(state):
            raise ValueError("Action analysis does not describe " "the current state.")


class RMonochromaticObjective(RObjective):
    """
    Use the exact monochromatic-clique score as the objective.
    """

    @property
    def name(self) -> str:
        return "monochromatic"

    def energy(
        self,
        state: RSearchState,
    ) -> float:
        return float(state.score)

    def action_rewards(
        self,
        state: RSearchState,
        analysis: RActionAnalysis,
    ) -> NDArray[np.float64]:
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
    """

    decay: float = 0.25

    def __post_init__(
        self,
    ) -> None:
        object.__setattr__(
            self,
            "decay",
            _validate_decay(self.decay),
        )

    @property
    def name(self) -> str:
        return "danger"

    def energy(
        self,
        state: RSearchState,
    ) -> float:
        return danger_energy(
            state.histogram,
            decay=self.decay,
        )

    def action_rewards(
        self,
        state: RSearchState,
        analysis: RActionAnalysis,
    ) -> NDArray[np.float64]:
        self._require_current_analysis(
            state,
            analysis,
        )

        return all_danger_rewards(
            analysis.histogram_deltas,
            decay=self.decay,
        )
