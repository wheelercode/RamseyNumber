"""Lifecycle and results for complete search attempts."""

from __future__ import annotations

from dataclasses import dataclass

from .RColoring import RColoring
from .REnvironment import (
    REnvironment,
    RStepResult,
)
from .RPolicy import RPolicy


@dataclass(frozen=True, slots=True)
class RSearchResult:
    """
    Immutable outcome of one complete search attempt.
    """

    policy_name: str
    objective_name: str

    initial_coloring: RColoring
    final_coloring: RColoring
    best_coloring: RColoring

    initial_score: int
    final_score: int
    best_score: int

    steps_completed: int

    terminated: bool
    truncated: bool

    step_results: tuple[RStepResult, ...] = ()

    @property
    def score_reduction(self) -> int:
        """
        Return the exact improvement from initial to final state.

        This can be negative if the final state is worse than the
        initial state.
        """
        return self.initial_score - self.final_score

    @property
    def best_score_reduction(self) -> int:
        """
        Return the exact improvement from initial to best state.
        """
        return self.initial_score - self.best_score


class RSearch:
    """
    Run an environment using one interchangeable policy.

    RSearch coordinates an episode but does not:

    - construct seed colorings;
    - choose how action rewards are calculated;
    - implement tabu memory;
    - persist results;
    - plot results; or
    - train neural networks.

    Those responsibilities belong to separate components.
    """

    def __init__(
        self,
        environment: REnvironment,
        policy: RPolicy,
    ) -> None:
        self._environment = environment
        self._policy = policy

    @property
    def environment(self) -> REnvironment:
        """
        Return the environment used for each search attempt.
        """
        return self._environment

    @property
    def policy(self) -> RPolicy:
        """
        Return the action-selection policy.
        """
        return self._policy

    def run(
        self,
        coloring: RColoring,
        *,
        record_steps: bool = False,
    ) -> RSearchResult:
        """
        Run from an explicit seed until termination or truncation.

        Parameters
        ----------
        coloring:
            Immutable seed coloring for the search attempt.

        record_steps:
            If true, preserve every RStepResult in the returned result.

            If false, only summary information and important coloring
            snapshots are retained.
        """
        if not isinstance(
            record_steps,
            bool,
        ):
            raise TypeError("record_steps must be boolean.")

        initial_state = self._environment.reset(coloring)

        initial_coloring = initial_state.coloring_snapshot()

        initial_score = initial_state.score

        recorded_steps: list[RStepResult] = []

        while not (self._environment.terminated or self._environment.truncated):
            edge = self._policy.select_action(self._environment)

            step_result = self._environment.step(
                edge,
                full_analysis=self._policy.requires_full_analysis,
            )

            if record_steps:
                recorded_steps.append(step_result)

        return RSearchResult(
            policy_name=self._policy.name,
            objective_name=self._environment.objective.name,
            initial_coloring=initial_coloring,
            final_coloring=self._environment.state.coloring_snapshot(),
            best_coloring=self._environment.best_coloring,
            initial_score=initial_score,
            final_score=self._environment.state.score,
            best_score=self._environment.best_score,
            steps_completed=self._environment.step_number,
            terminated=self._environment.terminated,
            truncated=self._environment.truncated,
            step_results=tuple(recorded_steps),
        )
