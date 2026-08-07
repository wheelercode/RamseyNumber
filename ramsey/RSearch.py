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

    Attributes:
        policy_name (str): Name of the policy that selected actions.
        objective_name (str): Name of the objective the environment
            used.
        initial_coloring (RColoring): Seed coloring the episode began
            from.
        final_coloring (RColoring): Coloring at the moment the episode
            ended (termination or truncation); not necessarily the
            best coloring seen.
        best_coloring (RColoring): Coloring with the best (lowest)
            exact score seen during the episode.
        initial_score (int): Exact score of ``initial_coloring``.
        final_score (int): Exact score of ``final_coloring``.
        best_score (int): Exact score of ``best_coloring``.
        steps_completed (int): Number of edge flips applied.
        terminated (bool): Whether the episode ended because an exact
            score of zero was reached.
        truncated (bool): Whether the episode ended because it reached
            its step limit.
        step_results (tuple[RStepResult, ...]): Per-step transition
            results, populated only when the run was requested with
            ``record_steps=True``; empty otherwise.
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
        """Bind a search coordinator to one environment and policy.

        Args:
            environment (REnvironment): Environment each attempt is
                run against.
            policy (RPolicy): Strategy used to select the action taken
                at every step.
        """
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

        Resets the environment to ``coloring``, then repeatedly asks
        the policy to select an available action and applies it to the
        environment until the episode terminates (exact score reaches
        zero) or is truncated (the step limit is reached).

        Args:
            coloring (RColoring): Immutable seed coloring for the
                search attempt.
            record_steps (bool): If true, preserve every
                ``RStepResult`` in the returned result. If false, only
                summary information and important coloring snapshots
                are retained. Defaults to False.

        Returns:
            RSearchResult: Immutable summary of the completed attempt,
            including initial, final, and best colorings/scores.

        Raises:
            TypeError: If ``record_steps`` is not a ``bool``.
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
