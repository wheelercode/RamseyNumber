"""Validation and application of state transitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .RAction import (
    RActionAnalysis,
    analyze_actions,
    immediate_rewards_for_edges,
)
from .RColoring import RColoring
from .REnvironmentConfig import REnvironmentConfig
from .REnvironmentMemory import (
    RMemory,
    RMemoryStatus,
)
from .RGraph import RGraph
from .RObjective import RObjective
from .RState import RSearchState


def _read_only(
    values: NDArray,
    dtype,
) -> NDArray:
    """
    Return an owned, typed, read-only array.
    """
    result = np.asarray(
        values,
        dtype=dtype,
    ).copy()

    result.flags.writeable = False

    return result


@dataclass(
    frozen=True,
    slots=True,
    eq=False,
)
class REnvironmentAnalysis:
    """
    Complete action information for the current environment state.

    action_analysis:
        Exact structural and score consequences of every edge flip.

    objective_rewards:
        Reward assigned to every action by the selected objective.

    memory_status:
        Restrictions produced by tabu or other search memory.

    aspiration_mask:
        Blocked actions allowed because they establish a new best
        exact score.

    available_mask:
        Actions currently permitted by the environment.

    forced_fallback:
        True when every normal action was blocked and the environment
        exposed the actions producing the best exact resulting score.
    """

    action_analysis: RActionAnalysis
    objective_rewards: NDArray[np.float64]
    memory_status: RMemoryStatus
    aspiration_mask: NDArray[np.bool_]
    available_mask: NDArray[np.bool_]
    forced_fallback: bool

    def __post_init__(self) -> None:
        number_of_actions = self.action_analysis.number_of_actions

        expected_shape = (number_of_actions,)

        objective_rewards = _read_only(
            self.objective_rewards,
            np.float64,
        )

        aspiration_mask = _read_only(
            self.aspiration_mask,
            np.bool_,
        )

        available_mask = _read_only(
            self.available_mask,
            np.bool_,
        )

        if objective_rewards.shape != expected_shape:
            raise ValueError("objective_rewards has the wrong shape.")

        if aspiration_mask.shape != expected_shape:
            raise ValueError("aspiration_mask has the wrong shape.")

        if available_mask.shape != expected_shape:
            raise ValueError("available_mask has the wrong shape.")

        if self.memory_status.blocked_mask.shape != expected_shape:
            raise ValueError("Memory masks have the wrong shape.")

        object.__setattr__(
            self,
            "objective_rewards",
            objective_rewards,
        )

        object.__setattr__(
            self,
            "aspiration_mask",
            aspiration_mask,
        )

        object.__setattr__(
            self,
            "available_mask",
            available_mask,
        )

        object.__setattr__(
            self,
            "forced_fallback",
            bool(self.forced_fallback),
        )


@dataclass(
    frozen=True,
    slots=True,
)
class RStepResult:
    """
    Observable result of one environment transition.
    """

    edge: int

    previous_score: int
    score: int
    immediate_reward: int

    previous_energy: float
    energy: float
    objective_reward: float

    best_score: int
    step_number: int

    terminated: bool
    truncated: bool


class REnvironment:
    """
    Own one Ramsey-search episode without choosing actions.

    The environment:

    - owns the mutable search state;
    - asks memory which actions are blocked;
    - applies aspiration and deadlock fallback;
    - validates and applies selected edge flips;
    - tracks the best exact score and coloring;
    - determines episode termination and truncation.

    It deliberately does not select an action. A greedy search,
    neural policy, evolutionary algorithm, or other search strategy
    must select from the environment's available actions.
    """

    def __init__(
        self,
        graph: RGraph,
        objective: RObjective,
        memory: RMemory,
        config: REnvironmentConfig | None = None,
    ) -> None:
        self._graph = graph
        self._objective = objective
        self._memory = memory

        self._config = config if config is not None else REnvironmentConfig()

        self._state: RSearchState | None = None

        self._best_coloring: RColoring | None = None

        self._best_score: int | None = None

        self._step_number = 0

        self._cached_analysis: REnvironmentAnalysis | None = None

        self._cached_available_mask: NDArray[np.bool_] | None = None

    @property
    def graph(self) -> RGraph:
        """
        Return the immutable graph topology.
        """
        return self._graph

    @property
    def objective(self) -> RObjective:
        """
        Return the optimization objective.
        """
        return self._objective

    @property
    def memory(self) -> RMemory:
        """
        Return the environment's search memory.
        """
        return self._memory

    @property
    def config(self) -> REnvironmentConfig:
        """
        Return the immutable environment configuration.
        """
        return self._config

    @property
    def state(self) -> RSearchState:
        """
        Return the active mutable search state.
        """
        if self._state is None:
            raise RuntimeError("Environment has not been reset.")

        return self._state

    @property
    def best_coloring(self) -> RColoring:
        """
        Return an immutable snapshot of the episode's best coloring.
        """
        if self._best_coloring is None:
            raise RuntimeError("Environment has not been reset.")

        return self._best_coloring

    @property
    def best_score(self) -> int:
        """
        Return the best exact score found in the episode.
        """
        if self._best_score is None:
            raise RuntimeError("Environment has not been reset.")

        return self._best_score

    @property
    def step_number(self) -> int:
        """
        Return the number of completed transitions.
        """
        return self._step_number

    @property
    def terminated(self) -> bool:
        """
        Return whether an exact score of zero has been reached.
        """
        return self._state is not None and self._state.score == 0

    @property
    def truncated(self) -> bool:
        """
        Return whether the episode reached its step limit.
        """
        return (
            self._state is not None
            and not self.terminated
            and self._step_number >= self._config.max_steps
        )

    def reset(
        self,
        coloring: RColoring,
    ) -> RSearchState:
        """
        Begin a new episode from an explicit seed coloring.

        Seed generation is deliberately kept outside the environment.
        This allows random, constructive, database, mutated, or
        manually supplied seed generators to use the same environment.
        """
        if coloring.graph is not self._graph:
            if coloring.graph.problem != self._graph.problem:
                raise ValueError(
                    "Coloring problem does not match " "environment graph."
                )

            coloring = RColoring(
                self._graph,
                coloring.colors,
            )

        self._state = RSearchState(coloring)

        self._best_coloring = self._state.coloring_snapshot()

        self._best_score = self._state.score

        self._step_number = 0

        self._memory.reset(self._state)

        self._clear_cache()

        return self._state

    def analyze_actions(
        self,
        use_cache: bool = True,
    ) -> REnvironmentAnalysis:
        """
        Calculate exact consequences and availability for all actions.
        """
        if use_cache and self._cached_analysis is not None:
            cached_action_analysis = self._cached_analysis.action_analysis

            if cached_action_analysis.applies_to(self.state):
                return self._cached_analysis

        action_analysis = analyze_actions(self.state)

        objective_rewards = self._objective.action_rewards(
            self.state,
            action_analysis,
        )

        memory_status = self._memory.status(
            self.state,
            self._step_number,
        )

        if self._config.use_aspiration:
            aspiration_mask = action_analysis.resulting_scores < self.best_score
        else:
            aspiration_mask = np.zeros(
                self._graph.number_of_edges,
                dtype=np.bool_,
            )

        # A blocked action becomes available if aspiration applies.
        available_mask = ~(memory_status.blocked_mask & ~aspiration_mask)

        (
            available_mask,
            forced_fallback,
        ) = self._ensure_available(
            available_mask,
            action_analysis.resulting_scores,
        )

        result = REnvironmentAnalysis(
            action_analysis=action_analysis,
            objective_rewards=objective_rewards,
            memory_status=memory_status,
            aspiration_mask=aspiration_mask,
            available_mask=available_mask,
            forced_fallback=forced_fallback,
        )

        self._cached_analysis = result

        self._cached_available_mask = result.available_mask

        return result

    def available_action_mask_fast(
        self,
        use_cache: bool = True,
    ) -> NDArray[np.bool_]:
        """
        Return availability without full all-action analysis.

        Exact score rewards are calculated only for blocked edges
        when aspiration must be evaluated. If every action is blocked,
        exact score rewards are calculated for all edges to provide
        the deadlock fallback.
        """
        if use_cache:
            if self._cached_analysis is not None:
                cached_action_analysis = self._cached_analysis.action_analysis

                if cached_action_analysis.applies_to(self.state):
                    return self._cached_analysis.available_mask

            if self._cached_available_mask is not None:
                return self._cached_available_mask

        memory_status = self._memory.status(
            self.state,
            self._step_number,
        )

        blocked_mask = memory_status.blocked_mask

        available_mask = ~blocked_mask

        # Aspiration allows a blocked action if it creates a new
        # episode-best exact score.
        if self._config.use_aspiration and np.any(blocked_mask):
            blocked_edges = np.flatnonzero(blocked_mask).astype(np.int32)

            blocked_rewards = immediate_rewards_for_edges(
                self.state,
                blocked_edges,
            )

            blocked_resulting_scores = self.state.score - blocked_rewards

            aspiration = blocked_resulting_scores < self.best_score

            available_mask[blocked_edges[aspiration]] = True

        # The environment must never deadlock.
        if not np.any(available_mask):
            all_edges = np.arange(
                self._graph.number_of_edges,
                dtype=np.int32,
            )

            all_rewards = immediate_rewards_for_edges(
                self.state,
                all_edges,
            )

            resulting_scores = self.state.score - all_rewards

            best_resulting_score = int(resulting_scores.min())

            available_mask = resulting_scores == best_resulting_score

        available_mask = _read_only(
            available_mask,
            np.bool_,
        )

        self._cached_available_mask = available_mask

        return available_mask

    def available_actions(
        self,
    ) -> NDArray[np.int32]:
        """
        Return the indexes of currently available edge actions.
        """
        return np.flatnonzero(self.available_action_mask_fast()).astype(np.int32)

    def step(
        self,
        edge: int,
        *,
        full_analysis: bool = True,
    ) -> RStepResult:
        """
        Validate and apply one externally selected edge flip.

        Parameters
        ----------
        edge:
            Encoded edge index selected by a search strategy.

        full_analysis:
            If true, verify the selected action against complete
            all-action analysis.

            If false, use the lightweight availability path and
            calculate the selected action's reward during mutation.
        """
        if self.terminated or self.truncated:
            raise RuntimeError("The environment episode has ended.")

        if isinstance(edge, bool) or not isinstance(
            edge,
            (int, np.integer),
        ):
            raise TypeError("edge must be an integer.")

        edge = int(edge)

        if edge < 0 or edge >= self._graph.number_of_edges:
            raise IndexError(f"Invalid edge index: {edge}")

        if full_analysis:
            analysis = self.analyze_actions()

            available_mask = analysis.available_mask

            predicted_immediate_reward: int | None = int(
                analysis.action_analysis.immediate_rewards[edge]
            )

            predicted_objective_reward: float | None = float(
                analysis.objective_rewards[edge]
            )
        else:
            available_mask = self.available_action_mask_fast()

            predicted_immediate_reward = None
            predicted_objective_reward = None

        if not available_mask[edge]:
            raise ValueError(f"Edge {edge} is currently unavailable.")

        previous_score = self.state.score

        previous_energy = self._objective.energy(self.state)

        immediate_reward = self.state.apply_edge_flip(edge)

        if (
            predicted_immediate_reward is not None
            and immediate_reward != predicted_immediate_reward
        ):
            raise RuntimeError(
                "Predicted and actual immediate rewards "
                "disagree: "
                f"{predicted_immediate_reward} versus "
                f"{immediate_reward}."
            )

        energy = self._objective.energy(self.state)

        objective_reward = previous_energy - energy

        if predicted_objective_reward is not None and not np.isclose(
            objective_reward,
            predicted_objective_reward,
            rtol=1.0e-10,
            atol=1.0e-10,
        ):
            raise RuntimeError(
                "Predicted and actual objective rewards "
                "disagree: "
                f"{predicted_objective_reward} versus "
                f"{objective_reward}."
            )

        self._step_number += 1

        self._memory.record_transition(
            edge,
            self.state,
            self._step_number,
        )

        if self.state.score < self.best_score:
            self._best_score = self.state.score

            self._best_coloring = self.state.coloring_snapshot()

        self._clear_cache()

        return RStepResult(
            edge=edge,
            previous_score=previous_score,
            score=self.state.score,
            immediate_reward=immediate_reward,
            previous_energy=previous_energy,
            energy=energy,
            objective_reward=objective_reward,
            best_score=self.best_score,
            step_number=self._step_number,
            terminated=self.terminated,
            truncated=self.truncated,
        )

    @staticmethod
    def _ensure_available(
        available_mask: NDArray[np.bool_],
        resulting_scores: NDArray[np.integer],
    ) -> tuple[
        NDArray[np.bool_],
        bool,
    ]:
        """
        Apply the exact-score deadlock fallback when necessary.
        """
        if np.any(available_mask):
            return (
                available_mask,
                False,
            )

        best_resulting_score = int(resulting_scores.min())

        return (
            resulting_scores == best_resulting_score,
            True,
        )

    def _clear_cache(self) -> None:
        """
        Discard calculations tied to an earlier state or memory.
        """
        self._cached_analysis = None
        self._cached_available_mask = None
