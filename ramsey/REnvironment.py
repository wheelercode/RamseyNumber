"""Search-step environment tying search state, memory, and objective together.

Defines :class:`REnvironment`, which owns one Ramsey-search episode: it
consults search memory for tabu/revisit restrictions, applies the
aspiration criterion and a deadlock fallback to keep at least one action
available, validates and applies externally selected edge flips, and
tracks the episode's best exact score and coloring. It does not itself
select actions; that is the responsibility of an :class:`RPolicy`.
"""

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

    Args:
        values (numpy.ndarray): Source values.
        dtype: NumPy dtype the returned array is coerced to.

    Returns:
        numpy.ndarray: An independent copy of ``values`` cast to
        ``dtype`` with the ``writeable`` flag cleared.
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

    Attributes:
        action_analysis (RActionAnalysis): Exact structural and score
            consequences of every edge flip.
        objective_rewards (numpy.ndarray): Float64 array, shape
            ``(number_of_actions,)``, read-only. Reward assigned to
            every action by the selected objective.
        memory_status (RMemoryStatus): Restrictions produced by tabu
            or other search memory.
        aspiration_mask (numpy.ndarray): Boolean array, shape
            ``(number_of_actions,)``, read-only. True for actions that
            would otherwise be blocked but are allowed because they
            establish a new best exact score (the aspiration
            criterion).
        available_mask (numpy.ndarray): Boolean array, shape
            ``(number_of_actions,)``, read-only. Actions currently
            permitted by the environment, after memory, aspiration,
            and deadlock fallback are combined.
        forced_fallback (bool): True when every normal action was
            blocked and the environment exposed the actions producing
            the best exact resulting score instead (deadlock
            fallback).
    """

    action_analysis: RActionAnalysis
    objective_rewards: NDArray[np.float64]
    memory_status: RMemoryStatus
    aspiration_mask: NDArray[np.bool_]
    available_mask: NDArray[np.bool_]
    forced_fallback: bool

    def __post_init__(self) -> None:
        """Validate array shapes against the action count and freeze fields.

        Raises:
            ValueError: If ``objective_rewards``, ``aspiration_mask``,
                ``available_mask``, or the memory status masks do not
                match ``action_analysis.number_of_actions`` in shape.
        """
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

    Attributes:
        edge (int): Encoded edge index that was flipped.
        previous_score (int): Exact monochromatic-clique score before
            the flip.
        score (int): Exact monochromatic-clique score after the flip.
        immediate_reward (int): Exact score reduction produced by the
            flip (``previous_score - score``); positive means the
            score improved.
        previous_energy (float): Objective energy before the flip.
        energy (float): Objective energy after the flip.
        objective_reward (float): Objective energy reduction produced
            by the flip (``previous_energy - energy``).
        best_score (int): Best exact score found in the episode so
            far, including this step.
        step_number (int): Number of transitions completed in the
            episode, including this one.
        terminated (bool): Whether the episode reached an exact score
            of zero after this step.
        truncated (bool): Whether the episode reached its step limit
            after this step.
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
        """Construct an environment bound to one graph, objective, and memory.

        No episode is active until :meth:`reset` is called; ``state``,
        ``best_coloring``, and ``best_score`` raise until then.

        Args:
            graph (RGraph): Immutable host-graph topology and clique
                index shared by every episode.
            objective (RObjective): Optimization objective used to
                score states and shape action rewards.
            memory (RMemory): Search-history memory used to compute
                per-action tabu/revisit restrictions.
            config (REnvironmentConfig | None): Environment behavior
                settings (step limit, aspiration). Defaults to
                ``REnvironmentConfig()`` when omitted.
        """
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

        Raises:
            RuntimeError: If the environment has not been reset.
        """
        if self._state is None:
            raise RuntimeError("Environment has not been reset.")

        return self._state

    @property
    def best_coloring(self) -> RColoring:
        """
        Return an immutable snapshot of the episode's best coloring.

        Raises:
            RuntimeError: If the environment has not been reset.
        """
        if self._best_coloring is None:
            raise RuntimeError("Environment has not been reset.")

        return self._best_coloring

    @property
    def best_score(self) -> int:
        """
        Return the best exact score found in the episode.

        Raises:
            RuntimeError: If the environment has not been reset.
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
        The best coloring/score are initialized from the seed, memory
        is reset, the step count returns to zero, and any cached
        action analysis is discarded.

        Args:
            coloring (RColoring): Seed coloring for the new episode.
                If it was built against a different but compatible
                graph object (same ``problem``), an equivalent
                coloring bound to this environment's graph is used
                instead.

        Returns:
            RSearchState: The freshly created mutable search state for
            the episode.

        Raises:
            ValueError: If ``coloring``'s graph belongs to a different
                problem than this environment's graph.
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

        Combines :func:`analyze_actions` (exact per-edge structural and
        score consequences), the objective's shaped rewards, current
        memory restrictions, the aspiration criterion (allowing a
        blocked action that would set a new best exact score, when
        ``config.use_aspiration`` is true), and the deadlock fallback
        (exposing the actions with the best resulting exact score if
        every action would otherwise be blocked).

        Args:
            use_cache (bool): If true and a cached analysis already
                applies to the current search-state version, return it
                instead of recomputing. Defaults to True.

        Returns:
            REnvironmentAnalysis: Complete action analysis, objective
            rewards, memory status, and derived availability masks for
            the current state.
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
        the deadlock fallback. This is the fast path used by policies
        that do not require complete action analysis (see
        ``RPolicy.requires_full_analysis``).

        Args:
            use_cache (bool): If true, reuse a cached full analysis or
                a cached fast-path mask when either already applies to
                the current search-state version. Defaults to True.

        Returns:
            numpy.ndarray: Boolean array, shape ``(number_of_edges,)``,
            read-only. True for edges currently permitted by the
            environment.
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

        Returns:
            numpy.ndarray: Int32 array of encoded edge indexes for
            which :meth:`available_action_mask_fast` is true.
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

        Applies the flip to the search state, updates search memory
        with the transition, and refreshes the episode's best score
        and coloring if the new state improves on it. Raises instead
        of applying the flip if the episode has already ended or the
        edge is not currently available.

        Args:
            edge (int): Encoded edge index selected by a search
                strategy.
            full_analysis (bool): If true, verify the selected action
                against complete all-action analysis, and cross-check
                the actual immediate and objective rewards against the
                analysis's predictions. If false, use the lightweight
                availability path and calculate the selected action's
                reward during mutation, skipping cross-checks.
                Defaults to True.

        Returns:
            RStepResult: Observable outcome of the transition,
            including scores, rewards, and updated termination state.

        Raises:
            RuntimeError: If the episode has already terminated or
                been truncated, or if ``full_analysis`` is true and
                the actual immediate or objective reward disagrees
                with the value predicted by the pre-flip analysis.
            TypeError: If ``edge`` is not an integer.
            IndexError: If ``edge`` is out of range.
            ValueError: If ``edge`` is not currently available.
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

        Args:
            available_mask (numpy.ndarray): Boolean array of actions
                available before deadlock fallback is considered.
            resulting_scores (numpy.ndarray): Exact score that results
                from taking each action.

        Returns:
            tuple[numpy.ndarray, bool]: The (possibly replaced)
            availability mask, and whether the deadlock fallback was
            applied. When ``available_mask`` already allows at least
            one action, it is returned unchanged with ``False``.
            Otherwise every action tied for the best (lowest)
            resulting score is made available, and ``True`` is
            returned.
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
