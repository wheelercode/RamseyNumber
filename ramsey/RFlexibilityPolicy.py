"""Adaptive edge policy that treats flexibility as search capital."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from .RActionFlexibility import calculate_action_flexibility_landscape
from .REnvironment import REnvironment
from .RFlexibility import (
    DEFAULT_FLEXIBILITY_BUDGETS,
    calculate_flexibility,
)
from .RPolicy import RPolicy


@dataclass(frozen=True, slots=True)
class RFlexibilityPolicyConfig:
    """Configure when and how the policy buys back flexibility.

    Attributes:
        budgets (tuple[int, ...]): Strictly increasing, non-negative
            damage budgets used to build flexibility profiles. Defaults
            to :data:`RFlexibility.DEFAULT_FLEXIBILITY_BUDGETS`.
        monitor_budget (int): The single budget ``b`` whose F(b) is
            monitored as the policy's maneuverability reserve; must be
            one of the values in ``budgets``.
        flexibility_floor (float): Threshold in ``[0.0, 1.0]`` on the
            monitored F(``monitor_budget``). Above this floor the
            policy follows ordinary exact greedy score.
        maximum_temporary_damage (int): When the monitored flexibility
            is below the floor, allow score-neutral actions and
            temporary worsening up to this amount, then maximize future
            F(``monitor_budget``). Positive-reward actions remain
            candidates; they simply stop receiving automatic priority
            over structurally more flexible alternatives.
    """

    budgets: tuple[int, ...] = DEFAULT_FLEXIBILITY_BUDGETS

    # F(b) monitored as the maneuverability reserve.
    monitor_budget: int = 5

    # Above this floor the policy follows ordinary exact greedy score.
    flexibility_floor: float = 0.10

    # When below the floor, allow score-neutral actions and temporary
    # worsening up to this amount, then maximize future F(b). Positive
    # actions remain candidates; they simply stop receiving automatic
    # priority over structurally more flexible alternatives.
    maximum_temporary_damage: int = 5

    def __post_init__(self) -> None:
        """Validate and normalize all configuration fields.

        Raises:
            TypeError: If ``monitor_budget`` is not an integer, if
                ``flexibility_floor`` is not numeric, or if
                ``maximum_temporary_damage`` is not an integer.
            ValueError: If ``budgets`` fails the canonical budget
                validation (see :func:`RFlexibility.calculate_flexibility`),
                if ``monitor_budget`` is not present in ``budgets``, if
                ``flexibility_floor`` is outside ``[0.0, 1.0]``, or if
                ``maximum_temporary_damage`` is negative.
        """
        # Reuse the canonical budget validation.
        probe = calculate_flexibility(
            np.asarray([0], dtype=np.int32),
            self.budgets,
        )

        if isinstance(self.monitor_budget, bool) or not isinstance(
            self.monitor_budget,
            Integral,
        ):
            raise TypeError("monitor_budget must be an integer.")

        monitor_budget = int(self.monitor_budget)

        if monitor_budget not in probe.budgets:
            raise ValueError("monitor_budget must be present in budgets.")

        if isinstance(self.flexibility_floor, bool) or not isinstance(
            self.flexibility_floor,
            Real,
        ):
            raise TypeError("flexibility_floor must be numeric.")

        flexibility_floor = float(self.flexibility_floor)

        if not 0.0 <= flexibility_floor <= 1.0:
            raise ValueError("flexibility_floor must lie between zero and one.")

        if isinstance(self.maximum_temporary_damage, bool) or not isinstance(
            self.maximum_temporary_damage,
            Integral,
        ):
            raise TypeError("maximum_temporary_damage must be an integer.")

        maximum_temporary_damage = int(self.maximum_temporary_damage)

        if maximum_temporary_damage < 0:
            raise ValueError("maximum_temporary_damage cannot be negative.")

        object.__setattr__(self, "monitor_budget", monitor_budget)
        object.__setattr__(self, "flexibility_floor", flexibility_floor)
        object.__setattr__(
            self,
            "maximum_temporary_damage",
            maximum_temporary_damage,
        )


@dataclass(frozen=True, slots=True)
class RFlexibilityPolicyDecision:
    """Explain one adaptive policy choice for experiment reporting.

    Attributes:
        edge (int): Index of the selected host-graph edge.
        mode (str): ``"score"`` if the choice followed ordinary exact
            greedy score, or ``"flexibility"`` if it was made to
            maximize future maneuverability because the monitored
            flexibility was below the configured floor.
        immediate_reward (int): Exact immediate reward (score
            reduction) of the selected edge.
        current_flexibility (float): F(``monitor_budget``) measured
            over the complete action space before selecting an edge.
        resulting_flexibility (float | None): F(``monitor_budget``)
            after taking the selected action, or ``None`` in
            ``"score"`` mode where it was not computed.
        candidate_count (int): Number of edges considered when making
            the final selection (available best-reward edges in
            ``"score"`` mode, or flexibility-landscape candidate edges
            in ``"flexibility"`` mode).
    """

    edge: int
    mode: str
    immediate_reward: int
    current_flexibility: float
    resulting_flexibility: float | None
    candidate_count: int


def select_flexibility_action(
    environment: REnvironment,
    rng: np.random.Generator,
    config: RFlexibilityPolicyConfig,
) -> RFlexibilityPolicyDecision:
    """Select one edge while preserving a configurable flexibility reserve.

    Measures the current structural flexibility F(``monitor_budget``)
    over the complete action space (not just currently available
    actions). If that flexibility is at or above
    ``config.flexibility_floor``, the policy behaves as ordinary exact
    greedy search: it picks uniformly among available edges achieving
    the maximum immediate reward (``mode="score"``).

    Otherwise, the policy treats flexibility as a resource to protect:
    it restricts candidates to available edges whose reward is at least
    ``-config.maximum_temporary_damage`` (falling back to the
    least-damaging available edges if none qualify), simulates each
    candidate flip, and selects the candidate that maximizes the
    resulting F(``monitor_budget``), breaking ties by the largest
    immediate reward and then uniformly at random (``mode="flexibility"``).

    Args:
        environment (REnvironment): Environment supplying the current
            search state, action analysis, and available-action mask.
        rng (numpy.random.Generator): Random source used to break ties
            uniformly.
        config (RFlexibilityPolicyConfig): Budgets, monitored budget,
            flexibility floor, and temporary-damage allowance governing
            the decision.

    Returns:
        RFlexibilityPolicyDecision: The selected edge together with the
        mode used and the flexibility/reward measurements behind the
        choice.

    Raises:
        RuntimeError: If no action is available in the current state.
    """
    environment_analysis = environment.analyze_actions()
    action_analysis = environment_analysis.action_analysis
    rewards = action_analysis.immediate_rewards
    available = environment_analysis.available_mask

    current = calculate_flexibility(
        rewards,
        config.budgets,
    )
    current_flexibility = current.fraction_at(config.monitor_budget)

    available_rewards = rewards[available]

    if available_rewards.size == 0:
        raise RuntimeError("No actions are available.")

    best_reward = int(available_rewards.max())

    if current_flexibility >= config.flexibility_floor:
        candidates = np.flatnonzero(
            available & (rewards == best_reward)
        ).astype(np.int32)
        edge = int(rng.choice(candidates))

        return RFlexibilityPolicyDecision(
            edge=edge,
            mode="score",
            immediate_reward=int(rewards[edge]),
            current_flexibility=current_flexibility,
            resulting_flexibility=None,
            candidate_count=len(candidates),
        )

    minimum_reward = -config.maximum_temporary_damage
    candidate_mask = available & (rewards >= minimum_reward)

    # A sufficiently rigid state can have no action inside the configured
    # damage budget. Fall back to the least-damaging available actions so
    # the policy remains total rather than deadlocking.
    if not np.any(candidate_mask):
        candidate_mask = available & (rewards == best_reward)

    landscape = calculate_action_flexibility_landscape(
        environment.state,
        analysis=action_analysis,
        budgets=config.budgets,
        candidate_mask=candidate_mask,
    )
    budget_index = landscape.budget_index(config.monitor_budget)
    future_values = landscape.resulting_fractions[:, budget_index]
    best_future = float(future_values.max())

    best_flexibility_rows = np.flatnonzero(
        np.isclose(future_values, best_future)
    )

    # Once future flexibility ties, spend as little score as possible.
    tied_rewards = landscape.score_rewards[best_flexibility_rows]
    best_tied_reward = int(tied_rewards.max())
    final_rows = best_flexibility_rows[
        tied_rewards == best_tied_reward
    ]
    row = int(rng.choice(final_rows))
    edge = int(landscape.candidate_edges[row])

    return RFlexibilityPolicyDecision(
        edge=edge,
        mode="flexibility",
        immediate_reward=int(landscape.score_rewards[row]),
        current_flexibility=current_flexibility,
        resulting_flexibility=float(future_values[row]),
        candidate_count=len(landscape.candidate_edges),
    )


@dataclass(slots=True)
class RFlexibilityPolicy(RPolicy):
    """RPolicy adapter for adaptive score/flexibility selection.

    Wraps :func:`select_flexibility_action` as an :class:`RPolicy`,
    following ordinary exact greedy score while flexibility remains
    ample and switching to flexibility-preserving selection when it
    drops below the configured floor.

    Attributes:
        rng (numpy.random.Generator): Random source used to break ties
            uniformly.
        config (RFlexibilityPolicyConfig): Budgets, monitored budget,
            flexibility floor, and temporary-damage allowance governing
            the policy's decisions.
    """

    rng: np.random.Generator
    config: RFlexibilityPolicyConfig = RFlexibilityPolicyConfig()

    @property
    def name(self) -> str:
        """str: Stable identifier for this policy, ``"adaptive-flexibility"``."""
        return "adaptive-flexibility"

    @property
    def requires_full_analysis(self) -> bool:
        """bool: Always ``True``; this policy needs the complete action analysis."""
        return True

    def select_action(self, environment: REnvironment) -> int:
        """Select an edge using :func:`select_flexibility_action`.

        Args:
            environment (REnvironment): Environment supplying the
                current search state and action analysis.

        Returns:
            int: Index of the selected host-graph edge.
        """
        return select_flexibility_action(
            environment,
            self.rng,
            self.config,
        ).edge