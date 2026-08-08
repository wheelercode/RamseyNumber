"""Exact-score greedy edge selection with configurable greediness."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real

import numpy as np

from .REnvironment import REnvironment
from .RPolicy import RPolicy


@dataclass(frozen=True, slots=True)
class RConfigurableGreedyPolicyConfig:
    """Configure which positive exact-score reward percentile is selected.

    Args:
        greediness (float): Position in the currently available positive
            exact-score reward distribution. ``1.0`` selects the maximum
            positive reward and is ordinary exact-score greedy search.
            ``0.0`` selects the minimum reward that is still positive.
    """

    greediness: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.greediness, bool) or not isinstance(
            self.greediness,
            Real,
        ):
            raise TypeError("greediness must be a real number.")

        greediness = float(self.greediness)

        if not np.isfinite(greediness):
            raise ValueError("greediness must be finite.")

        if not 0.0 <= greediness <= 1.0:
            raise ValueError("greediness must be between zero and one.")

        object.__setattr__(self, "greediness", greediness)


@dataclass(frozen=True, slots=True)
class RConfigurableGreedyDecision:
    """Describe one configurable-greedy edge selection.

    Attributes:
        edge (int): Selected host-edge index.
        exact_reward (int): Exact reduction in monochromatic K5 score.
        positive_action_count (int): Number of available improving edges.
        minimum_positive_reward (int): Smallest currently positive reward.
        maximum_positive_reward (int): Largest currently positive reward.
        reward_rank (int): Zero-based rank selected in the sorted positive
            edge-reward distribution.
    """

    edge: int
    exact_reward: int
    positive_action_count: int
    minimum_positive_reward: int
    maximum_positive_reward: int
    reward_rank: int


@dataclass(slots=True)
class RConfigurableGreedyPolicy(RPolicy):
    """Choose a configurable percentile of positive exact-score actions.

    The percentile is taken over edge actions rather than unique reward
    values. After the percentile determines an actual reward level, ties at
    that reward level are broken uniformly at random.
    """

    rng: np.random.Generator
    config: RConfigurableGreedyPolicyConfig = field(
        default_factory=RConfigurableGreedyPolicyConfig
    )
    last_decision: RConfigurableGreedyDecision | None = field(
        init=False,
        default=None,
    )

    @property
    def name(self) -> str:
        """Return a stable name containing the configured greediness."""
        return f"configurable-greedy-{self.config.greediness:.3f}"

    @property
    def requires_full_analysis(self) -> bool:
        """Return true because selection uses exact all-edge rewards."""
        return True

    def select_action(
        self,
        environment: REnvironment,
    ) -> int:
        """Select an improving edge at the configured reward percentile.

        Args:
            environment (REnvironment): Active search environment.

        Returns:
            int: Selected host-edge index.

        Raises:
            RuntimeError: If no currently available edge has positive exact
                score reward. This is the natural exhaustion condition for
                the policy.
        """
        analysis = environment.analyze_actions()
        rewards = analysis.action_analysis.immediate_rewards

        positive_edges = np.flatnonzero(
            analysis.available_mask & (rewards > 0)
        ).astype(np.int32)

        if positive_edges.size == 0:
            self.last_decision = None
            raise RuntimeError(
                "Configurable greedy policy has no positive-reward action."
            )

        positive_rewards = rewards[positive_edges]
        sorted_rewards = np.sort(positive_rewards)

        reward_rank = int(
            round(
                self.config.greediness
                * (int(positive_edges.size) - 1)
            )
        )

        target_reward = int(sorted_rewards[reward_rank])
        candidate_edges = positive_edges[
            positive_rewards == target_reward
        ]
        edge = int(self.rng.choice(candidate_edges))

        self.last_decision = RConfigurableGreedyDecision(
            edge=edge,
            exact_reward=target_reward,
            positive_action_count=int(positive_edges.size),
            minimum_positive_reward=int(sorted_rewards[0]),
            maximum_positive_reward=int(sorted_rewards[-1]),
            reward_rank=reward_rank,
        )

        return edge
