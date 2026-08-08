"""Interchangeable strategies for selecting the next edge action."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .REnvironment import REnvironment


class RPolicyExhausted(RuntimeError):
    """Signal that a policy has no action satisfying its own rules."""


def _choose_uniformly(
    actions: np.ndarray,
    rng: np.random.Generator,
) -> int:
    """
    Choose one encoded action uniformly from a nonempty array.
    """
    actions = np.asarray(
        actions,
        dtype=np.int32,
    )

    if actions.ndim != 1:
        raise ValueError("actions must be one-dimensional.")

    if actions.size == 0:
        raise RuntimeError("No candidate actions were provided.")

    return int(rng.choice(actions))


class RPolicy(ABC):
    """
    Select an action without mutating the environment.

    Search policies decide which available action to take. State
    mutation, action validation, termination, and history restrictions
    remain responsibilities of REnvironment.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return a stable human-readable policy name.
        """
        ...

    @property
    def requires_full_analysis(self) -> bool:
        """
        Return whether the policy requires complete action analysis.

        RSearch uses this value when applying the selected action.
        Policies that require complete analysis can reuse the cached
        analysis during the environment step.
        """
        return False

    @abstractmethod
    def select_action(
        self,
        environment: REnvironment,
    ) -> int:
        """
        Return one currently available encoded edge index.
        """
        ...


@dataclass(slots=True)
class RRandomPolicy(RPolicy):
    """
    Select uniformly from the currently available actions.
    """

    rng: np.random.Generator

    @property
    def name(self) -> str:
        return "random"

    def select_action(
        self,
        environment: REnvironment,
    ) -> int:
        return _choose_uniformly(
            environment.available_actions(),
            self.rng,
        )


@dataclass(slots=True)
class RGreedyPolicy(RPolicy):
    """
    Select an available action having the greatest current reward.

    The policy can rank actions using either:

    - the active objective's shaped reward; or
    - the exact reduction in monochromatic score.

    Ties are broken uniformly at random.
    """

    rng: np.random.Generator
    use_objective_reward: bool = True

    def __post_init__(self) -> None:
        if not isinstance(
            self.use_objective_reward,
            bool,
        ):
            raise TypeError("use_objective_reward must be boolean.")

    @property
    def name(self) -> str:
        if self.use_objective_reward:
            return "greedy-objective"

        return "greedy-exact-score"

    @property
    def requires_full_analysis(self) -> bool:
        return True

    def select_action(
        self,
        environment: REnvironment,
    ) -> int:
        analysis = environment.analyze_actions()

        if self.use_objective_reward:
            rewards = analysis.objective_rewards
        else:
            rewards = analysis.action_analysis.immediate_rewards

        masked_rewards = np.where(
            analysis.available_mask,
            rewards,
            -np.inf,
        )

        best_reward = np.max(masked_rewards)

        best_actions = np.flatnonzero(masked_rewards == best_reward).astype(np.int32)

        return _choose_uniformly(
            best_actions,
            self.rng,
        )