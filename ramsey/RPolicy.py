"""Interchangeable strategies for selecting the next edge action."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .REnvironment import REnvironment


def _choose_uniformly(
    actions: np.ndarray,
    rng: np.random.Generator,
) -> int:
    """
    Choose one encoded action uniformly from a nonempty array.

    Args:
        actions (np.ndarray): One-dimensional array of encoded action
            (edge) indices to choose among.
        rng (np.random.Generator): Random generator used to make the
            selection.

    Returns:
        int: One action index drawn uniformly at random from
        ``actions``.

    Raises:
        ValueError: If ``actions`` is not one-dimensional.
        RuntimeError: If ``actions`` is empty.
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
    remain responsibilities of ``REnvironment``.

    Concrete policies must implement :attr:`name` and
    :meth:`select_action`, and may override
    :attr:`requires_full_analysis` when they consume the environment's
    cached action analysis to make their selection.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        str: Stable human-readable policy name.
        """
        ...

    @property
    def requires_full_analysis(self) -> bool:
        """
        Return whether the policy requires complete action analysis.

        ``RSearch`` uses this value when applying the selected action.
        Policies that require complete analysis can reuse the cached
        analysis during the environment step.

        Returns:
            bool: ``True`` if the policy calls
            ``environment.analyze_actions()`` inside
            :meth:`select_action`; ``False`` by default.
        """
        return False

    @abstractmethod
    def select_action(
        self,
        environment: REnvironment,
    ) -> int:
        """
        Return one currently available encoded edge index.

        Args:
            environment (REnvironment): Environment to query for
                available actions and, if needed, their analysis.

        Returns:
            int: Encoded edge index of the action chosen from the
            environment's currently available actions.
        """
        ...


@dataclass(slots=True)
class RRandomPolicy(RPolicy):
    """
    Select uniformly from the currently available actions.

    Attributes:
        rng (np.random.Generator): Random generator used to make the
            uniform selection.
    """

    rng: np.random.Generator

    @property
    def name(self) -> str:
        """
        str: The literal name ``"random"``.
        """
        return "random"

    def select_action(
        self,
        environment: REnvironment,
    ) -> int:
        """
        Return one action chosen uniformly from the available actions.

        Args:
            environment (REnvironment): Environment queried for its
                currently available actions.

        Returns:
            int: Encoded edge index chosen uniformly at random.
        """
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

    Attributes:
        rng (np.random.Generator): Random generator used to break
            ties.
        use_objective_reward (bool): If ``True``, rank actions by the
            environment's shaped objective reward. If ``False``, rank
            actions by the exact score reduction from the environment's
            full action analysis. Defaults to ``True``.
    """

    rng: np.random.Generator
    use_objective_reward: bool = True

    def __post_init__(self) -> None:
        """
        Validate that ``use_objective_reward`` is a boolean.

        Raises:
            TypeError: If ``use_objective_reward`` is not a ``bool``.
        """
        if not isinstance(
            self.use_objective_reward,
            bool,
        ):
            raise TypeError("use_objective_reward must be boolean.")

    @property
    def name(self) -> str:
        """
        str: ``"greedy-objective"`` or ``"greedy-exact-score"``,
        depending on ``use_objective_reward``.
        """
        if self.use_objective_reward:
            return "greedy-objective"

        return "greedy-exact-score"

    @property
    def requires_full_analysis(self) -> bool:
        """
        Return ``True``: greedy selection always needs complete
        action analysis to rank the available actions.

        Returns:
            bool: Always ``True``.
        """
        return True

    def select_action(
        self,
        environment: REnvironment,
    ) -> int:
        """
        Return the available action with the greatest current reward.

        Analyzes every action in ``environment``, masks out actions
        that are not currently available, and selects the
        maximum-reward action among those remaining. When
        ``use_objective_reward`` is ``True`` the ranking uses the
        environment's shaped objective reward; otherwise it uses the
        exact score reduction from the full action analysis. Ties for
        the maximum reward are broken uniformly at random using
        ``rng``.

        Args:
            environment (REnvironment): Environment to analyze and
                select an action from.

        Returns:
            int: Encoded edge index of the selected best-reward
            action.
        """
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
