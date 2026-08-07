"""Greedy policies for vertex color-balance transfer actions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .RObjective import (
    all_danger_rewards,
    danger_weights,
)
from .RState import RSearchState
from .RVertexBalanceAction import (
    RVertexBalanceTransfer,
    analyze_vertex_balance_transfers,
)


@dataclass(
    frozen=True,
    slots=True,
)
class RVertexBalancePolicyConfig:
    """Configure greedy selection among balance-improving transfers.

    Attributes:
        use_danger_reward (bool): If ``True``, rank candidate transfers
            by the decayed danger-energy reward computed from their
            histogram deltas (see
            :func:`~ramsey.RObjective.all_danger_rewards`). If
            ``False``, rank by each transfer's exact score reward.
            Defaults to ``True``.
        danger_decay (float): Decay factor passed to the danger-reward
            calculation; eagerly validated in :meth:`__post_init__`.
            Defaults to ``0.25``.
        prioritize_balance (bool): If ``True``, balance reduction is
            the primary objective and the selected reward (danger or
            exact) is used only as a tie-breaker among transfers that
            achieve the maximum balance reward. If ``False``, the
            selected reward chooses among all moves that still
            strictly improve vertex balance. Defaults to ``True``.
    """

    use_danger_reward: bool = True
    danger_decay: float = 0.25

    # True makes balance reduction the primary objective and Ramsey
    # reward the tie-breaker. False lets Ramsey reward choose among all
    # moves that still strictly improve vertex balance.
    prioritize_balance: bool = True

    def __post_init__(self) -> None:
        """Validate configuration types and eagerly validate ``danger_decay``.

        Raises:
            TypeError: If ``use_danger_reward`` or
                ``prioritize_balance`` is not a ``bool``.
            ValueError: If ``danger_decay`` is not a valid decay factor
                for :func:`~ramsey.RObjective.danger_weights`.
        """
        if not isinstance(
            self.use_danger_reward,
            bool,
        ):
            raise TypeError(
                "use_danger_reward must be boolean."
            )

        if not isinstance(
            self.prioritize_balance,
            bool,
        ):
            raise TypeError(
                "prioritize_balance must be boolean."
            )

        danger_weights(
            2,
            decay=self.danger_decay,
        )


@dataclass(
    frozen=True,
    slots=True,
)
class RVertexBalanceSelection:
    """One balance transfer selected by the greedy policy.

    Attributes:
        transfer (RVertexBalanceTransfer): The selected transfer.
        balance_reward (int): Exact reduction in vertex color-balance
            energy the transfer produces.
        exact_reward (int): Exact score reduction the transfer
            produces.
        objective_reward (float): Reward used to rank the transfer
            among its eligible competitors -- either the danger-energy
            reward or the exact score reward, depending on
            configuration.
        resulting_score (int): Exact score that would result from
            applying the transfer.
    """

    transfer: RVertexBalanceTransfer
    balance_reward: int
    exact_reward: int
    objective_reward: float
    resulting_score: int


def select_greedy_vertex_balance_action(
    state: RSearchState,
    config: RVertexBalancePolicyConfig | None = None,
) -> RVertexBalanceSelection:
    """Select one balance-improving pivot transfer without mutation.

    Analyzes every candidate transfer for ``state``, computes each
    transfer's objective reward (danger-energy or exact score
    reduction, per ``config.use_danger_reward``), optionally restricts
    eligibility to transfers achieving the maximum balance reward (per
    ``config.prioritize_balance``), and returns the highest-reward
    eligible transfer. The transfer is only selected, not applied; use
    :func:`~ramsey.RVertexBalanceAction.apply_vertex_balance_transfer`
    to mutate ``state``.

    Args:
        state (RSearchState): Search state to analyze.
        config (RVertexBalancePolicyConfig | None): Selection
            configuration. Defaults to a freshly constructed
            ``RVertexBalancePolicyConfig()`` when ``None``.

    Returns:
        RVertexBalanceSelection: The selected transfer together with
        its balance, exact, and objective rewards, and its resulting
        score.

    Raises:
        RuntimeError: If ``state`` currently has no candidate
            vertex-balance transfers.
    """
    config = (
        config
        if config is not None
        else RVertexBalancePolicyConfig()
    )

    analysis = analyze_vertex_balance_transfers(
        state,
    )

    if analysis.number_of_actions == 0:
        raise RuntimeError(
            "There are no vertex-balance transfers available."
        )

    if config.use_danger_reward:
        objective_rewards = all_danger_rewards(
            analysis.histogram_deltas,
            decay=config.danger_decay,
        )
    else:
        objective_rewards = analysis.exact_rewards.astype(
            np.float64,
        )

    eligible = np.ones(
        analysis.number_of_actions,
        dtype=np.bool_,
    )

    if config.prioritize_balance:
        maximum_balance_reward = int(
            np.max(analysis.balance_rewards)
        )

        eligible &= (
            analysis.balance_rewards
            == maximum_balance_reward
        )

    eligible_rewards = np.where(
        eligible,
        objective_rewards,
        -np.inf,
    )

    action_index = int(
        np.argmax(eligible_rewards)
    )

    return RVertexBalanceSelection(
        transfer=analysis.transfer(
            action_index
        ),
        balance_reward=int(
            analysis.balance_rewards[
                action_index
            ]
        ),
        exact_reward=int(
            analysis.exact_rewards[
                action_index
            ]
        ),
        objective_reward=float(
            objective_rewards[action_index]
        ),
        resulting_score=int(
            analysis.resulting_scores[
                action_index
            ]
        ),
    )