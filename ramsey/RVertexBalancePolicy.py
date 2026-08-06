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
    """Configure greedy selection among balance-improving transfers."""

    use_danger_reward: bool = True
    danger_decay: float = 0.25

    # True makes balance reduction the primary objective and Ramsey
    # reward the tie-breaker. False lets Ramsey reward choose among all
    # moves that still strictly improve vertex balance.
    prioritize_balance: bool = True

    def __post_init__(self) -> None:
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
    """One balance transfer selected by the greedy policy."""

    transfer: RVertexBalanceTransfer
    balance_reward: int
    exact_reward: int
    objective_reward: float
    resulting_score: int


def select_greedy_vertex_balance_action(
    state: RSearchState,
    config: RVertexBalancePolicyConfig | None = None,
) -> RVertexBalanceSelection:
    """Select one balance-improving pivot transfer without mutation."""
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