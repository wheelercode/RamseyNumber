"""Objective-aware policies for selecting K5 recoloring actions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .RK5Action import (
    K5_EDGE_COUNT,
    analyze_k5_patterns,
    monochromatic_k5_indices,
)
from .RObjective import (
    all_danger_rewards,
    danger_weights,
)
from .RState import RSearchState


def _read_only_copy(
    values: NDArray,
) -> NDArray:
    result = np.asarray(values).copy()

    result.flags.writeable = False

    return result


@dataclass(
    frozen=True,
    slots=True,
    eq=False,
)
class RK5TargetRanking:
    """Monochromatic K5 targets ranked by objective-aware leverage."""

    clique_indices: NDArray[np.uint32]
    leverage_scores: NDArray[np.float64]

    def __post_init__(self) -> None:
        clique_indices = _read_only_copy(
            np.asarray(
                self.clique_indices,
                dtype=np.uint32,
            )
        )

        leverage_scores = _read_only_copy(
            np.asarray(
                self.leverage_scores,
                dtype=np.float64,
            )
        )

        if clique_indices.shape != leverage_scores.shape:
            raise ValueError(
                "clique_indices and leverage_scores must have equal shape."
            )

        if clique_indices.ndim != 1:
            raise ValueError(
                "target ranking arrays must be one-dimensional."
            )

        object.__setattr__(
            self,
            "clique_indices",
            clique_indices,
        )

        object.__setattr__(
            self,
            "leverage_scores",
            leverage_scores,
        )

    @property
    def number_of_targets(self) -> int:
        return len(self.clique_indices)

    @property
    def best_clique(self) -> int:
        if self.number_of_targets == 0:
            raise RuntimeError(
                "There are no monochromatic K5 targets."
            )

        return int(self.clique_indices[0])


@dataclass(
    frozen=True,
    slots=True,
)
class RK5PolicyConfig:
    """Configure greedy selection among K5-pattern macro actions."""

    # Only the highest-leverage bad K5s receive the more expensive
    # exact 1,022-pattern analysis.
    target_limit: int = 10

    # A minimum greater than one prevents the K5 macro action from
    # collapsing back into the ordinary single-edge action space.
    minimum_changed_edges: int = 2
    maximum_changed_edges: int = 9

    # True ranks replacement patterns by graded danger energy.
    # False ranks them by exact monochromatic-K5 reduction.
    use_danger_reward: bool = True
    danger_decay: float = 0.25

    def __post_init__(self) -> None:
        if isinstance(self.target_limit, bool) or not isinstance(
            self.target_limit,
            (int, np.integer),
        ):
            raise TypeError(
                "target_limit must be an integer."
            )

        if self.target_limit <= 0:
            raise ValueError(
                "target_limit must be positive."
            )

        for name, value in (
            (
                "minimum_changed_edges",
                self.minimum_changed_edges,
            ),
            (
                "maximum_changed_edges",
                self.maximum_changed_edges,
            ),
        ):
            if isinstance(value, bool) or not isinstance(
                value,
                (int, np.integer),
            ):
                raise TypeError(
                    f"{name} must be an integer."
                )

        if self.minimum_changed_edges < 1:
            raise ValueError(
                "minimum_changed_edges must be at least one."
            )

        if self.maximum_changed_edges > K5_EDGE_COUNT - 1:
            raise ValueError(
                "maximum_changed_edges cannot exceed nine."
            )

        if (
            self.minimum_changed_edges
            > self.maximum_changed_edges
        ):
            raise ValueError(
                "minimum_changed_edges cannot exceed "
                "maximum_changed_edges."
            )

        if not isinstance(
            self.use_danger_reward,
            bool,
        ):
            raise TypeError(
                "use_danger_reward must be boolean."
            )

        # Use the objective layer's canonical validation for decay.
        danger_weights(
            K5_EDGE_COUNT + 1,
            decay=self.danger_decay,
        )


@dataclass(
    frozen=True,
    slots=True,
)
class RActionSelectionK5:
    """The K5 target/pattern pair selected by one greedy policy call."""

    target_clique: int
    pattern_id: int
    target_leverage: float
    exact_reward: int
    objective_reward: float
    resulting_score: int
    changed_edge_count: int


def rank_monochromatic_k5_targets(
    state: RSearchState,
    *,
    decay: float = 0.25,
) -> RK5TargetRanking:
    """
    Rank bad K5s by weighted dangerous-clique edge incidence.

    The structural action layer identifies the monochromatic K5s.
    This policy layer deliberately supplies the danger weights used to
    prioritize them.
    """
    targets = monochromatic_k5_indices(state)

    if len(targets) == 0:
        return RK5TargetRanking(
            clique_indices=targets,
            leverage_scores=np.empty(
                0,
                dtype=np.float64,
            ),
        )

    weights = danger_weights(
        state.edges_per_clique + 1,
        decay=decay,
    )

    edge_danger = state.action_profiles @ weights

    target_edges = state.index.clique_edges[targets]

    leverage = edge_danger[target_edges].sum(
        axis=1,
        dtype=np.float64,
    )

    order = np.argsort(
        -leverage,
        kind="stable",
    )

    return RK5TargetRanking(
        clique_indices=targets[order],
        leverage_scores=leverage[order],
    )


def select_greedy_k5_action(
    state: RSearchState,
    config: RK5PolicyConfig | None = None,
) -> RActionSelectionK5:
    """
    Select one K5-pattern macro action without mutating the state.

    Target ranking and objective evaluation live here rather than in
    RAction so the action representation remains objective-neutral.
    """
    config = (
        config
        if config is not None
        else RK5PolicyConfig()
    )

    ranking = rank_monochromatic_k5_targets(
        state,
        decay=config.danger_decay,
    )

    if ranking.number_of_targets == 0:
        raise RuntimeError(
            "There are no monochromatic K5 targets."
        )

    best_selection: RActionSelectionK5 | None = None
    best_objective_reward = -np.inf

    for target_offset, target_clique in enumerate(
        ranking.clique_indices[: config.target_limit]
    ):
        analysis = analyze_k5_patterns(
            state,
            int(target_clique),
        )

        allowed = (
            (
                analysis.changed_edge_counts
                >= config.minimum_changed_edges
            )
            & (
                analysis.changed_edge_counts
                <= config.maximum_changed_edges
            )
        )

        if config.use_danger_reward:
            rewards = all_danger_rewards(
                analysis.histogram_deltas,
                decay=config.danger_decay,
            )
        else:
            rewards = analysis.exact_rewards.astype(
                np.float64,
            )

        eligible_rewards = np.where(
            allowed,
            rewards,
            -np.inf,
        )

        pattern_offset = int(
            np.argmax(eligible_rewards)
        )

        objective_reward = float(
            eligible_rewards[pattern_offset]
        )

        if not np.isfinite(objective_reward):
            raise RuntimeError(
                "No K5 replacement pattern is eligible."
            )

        if objective_reward > best_objective_reward:
            best_objective_reward = objective_reward

            best_selection = RActionSelectionK5(
                target_clique=int(target_clique),
                pattern_id=int(
                    analysis.pattern_ids[pattern_offset]
                ),
                target_leverage=float(
                    ranking.leverage_scores[target_offset]
                ),
                exact_reward=int(
                    analysis.exact_rewards[pattern_offset]
                ),
                objective_reward=objective_reward,
                resulting_score=int(
                    analysis.resulting_scores[pattern_offset]
                ),
                changed_edge_count=int(
                    analysis.changed_edge_counts[pattern_offset]
                ),
            )

    if best_selection is None:
        raise RuntimeError(
            "No K5 macro action could be selected."
        )

    return best_selection