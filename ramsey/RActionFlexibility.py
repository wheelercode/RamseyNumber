"""Counterfactual flexibility produced by candidate edge flips."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from .RAction import RActionAnalysis, analyze_actions
from .RFlexibility import (
    DEFAULT_FLEXIBILITY_BUDGETS,
    RFlexibilityProfile,
    calculate_flexibility,
)
from .RState import RSearchState


def _read_only_copy(array: NDArray) -> NDArray:
    result = np.asarray(array).copy()
    result.flags.writeable = False
    return result


def _pareto_mask(objectives: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Return rows not dominated when every objective is maximized."""
    number_of_rows = len(objectives)
    result = np.ones(number_of_rows, dtype=np.bool_)

    for row in range(number_of_rows):
        at_least_as_good = np.all(
            objectives >= objectives[row],
            axis=1,
        )
        strictly_better = np.any(
            objectives > objectives[row],
            axis=1,
        )

        if np.any(at_least_as_good & strictly_better):
            result[row] = False

    return result


@dataclass(frozen=True, slots=True, eq=False)
class RActionFlexibilityLandscape:
    """Score/flexibility consequences for a selected set of edge flips."""

    source_state: RSearchState
    state_version: int
    current: RFlexibilityProfile
    candidate_edges: NDArray[np.int32]
    score_rewards: NDArray[np.int32]
    resulting_fractions: NDArray[np.float64]
    flexibility_deltas: NDArray[np.float64]
    pareto_mask: NDArray[np.bool_]

    def __post_init__(self) -> None:
        number_of_candidates = len(self.candidate_edges)
        number_of_budgets = len(self.current.budgets)

        vector_shape = (number_of_candidates,)
        matrix_shape = (number_of_candidates, number_of_budgets)

        if self.candidate_edges.shape != vector_shape:
            raise ValueError("candidate_edges has the wrong shape.")

        if self.score_rewards.shape != vector_shape:
            raise ValueError("score_rewards has the wrong shape.")

        if self.pareto_mask.shape != vector_shape:
            raise ValueError("pareto_mask has the wrong shape.")

        if self.resulting_fractions.shape != matrix_shape:
            raise ValueError("resulting_fractions has the wrong shape.")

        if self.flexibility_deltas.shape != matrix_shape:
            raise ValueError("flexibility_deltas has the wrong shape.")

        for name in (
            "candidate_edges",
            "score_rewards",
            "resulting_fractions",
            "flexibility_deltas",
            "pareto_mask",
        ):
            object.__setattr__(
                self,
                name,
                _read_only_copy(getattr(self, name)),
            )

    def applies_to(self, state: RSearchState) -> bool:
        return self.source_state is state and self.state_version == state.version

    @property
    def pareto_edges(self) -> NDArray[np.int32]:
        """Return actions nondominated in score and the full F vector."""
        return self.candidate_edges[self.pareto_mask]

    def budget_index(self, budget: int) -> int:
        indexes = np.flatnonzero(self.current.budgets == budget)

        if indexes.size == 0:
            raise KeyError(f"budget {budget} is not present in the landscape.")

        return int(indexes[0])


def calculate_action_flexibility_landscape(
    state: RSearchState,
    analysis: RActionAnalysis | None = None,
    budgets: Iterable[int] = DEFAULT_FLEXIBILITY_BUDGETS,
    *,
    candidate_mask: NDArray[np.bool_] | None = None,
) -> RActionFlexibilityLandscape:
    """
    Calculate the exact future flexibility produced by candidate flips.

    The source state is never mutated. A private scratch state reuses the
    incremental action-profile machinery, making exact counterfactuals much
    cheaper than rebuilding K5 incidence information for every candidate.

    Pareto dominance maximizes exact score reward and every resulting F(b)
    component simultaneously. No arbitrary scalar weighting is introduced.
    """
    if analysis is None:
        analysis = analyze_actions(state)
    elif not analysis.applies_to(state):
        raise ValueError("Action analysis does not describe the supplied state.")

    current = calculate_flexibility(
        analysis.immediate_rewards,
        budgets,
    )

    if candidate_mask is None:
        candidate_edges = np.arange(
            state.number_of_edges,
            dtype=np.int32,
        )
    else:
        candidate_mask = np.asarray(candidate_mask, dtype=np.bool_)

        if candidate_mask.shape != (state.number_of_edges,):
            raise ValueError("candidate_mask has the wrong shape.")

        candidate_edges = np.flatnonzero(candidate_mask).astype(np.int32)

    if candidate_edges.size == 0:
        raise ValueError("at least one candidate action is required.")

    score_rewards = analysis.immediate_rewards[candidate_edges].astype(
        np.int32,
        copy=True,
    )

    resulting_fractions = np.empty(
        (len(candidate_edges), len(current.budgets)),
        dtype=np.float64,
    )

    scratch = state.copy()

    # Materialize this cache once. Each flip and undo then updates it
    # incrementally rather than rebuilding all edge/K5 profiles.
    _ = scratch.action_profiles

    for row, edge in enumerate(candidate_edges):
        edge = int(edge)
        scratch.apply_edge_flip(edge)

        future_analysis = analyze_actions(scratch)
        future = calculate_flexibility(
            future_analysis.immediate_rewards,
            current.budgets,
        )
        resulting_fractions[row] = future.fractions

        scratch.apply_edge_flip(edge)

    if scratch.score != state.score or not np.array_equal(
        scratch.colors,
        state.colors,
    ):
        raise RuntimeError("Counterfactual scratch state failed to restore.")

    flexibility_deltas = (
        resulting_fractions
        - current.fractions[np.newaxis, :]
    )

    objectives = np.column_stack(
        (
            score_rewards.astype(np.float64),
            resulting_fractions,
        )
    )

    return RActionFlexibilityLandscape(
        source_state=state,
        state_version=state.version,
        current=current,
        candidate_edges=candidate_edges,
        score_rewards=score_rewards,
        resulting_fractions=resulting_fractions,
        flexibility_deltas=flexibility_deltas,
        pareto_mask=_pareto_mask(objectives),
    )