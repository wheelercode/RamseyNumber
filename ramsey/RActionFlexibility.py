"""Counterfactual flexibility produced by candidate edge flips.

Where :mod:`ramsey.RAction` computes the *immediate* exact score reward of
every edge flip, this module looks one flip ahead: for a set of candidate
edges it actually applies each flip to a scratch copy of the search state,
recomputes the action analysis, measures the resulting local flexibility
(the fraction of subsequent actions available within various damage
budgets, see :mod:`ramsey.RFlexibility`), and undoes the flip. The result
is an exact score/flexibility landscape over candidate actions, together
with the Pareto-optimal subset that jointly maximizes exact score reward
and every resulting flexibility budget without imposing an arbitrary
scalar weighting between them.
"""

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
    """Copy an array, mark the copy read-only, and return it.

    Args:
        array (NDArray): Source array; may be any array-like value.

    Returns:
        NDArray: An independently owned, read-only copy.
    """
    result = np.asarray(array).copy()
    result.flags.writeable = False
    return result


def _pareto_mask(objectives: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Return rows not dominated when every objective is maximized.

    A row is dominated when some other row is at least as good in
    every objective column and strictly better in at least one.

    Args:
        objectives (NDArray[np.float64]): Array of shape
            ``(number_of_candidates, number_of_objectives)`` where every
            column is an objective to maximize.

    Returns:
        NDArray[np.bool_]: Boolean mask of shape
        ``(number_of_candidates,)``, ``True`` for rows on the Pareto
        front (not dominated by any other row).
    """
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
    """Score/flexibility consequences for a selected set of edge flips.

    Attributes:
        source_state (RSearchState): Search state the landscape was
            computed from.
        state_version (int): ``source_state.version`` at computation
            time, used to detect staleness.
        current (RFlexibilityProfile): Flexibility profile of
            ``source_state`` itself, before any candidate flip.
        candidate_edges (NDArray[np.int32]): Host-edge indices that were
            evaluated, shape ``(number_of_candidates,)``.
        score_rewards (NDArray[np.int32]): Exact immediate score reward
            of flipping each candidate edge, shape
            ``(number_of_candidates,)``.
        resulting_fractions (NDArray[np.float64]): Flexibility fraction
            ``F(budget)`` after applying each candidate flip, shape
            ``(number_of_candidates, number_of_budgets)`` with columns
            aligned to ``current.budgets``.
        flexibility_deltas (NDArray[np.float64]): ``resulting_fractions``
            minus ``current.fractions``, i.e. the change in flexibility
            each candidate flip would produce, same shape as
            ``resulting_fractions``.
        pareto_mask (NDArray[np.bool_]): ``True`` for candidates that are
            Pareto-optimal when jointly maximizing ``score_rewards`` and
            every column of ``resulting_fractions``, shape
            ``(number_of_candidates,)``.
    """

    source_state: RSearchState
    state_version: int
    current: RFlexibilityProfile
    candidate_edges: NDArray[np.int32]
    score_rewards: NDArray[np.int32]
    resulting_fractions: NDArray[np.float64]
    flexibility_deltas: NDArray[np.float64]
    pareto_mask: NDArray[np.bool_]

    def __post_init__(self) -> None:
        """Validate array shapes and freeze all array fields.

        Raises:
            ValueError: If any of ``candidate_edges``, ``score_rewards``,
                ``pareto_mask``, ``resulting_fractions``, or
                ``flexibility_deltas`` does not match the shape implied
                by ``candidate_edges`` and ``current.budgets``.
        """
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
        """Return whether this landscape still describes ``state``.

        Args:
            state (RSearchState): Candidate state to check freshness
                against.

        Returns:
            bool: ``True`` if ``state`` is the exact object this
            landscape was computed from and has not been mutated since.
        """
        return self.source_state is state and self.state_version == state.version

    @property
    def pareto_edges(self) -> NDArray[np.int32]:
        """NDArray[np.int32]: Actions nondominated in score and the full F vector."""
        return self.candidate_edges[self.pareto_mask]

    def budget_index(self, budget: int) -> int:
        """Return the column index of one flexibility budget.

        Args:
            budget (int): Damage budget value to locate among
                ``current.budgets``.

        Returns:
            int: Column index of ``budget`` in ``current.budgets``,
            usable to index ``resulting_fractions`` and
            ``flexibility_deltas``.

        Raises:
            KeyError: If ``budget`` is not present in the landscape.
        """
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
    """Calculate the exact future flexibility produced by candidate flips.

    For every candidate edge, this actually applies the flip to a scratch
    copy of ``state``, recomputes the full action analysis, measures the
    resulting flexibility profile, and flips the edge back — so
    ``resulting_fractions`` reflects the exact one-move-ahead flexibility
    landscape rather than an approximation. The source state is never
    mutated. The scratch state reuses the incremental action-profile
    machinery (materialized once up front via ``scratch.action_profiles``),
    making the repeated flip/analyze/undo cycle much cheaper than rebuilding
    K5 incidence information from scratch for every candidate.

    Pareto dominance (see :func:`_pareto_mask`) maximizes exact score
    reward and every resulting ``F(b)`` component simultaneously. No
    arbitrary scalar weighting between score and flexibility is
    introduced.

    Args:
        state (RSearchState): Search state to evaluate candidate flips
            against; never mutated.
        analysis (RActionAnalysis | None): Action analysis for
            ``state``. If ``None``, it is computed via
            :func:`ramsey.RAction.analyze_actions`.
        budgets (Iterable[int]): Damage budgets defining the flexibility
            curve; forwarded to :func:`calculate_flexibility`.
        candidate_mask (NDArray[np.bool_] | None): Boolean mask of shape
            ``(state.number_of_edges,)`` selecting which edges to
            evaluate as candidates. If ``None``, every edge is a
            candidate.

    Returns:
        RActionFlexibilityLandscape: Exact score and flexibility
        consequences of every candidate flip, including the Pareto
        front over score reward and resulting flexibility.

    Raises:
        ValueError: If ``analysis`` does not apply to ``state``, if
            ``candidate_mask`` has the wrong shape, or if no candidate
            edges are selected.
        RuntimeError: If the scratch state fails to restore to
            ``state``'s original score and colors after the flip/undo
            cycle for every candidate (an internal consistency check).
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