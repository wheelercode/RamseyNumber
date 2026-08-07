"""Tests for exact counterfactual score/flexibility landscapes."""

import numpy as np
import pytest

from ramsey.RAction import analyze_actions
from ramsey.RActionFlexibility import calculate_action_flexibility_landscape
from ramsey.RColoring import RColoring
from ramsey.RFlexibility import calculate_flexibility
from ramsey.RGraph import RGraph
from ramsey.RProblem import RProblem
from ramsey.RState import RSearchState


def make_state(seed: int = 501) -> RSearchState:
    graph = RGraph(RProblem.r55(n_vertices=7))
    rng = np.random.default_rng(seed)
    colors = rng.integers(
        0,
        2,
        size=graph.number_of_edges,
        dtype=np.uint8,
    )
    return RSearchState(RColoring(graph, colors))


def test_counterfactual_flexibility_matches_real_flips() -> None:
    state = make_state()
    source_colors = state.colors.copy()
    source_version = state.version
    budgets = (0, 1, 2, 5)

    candidate_mask = np.zeros(state.number_of_edges, dtype=np.bool_)
    candidate_mask[[0, 4, 9]] = True

    landscape = calculate_action_flexibility_landscape(
        state,
        budgets=budgets,
        candidate_mask=candidate_mask,
    )

    for row, edge in enumerate(landscape.candidate_edges):
        actual = state.copy()
        actual.apply_edge_flip(int(edge))
        expected = calculate_flexibility(
            analyze_actions(actual).immediate_rewards,
            budgets,
        )

        assert np.allclose(
            landscape.resulting_fractions[row],
            expected.fractions,
        )

    assert state.version == source_version
    assert np.array_equal(state.colors, source_colors)


def test_flexibility_delta_is_result_minus_current() -> None:
    state = make_state(seed=502)
    mask = np.zeros(state.number_of_edges, dtype=np.bool_)
    mask[:5] = True

    landscape = calculate_action_flexibility_landscape(
        state,
        budgets=(0, 2),
        candidate_mask=mask,
    )

    assert np.allclose(
        landscape.flexibility_deltas,
        landscape.resulting_fractions - landscape.current.fractions,
    )


def test_pareto_actions_are_not_dominated() -> None:
    state = make_state(seed=503)
    landscape = calculate_action_flexibility_landscape(
        state,
        budgets=(0, 1, 3),
    )
    objectives = np.column_stack(
        (landscape.score_rewards, landscape.resulting_fractions)
    )

    for row in np.flatnonzero(landscape.pareto_mask):
        dominates = (
            np.all(objectives >= objectives[row], axis=1)
            & np.any(objectives > objectives[row], axis=1)
        )
        assert not np.any(dominates)


def test_stale_action_analysis_is_rejected() -> None:
    state = make_state(seed=504)
    analysis = analyze_actions(state)
    state.apply_edge_flip(0)

    with pytest.raises(ValueError):
        calculate_action_flexibility_landscape(
            state,
            analysis=analysis,
        )