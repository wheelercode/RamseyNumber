"""Test objective-aware K5 target and pattern selection."""

import numpy as np
import pytest

from ramsey.RColoring import RColoring
from ramsey.RGraph import RGraph
from ramsey.RK5Action import (
    apply_k5_pattern,
    monochromatic_k5_indices,
)
from ramsey.RK5Policy import (
    RK5PolicyConfig,
    rank_monochromatic_k5_targets,
    select_greedy_k5_action,
)
from ramsey.RObjective import danger_energy
from ramsey.RProblem import RProblem
from ramsey.RState import RSearchState


@pytest.fixture(scope="module")
def graph() -> RGraph:
    return RGraph(RProblem.r55(n_vertices=10))


def all_red_state(
    graph: RGraph,
) -> RSearchState:
    return RSearchState(
        RColoring(
            graph,
            np.zeros(
                graph.number_of_edges,
                dtype=np.uint8,
            ),
        )
    )


def test_target_ranking_contains_every_violation_once(
    graph: RGraph,
) -> None:
    state = all_red_state(graph)

    targets = monochromatic_k5_indices(state)

    ranking = rank_monochromatic_k5_targets(
        state,
        decay=0.25,
    )

    assert (
        ranking.number_of_targets
        == len(targets)
    )

    assert set(
        map(int, ranking.clique_indices)
    ) == set(
        map(int, targets)
    )

    assert np.all(
        ranking.leverage_scores[:-1]
        >= ranking.leverage_scores[1:]
    )


def test_greedy_k5_selection_respects_macro_distance_and_prediction(
    graph: RGraph,
) -> None:
    state = all_red_state(graph)

    selection = select_greedy_k5_action(
        state,
        RK5PolicyConfig(
            target_limit=5,
            minimum_changed_edges=3,
            maximum_changed_edges=4,
            use_danger_reward=True,
        ),
    )

    assert (
        3
        <= selection.changed_edge_count
        <= 4
    )

    before_score = state.score

    before_energy = danger_energy(
        state.histogram,
        decay=0.25,
    )

    actual_reward = apply_k5_pattern(
        state,
        selection.target_clique,
        selection.pattern_id,
    )

    after_energy = danger_energy(
        state.histogram,
        decay=0.25,
    )

    assert (
        actual_reward
        == selection.exact_reward
    )

    assert (
        state.score
        == selection.resulting_score
    )

    assert (
        before_score
        - state.score
        == selection.exact_reward
    )

    assert np.isclose(
        before_energy - after_energy,
        selection.objective_reward,
    )