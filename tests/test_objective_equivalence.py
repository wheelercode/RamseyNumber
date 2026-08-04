"""Compare objective calculations with the untouched danger functions."""

import numpy as np
import pytest

import RamseyDanger as reference_danger
import RamseyGraph as reference_graph

from ramsey.RAction import analyze_actions
from ramsey.RColoring import RColoring
from ramsey.RGraph import RGraph
from ramsey.RObjective import (
    RDangerObjective,
    RMonochromaticObjective,
    all_danger_rewards,
    danger_energy,
    danger_weights,
    minority_histogram,
)
from ramsey.RProblem import RProblem
from ramsey.RState import RSearchState


@pytest.fixture(scope="module")
def state() -> RSearchState:
    """
    Return one deterministic state for objective tests.
    """
    graph = RGraph(RProblem.r55(n_vertices=10))

    colors = reference_graph.random_coloring(
        graph.number_of_edges,
        np.random.default_rng(51),
    )

    return RSearchState(
        RColoring(
            graph,
            colors,
        )
    )


def test_minority_histogram_matches_reference(
    state: RSearchState,
) -> None:
    expected = reference_danger.minority_histogram(state.histogram)

    actual = minority_histogram(state.histogram)

    assert np.array_equal(
        actual,
        expected,
    )


@pytest.mark.parametrize(
    "decay",
    [
        0.0,
        0.25,
        0.5,
        1.0,
    ],
)
def test_danger_energy_matches_reference(
    state: RSearchState,
    decay: float,
) -> None:
    expected = reference_danger.danger_energy(
        state.histogram,
        decay=decay,
    )

    actual = danger_energy(
        state.histogram,
        decay=decay,
    )

    assert np.isclose(
        actual,
        expected,
    )


@pytest.mark.parametrize(
    "decay",
    [
        0.0,
        0.25,
        0.5,
        1.0,
    ],
)
def test_all_danger_rewards_match_reference(
    state: RSearchState,
    decay: float,
) -> None:
    analysis = analyze_actions(state)

    expected = reference_danger.all_danger_rewards(
        analysis.histogram_deltas,
        decay=decay,
    )

    actual = all_danger_rewards(
        analysis.histogram_deltas,
        decay=decay,
    )

    assert np.allclose(
        actual,
        expected,
    )


def test_monochromatic_objective_uses_exact_rewards(
    state: RSearchState,
) -> None:
    objective = RMonochromaticObjective()

    analysis = analyze_actions(state)

    version = state.version

    rewards = objective.action_rewards(
        state,
        analysis,
    )

    assert objective.name == "monochromatic"

    assert objective.energy(state) == state.score

    assert np.array_equal(
        rewards,
        analysis.immediate_rewards,
    )

    assert state.version == version


def test_danger_objective_matches_danger_functions(
    state: RSearchState,
) -> None:
    objective = RDangerObjective(decay=0.25)

    analysis = analyze_actions(state)

    assert objective.name == "danger"

    assert objective.energy(state) == danger_energy(
        state.histogram,
        decay=0.25,
    )

    assert np.allclose(
        objective.action_rewards(
            state,
            analysis,
        ),
        all_danger_rewards(
            analysis.histogram_deltas,
            decay=0.25,
        ),
    )


def test_objective_rejects_stale_analysis(
    state: RSearchState,
) -> None:
    independent_state = state.copy()

    analysis = analyze_actions(independent_state)

    independent_state.apply_edge_flip(0)

    with pytest.raises(
        ValueError,
        match="current state",
    ):
        RDangerObjective().action_rewards(
            independent_state,
            analysis,
        )


def test_danger_weights_are_symmetric() -> None:
    weights = danger_weights(
        11,
        decay=0.25,
    )

    assert np.array_equal(
        weights,
        weights[::-1],
    )

    assert weights[0] == 1.0
    assert weights[-1] == 1.0

    assert weights[5] == 0.25**5


def test_danger_objective_rejects_invalid_decay() -> None:
    with pytest.raises(
        ValueError,
        match="between zero and one",
    ):
        RDangerObjective(decay=1.1)
