"""Test objective-neutral K5-pattern analysis and application."""

import numpy as np
import pytest

from ramsey.RColoring import RColoring
from ramsey.RGraph import RGraph
from ramsey.RK5Action import (
    K5_ALLOWED_PATTERN_IDS,
    K5_PATTERN_COLORS,
    analyze_k5_patterns,
    apply_k5_pattern,
    monochromatic_k5_indices,
    pattern_colors,
)
from ramsey.RProblem import RProblem
from ramsey.RState import RSearchState
from ramsey.RVerification import verify_search_state


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


def test_k5_pattern_library_contains_exactly_1022_nonmonochromatic_patterns(
) -> None:
    assert K5_ALLOWED_PATTERN_IDS.shape == (1022,)

    assert K5_PATTERN_COLORS.shape == (1022, 10)

    blue_counts = K5_PATTERN_COLORS.sum(axis=1)

    assert np.all(blue_counts >= 1)

    assert np.all(blue_counts <= 9)

    assert np.array_equal(
        pattern_colors(1),
        K5_PATTERN_COLORS[0],
    )

    assert np.array_equal(
        pattern_colors(1022),
        K5_PATTERN_COLORS[-1],
    )


def test_monochromatic_target_count_matches_exact_score(
    graph: RGraph,
) -> None:
    state = all_red_state(graph)

    targets = monochromatic_k5_indices(state)

    assert len(targets) == state.score


def test_k5_pattern_prediction_matches_actual_recoloring(
    graph: RGraph,
) -> None:
    state = all_red_state(graph)

    target = int(monochromatic_k5_indices(state)[0])

    analysis = analyze_k5_patterns(
        state,
        target,
    )

    pattern_ids = (
        1,
        7,
        341,
        511,
        682,
        1022,
    )

    for pattern_id in pattern_ids:
        candidate = state.copy()

        before_score = candidate.score
        before_histogram = candidate.histogram.copy()

        actual_reward = apply_k5_pattern(
            candidate,
            target,
            pattern_id,
        )

        pattern_index = pattern_id - 1

        assert actual_reward == int(
            analysis.exact_rewards[pattern_index]
        )

        assert candidate.score == int(
            analysis.resulting_scores[pattern_index]
        )

        assert (
            before_score
            - candidate.score
            == actual_reward
        )

        assert np.array_equal(
            candidate.histogram - before_histogram,
            analysis.histogram_deltas[
                pattern_index
            ],
        )

        verify_search_state(
            candidate
        ).require_consistent()


def test_k5_recoloring_preserves_incremental_action_profiles(
    graph: RGraph,
) -> None:
    state = all_red_state(graph)

    # Force creation of the lazy all-edge
    # action-profile cache.
    state.action_profiles

    target = int(
        monochromatic_k5_indices(state)[3]
    )

    apply_k5_pattern(
        state,
        target,
        341,
    )

    expected_state = RSearchState(
        state.coloring_snapshot()
    )

    assert np.array_equal(
        state.action_profiles,
        expected_state.action_profiles,
    )


def test_pattern_analysis_rejects_nonmonochromatic_target(
    graph: RGraph,
) -> None:
    state = all_red_state(graph)

    target = int(
        monochromatic_k5_indices(state)[0]
    )

    apply_k5_pattern(
        state,
        target,
        341,
    )

    with pytest.raises(
        ValueError,
        match="monochromatic",
    ):
        analyze_k5_patterns(
            state,
            target,
        )