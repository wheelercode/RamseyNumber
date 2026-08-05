"""Test exact and incrementally maintained edge-action analysis."""

import numpy as np
from numpy.typing import NDArray
import pytest

from ramsey.RAction import (
    analyze_actions,
    edge_clique_profiles,
)
from ramsey.RColoring import RColoring
from ramsey.RGraph import RGraph
from ramsey.RProblem import RProblem
from ramsey.RState import RSearchState


@pytest.fixture(scope="module")
def graph() -> RGraph:
    return RGraph(RProblem.r55(n_vertices=10))


def make_state(
    graph: RGraph,
    seed: int,
) -> RSearchState:
    colors = np.random.default_rng(seed).integers(
        0,
        2,
        size=graph.number_of_edges,
        dtype=np.uint8,
    )

    return RSearchState(
        RColoring(
            graph,
            colors,
        )
    )


def independently_recompute_profiles(
    state: RSearchState,
) -> NDArray[np.uint16]:
    """Rebuild profiles without using the state's profile cache."""
    affected_counts = state.color_one_counts[
        state.index.edge_to_cliques
    ]

    number_of_bins = state.edges_per_clique + 1

    profiles = np.empty(
        (
            state.number_of_edges,
            number_of_bins,
        ),
        dtype=np.uint16,
    )

    for count in range(number_of_bins):
        profiles[:, count] = np.count_nonzero(
            affected_counts == count,
            axis=1,
        )

    return profiles


def test_action_profiles_match_independent_reconstruction(
    graph: RGraph,
) -> None:
    state = make_state(
        graph,
        seed=211,
    )

    expected = independently_recompute_profiles(state)

    assert np.array_equal(
        edge_clique_profiles(state),
        expected,
    )


def test_action_profiles_remain_exact_across_many_flips(
    graph: RGraph,
) -> None:
    state = make_state(
        graph,
        seed=212,
    )

    # Force construction of the lazy cache before mutations begin.
    edge_clique_profiles(state)

    rng = np.random.default_rng(213)

    edges = rng.integers(
        0,
        graph.number_of_edges,
        size=200,
    )

    for edge in edges:
        state.apply_edge_flip(int(edge))

        expected = independently_recompute_profiles(state)

        assert np.array_equal(
            edge_clique_profiles(state),
            expected,
        )


def test_action_analysis_uses_current_incremental_profiles(
    graph: RGraph,
) -> None:
    state = make_state(
        graph,
        seed=214,
    )

    analyze_actions(state)

    for edge in (0, 7, 19, 7, 31):
        state.apply_edge_flip(edge)

    analysis = analyze_actions(state)

    assert np.array_equal(
        analysis.profiles,
        independently_recompute_profiles(state),
    )


def test_state_action_profiles_are_read_only(
    graph: RGraph,
) -> None:
    state = make_state(
        graph,
        seed=215,
    )

    assert not state.action_profiles.flags.writeable