"""Test exact pairwise interaction measurements for edge flips."""

import numpy as np
import pytest

from ramsey.RColoring import RColoring
from ramsey.REdgeInteraction import calculate_edge_pair_interactions
from ramsey.RGraph import RGraph
from ramsey.RProblem import RProblem
from ramsey.RState import RSearchState


@pytest.fixture(scope="module")
def graph() -> RGraph:
    """Return K6 with K5 forbidden in both colors."""
    return RGraph(RProblem.r55(n_vertices=6))


def all_blue_state(graph: RGraph) -> RSearchState:
    """Return the maximally violating all-blue K6 state."""
    return RSearchState(
        RColoring(
            graph,
            np.ones(graph.number_of_edges, dtype=np.uint8),
        )
    )


def edge_index(graph: RGraph, i: int, j: int) -> int:
    """Return the encoded edge index for an unordered vertex pair."""
    target = np.asarray(sorted((i, j)), dtype=np.uint8)
    rows = np.flatnonzero(np.all(graph.edges == target, axis=1))
    return int(rows[0])


def test_all_blue_interaction_matches_common_k5_overlap(
    graph: RGraph,
) -> None:
    """Recover the exact adjacent/disjoint all-blue pair effects."""
    e01 = edge_index(graph, 0, 1)
    e02 = edge_index(graph, 0, 2)
    e23 = edge_index(graph, 2, 3)

    result = calculate_edge_pair_interactions(
        all_blue_state(graph),
        np.asarray(
            [
                [e01, e02],
                [e01, e23],
            ],
            dtype=np.int32,
        ),
    )

    # In K6, two adjacent edges occur together in C(3, 2)=3 K5s;
    # disjoint edges occur together in C(2, 1)=2 K5s. The first red
    # edge removes exactly those K5s from the second edge's reward.
    assert result.reward_interactions.tolist() == [-3, -2]
    assert result.common_cliques.tolist() == [3, 2]
    assert result.shared_vertices.tolist() == [1, 0]
    assert result.score_interactions.tolist() == [3, 2]


def test_pair_interaction_is_symmetric(graph: RGraph) -> None:
    """Verify the mixed score effect is independent of pair order."""
    rng = np.random.default_rng(17)
    state = RSearchState(
        RColoring(
            graph,
            rng.integers(
                0,
                2,
                size=graph.number_of_edges,
                dtype=np.uint8,
            ),
        )
    )

    pairs = np.asarray([[0, 7], [7, 0]], dtype=np.int32)
    result = calculate_edge_pair_interactions(state, pairs)

    assert result.reward_interactions[0] == result.reward_interactions[1]


def test_source_state_is_not_mutated(graph: RGraph) -> None:
    """Keep interaction measurement observational and memoryless."""
    state = all_blue_state(graph)
    colors_before = state.colors.copy()
    histogram_before = state.histogram.copy()
    score_before = state.score
    version_before = state.version

    calculate_edge_pair_interactions(
        state,
        np.asarray([[0, 1], [2, 3]], dtype=np.int32),
    )

    assert np.array_equal(state.colors, colors_before)
    assert np.array_equal(state.histogram, histogram_before)
    assert state.score == score_before
    assert state.version == version_before


def test_rejects_self_pair(graph: RGraph) -> None:
    """Reject an edge paired with itself."""
    with pytest.raises(ValueError):
        calculate_edge_pair_interactions(
            all_blue_state(graph),
            np.asarray([[0, 0]], dtype=np.int32),
        )