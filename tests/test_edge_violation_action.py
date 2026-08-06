"""Test edge participation analysis for monochromatic violations."""

import numpy as np

from ramsey.RColoring import RColoring
from ramsey.REdgeViolationAction import (
    analyze_edge_violations,
    edge_violation_loads,
)
from ramsey.RGraph import RGraph
from ramsey.RProblem import RProblem
from ramsey.RState import RSearchState


def make_state(
    seed: int = 401,
) -> RSearchState:
    graph = RGraph(
        RProblem.r55(n_vertices=10)
    )

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


def test_edge_violation_loads_match_direct_clique_counts() -> None:
    state = make_state()

    monochromatic = (
        (state.color_one_counts == 0)
        | (
            state.color_one_counts
            == state.edges_per_clique
        )
    )

    expected = np.count_nonzero(
        monochromatic[
            state.index.edge_to_cliques
        ],
        axis=1,
    ).astype(np.int32)

    assert np.array_equal(
        edge_violation_loads(state),
        expected,
    )


def test_total_edge_load_is_edges_per_clique_times_score() -> None:
    state = make_state(seed=402)

    analysis = analyze_edge_violations(state)

    assert analysis.total_load == (
        state.edges_per_clique
        * state.score
    )


def test_edge_violation_analysis_tracks_state_version() -> None:
    state = make_state(seed=403)

    analysis = analyze_edge_violations(state)

    assert analysis.applies_to(state)
    assert not analysis.violation_loads.flags.writeable

    state.apply_edge_flip(0)

    assert not analysis.applies_to(state)


def test_maximum_load_matches_load_array() -> None:
    state = make_state(seed=404)

    analysis = analyze_edge_violations(state)

    assert analysis.maximum_load == int(
        analysis.violation_loads.max()
    )