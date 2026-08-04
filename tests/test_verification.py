"""Test independent coloring and incremental-state verification."""

import numpy as np
import pytest

import RamseyGraph as reference_graph

from ramsey.RColoring import RColoring
from ramsey.RGraph import RGraph
from ramsey.RProblem import RProblem
from ramsey.RScoring import evaluate_coloring
from ramsey.RState import RSearchState
from ramsey.RVerification import (
    verify_coloring,
    verify_search_state,
)


@pytest.fixture(scope="module")
def graph() -> RGraph:
    """
    Return a small symmetric graph for verification tests.
    """
    return RGraph(RProblem.r55(n_vertices=10))


def make_state(
    graph: RGraph,
    seed: int,
) -> RSearchState:
    """
    Return one deterministic incremental state.
    """
    colors = reference_graph.random_coloring(
        graph.number_of_edges,
        np.random.default_rng(seed),
    )

    return RSearchState(
        RColoring(
            graph,
            colors,
        )
    )


def test_coloring_verification_matches_exact_scoring(
    graph: RGraph,
) -> None:
    coloring = make_state(
        graph,
        61,
    ).coloring_snapshot()

    verification = verify_coloring(coloring)

    report = evaluate_coloring(coloring)

    assert verification.coloring_hash == coloring.exact_hash()

    assert verification.score_report.total == report.total

    assert verification.score_report.by_color == report.by_color

    assert verification.ramsey_free == (report.total == 0)


def test_state_remains_consistent_over_long_flip_sequence(
    graph: RGraph,
) -> None:
    state = make_state(
        graph,
        62,
    )

    rng = np.random.default_rng(63)

    edges = rng.integers(
        0,
        state.number_of_edges,
        size=500,
    )

    for edge in edges:
        state.apply_edge_flip(int(edge))

    verification = verify_search_state(state)

    assert verification.consistent
    assert verification.errors == ()

    verification.require_consistent()


def test_verifier_detects_corrupted_histogram(
    graph: RGraph,
) -> None:
    state = make_state(
        graph,
        64,
    )

    # Tests deliberately corrupt a private field to prove
    # that independent verification detects the error.
    state._histogram[0] += 1

    verification = verify_search_state(state)

    assert not verification.consistent

    assert "histogram is incorrect" in verification.errors

    with pytest.raises(
        RuntimeError,
        match="inconsistent",
    ):
        verification.require_consistent()


def test_verifier_detects_corrupted_clique_counts(
    graph: RGraph,
) -> None:
    state = make_state(
        graph,
        65,
    )

    state._color_one_counts[0] ^= np.uint8(1)

    verification = verify_search_state(state)

    assert not verification.consistent

    assert "color-one clique counts " "are incorrect" in verification.errors


def test_verifier_detects_corrupted_score(
    graph: RGraph,
) -> None:
    state = make_state(
        graph,
        66,
    )

    state._score += 1

    verification = verify_search_state(state)

    assert not verification.consistent

    assert any(error.startswith("score is") for error in verification.errors)


def test_verifier_recognizes_ramsey_free_coloring() -> None:
    graph = RGraph(RProblem.r55(n_vertices=5))

    colors = np.zeros(
        graph.number_of_edges,
        dtype=np.uint8,
    )

    # The only K5 contains both colors, so it is not
    # monochromatic.
    colors[0] = 1

    verification = verify_coloring(
        RColoring(
            graph,
            colors,
        )
    )

    assert verification.ramsey_free

    assert verification.score_report.total == 0
