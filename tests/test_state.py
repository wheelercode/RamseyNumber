"""Incremental state versus full recomputation."""

"""Characterize incremental state against full recomputation."""

import numpy as np

import RamseyGraph as graph
import RamseySearch as search
import RamseyState as state_operations


def test_create_search_state_owns_its_coloring_copy(
    k10_k5_data,
) -> None:
    coloring = graph.random_coloring(
        len(k10_k5_data.edges),
        np.random.default_rng(31),
    )

    state = search.create_search_state(
        coloring,
        k10_k5_data.kn_edges,
    )

    original_value = int(state.coloring[0])

    coloring[0] ^= np.uint8(1)

    assert state.coloring[0] == original_value


def test_incremental_state_survives_long_flip_sequence(
    k10_k5_data,
) -> None:
    rng = np.random.default_rng(8675309)

    coloring = graph.random_coloring(
        len(k10_k5_data.edges),
        rng,
    )

    state = search.create_search_state(
        coloring,
        k10_k5_data.kn_edges,
    )

    edges_to_flip = rng.integers(
        0,
        len(coloring),
        size=500,
    )

    for edge in edges_to_flip:
        previous_score = state.score

        reward = state_operations.apply_edge_flip(
            state,
            k10_k5_data.edge_to_kn,
            int(edge),
        )

        assert reward == (previous_score - state.score)

    state_operations.verify_search_state(
        state,
        k10_k5_data.kn_edges,
    )


def test_flipping_the_same_edge_twice_restores_state(
    k10_k5_data,
) -> None:
    coloring = graph.random_coloring(
        len(k10_k5_data.edges),
        np.random.default_rng(99),
    )

    state = search.create_search_state(
        coloring,
        k10_k5_data.kn_edges,
    )

    original_coloring = state.coloring.copy()

    original_histogram = state.histogram.copy()

    original_score = state.score

    state_operations.apply_edge_flip(
        state,
        k10_k5_data.edge_to_kn,
        17,
    )

    state_operations.apply_edge_flip(
        state,
        k10_k5_data.edge_to_kn,
        17,
    )

    assert np.array_equal(
        state.coloring,
        original_coloring,
    )

    assert np.array_equal(
        state.histogram,
        original_histogram,
    )

    assert state.score == original_score
