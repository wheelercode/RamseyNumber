"""Predicted versus actual action consequences."""

"""Characterize predicted versus actual action consequences."""

import numpy as np

import RamseyAction as action
import RamseyGraph as graph
import RamseySearch as search
import RamseyState as state_operations


def build_analysis(
    k10_k5_data,
):
    """
    Return one deterministic state and its all-action analysis.
    """
    coloring = graph.random_coloring(
        len(k10_k5_data.edges),
        np.random.default_rng(101),
    )

    state = search.create_search_state(
        coloring,
        k10_k5_data.kn_edges,
    )

    profiles = action.edge_kn_profiles(
        state,
        k10_k5_data.edge_to_kn,
    )

    deltas = action.all_action_histogram_deltas(
        state.coloring,
        profiles,
    )

    rewards = action.all_immediate_rewards(deltas)

    return (
        state,
        profiles,
        deltas,
        rewards,
    )


def test_every_profile_accounts_for_every_k5_containing_edge(
    k10_k5_data,
) -> None:
    _, profiles, _, _ = build_analysis(k10_k5_data)

    assert profiles.shape == (
        45,
        11,
    )

    assert np.all(profiles.sum(axis=1) == 56)


def test_all_predicted_rewards_equal_actual_score_changes(
    k10_k5_data,
) -> None:
    original, _, _, rewards = build_analysis(k10_k5_data)

    for edge, predicted_reward in enumerate(rewards):
        candidate = search.create_search_state(
            original.coloring,
            k10_k5_data.kn_edges,
        )

        actual_reward = state_operations.apply_edge_flip(
            candidate,
            k10_k5_data.edge_to_kn,
            edge,
        )

        assert actual_reward == int(predicted_reward)


def test_selected_edge_fast_rewards_equal_full_analysis(
    k10_k5_data,
) -> None:
    state, _, _, full_rewards = build_analysis(k10_k5_data)

    selected_edges = np.asarray(
        [
            0,
            3,
            11,
            22,
            44,
        ],
        dtype=np.int32,
    )

    fast_rewards = action.immediate_rewards_for_edges(
        state,
        k10_k5_data.edge_to_kn,
        selected_edges,
        edges_per_kn=10,
    )

    assert np.array_equal(
        fast_rewards,
        full_rewards[selected_edges],
    )


def test_resulting_histograms_match_applied_flips(
    k10_k5_data,
) -> None:
    original, _, deltas, _ = build_analysis(k10_k5_data)

    predicted_histograms = action.resulting_histograms(
        original,
        deltas,
    )

    for edge in range(len(original.coloring)):
        candidate = search.create_search_state(
            original.coloring,
            k10_k5_data.kn_edges,
        )

        state_operations.apply_edge_flip(
            candidate,
            k10_k5_data.edge_to_kn,
            edge,
        )

        assert np.array_equal(
            candidate.histogram,
            predicted_histograms[edge],
        )
