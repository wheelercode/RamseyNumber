"""Tests for structurally restricted exact-score greedy selection."""

import numpy as np
import pytest

from ramsey.RColoring import RColoring
from ramsey.REnvironment import REnvironment
from ramsey.REnvironmentConfig import REnvironmentConfig
from ramsey.REnvironmentMemory import RNullMemory
from ramsey.RGraph import RGraph
from ramsey.RHistogramBandPolicy import (
    RHistogramBandGreedyPolicy,
    RHistogramBandPolicyConfig,
    histogram_band_loads,
)
from ramsey.RObjective import RMonochromaticObjective
from ramsey.RProblem import RProblem


@pytest.fixture(scope="module")
def graph() -> RGraph:
    """Return a small R(5,5) topology for policy tests."""
    return RGraph(RProblem.r55(n_vertices=9))


def make_environment(
    graph: RGraph,
    seed: int = 17,
) -> REnvironment:
    """Return a reset deterministic random-coloring environment."""
    rng = np.random.default_rng(seed)
    coloring = RColoring(
        graph,
        rng.integers(
            0,
            2,
            size=graph.number_of_edges,
            dtype=np.uint8,
        ),
    )
    environment = REnvironment(
        graph=graph,
        objective=RMonochromaticObjective(),
        memory=RNullMemory(),
        config=REnvironmentConfig(max_steps=8),
    )
    environment.reset(coloring)
    return environment


def test_histogram_band_loads_match_action_profiles(
    graph: RGraph,
) -> None:
    """Band loads equal the explicit sum of selected profile bins."""
    environment = make_environment(graph)

    expected = environment.state.action_profiles[:, 3:8].sum(
        axis=1,
        dtype=np.int64,
    )

    actual = histogram_band_loads(
        environment.state,
        3,
        7,
    )

    assert np.array_equal(actual, expected)


def test_policy_maximizes_exact_reward_inside_top_band_pool(
    graph: RGraph,
) -> None:
    """Greedy score ranking occurs only after structural filtering."""
    environment = make_environment(graph, seed=18)
    config = RHistogramBandPolicyConfig(
        h_min=3,
        h_max=7,
        candidate_pool_size=7,
    )
    policy = RHistogramBandGreedyPolicy(
        np.random.default_rng(19),
        config,
    )

    analysis = environment.analyze_actions()
    loads = histogram_band_loads(environment.state, 3, 7)
    available = np.flatnonzero(analysis.available_mask)
    target = min(config.candidate_pool_size, len(available))
    cutoff = np.sort(loads[available])[-target]
    candidate_mask = analysis.available_mask & (loads >= cutoff)
    rewards = analysis.action_analysis.immediate_rewards
    expected_reward = int(rewards[candidate_mask].max())

    edge = policy.select_action(environment)

    assert candidate_mask[edge]
    assert int(rewards[edge]) == expected_reward
    assert policy.last_decision is not None
    assert policy.last_decision.candidate_count >= target
    assert policy.last_decision.cutoff_band_load == int(cutoff)


def test_pool_expands_to_all_available_edges_when_requested(
    graph: RGraph,
) -> None:
    """An oversized structural pool reduces to ordinary exact greedy."""
    environment = make_environment(graph, seed=20)
    policy = RHistogramBandGreedyPolicy(
        np.random.default_rng(21),
        RHistogramBandPolicyConfig(
            h_min=3,
            h_max=7,
            candidate_pool_size=10_000,
        ),
    )
    analysis = environment.analyze_actions()
    expected = int(
        analysis.action_analysis.immediate_rewards[
            analysis.available_mask
        ].max()
    )

    edge = policy.select_action(environment)

    assert int(analysis.action_analysis.immediate_rewards[edge]) == expected


@pytest.mark.parametrize(
    ("kwargs", "exception"),
    [
        ({"h_min": -1}, ValueError),
        ({"h_min": 8, "h_max": 7}, ValueError),
        ({"candidate_pool_size": 0}, ValueError),
        ({"h_min": 3.5}, TypeError),
    ],
)
def test_config_rejects_invalid_values(
    kwargs: dict,
    exception: type[Exception],
) -> None:
    """Configuration rejects malformed structural-pool controls."""
    with pytest.raises(exception):
        RHistogramBandPolicyConfig(**kwargs)


def test_band_rejects_histogram_bin_beyond_state(
    graph: RGraph,
) -> None:
    """Selection cannot reference a histogram bin that does not exist."""
    environment = make_environment(graph)

    with pytest.raises(ValueError, match="cannot exceed"):
        histogram_band_loads(
            environment.state,
            3,
            11,
        )