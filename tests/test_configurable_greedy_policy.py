"""Tests for percentile-controlled exact-score greedy selection."""

import numpy as np

from ramsey import (
    RConfigurableGreedyPolicy,
    RConfigurableGreedyPolicyConfig,
    REnvironment,
    REnvironmentConfig,
    RGraph,
    RMonochromaticObjective,
    RNullMemory,
    RProblem,
    RRandomConstruction,
)


def _environment(seed: int = 1234) -> REnvironment:
    """Return a reset small environment containing positive actions."""
    graph = RGraph(RProblem.r55(n_vertices=15))
    coloring = RRandomConstruction(
        np.random.default_rng(seed)
    ).construct(graph)
    environment = REnvironment(
        graph=graph,
        objective=RMonochromaticObjective(),
        memory=RNullMemory(),
        config=REnvironmentConfig(max_steps=1_000),
    )
    environment.reset(coloring)
    return environment


def test_greediness_one_matches_maximum_positive_reward() -> None:
    """Greediness one must preserve ordinary exact-score greedy behavior."""
    environment = _environment()
    policy = RConfigurableGreedyPolicy(
        rng=np.random.default_rng(1),
        config=RConfigurableGreedyPolicyConfig(greediness=1.0),
    )

    edge = policy.select_action(environment)
    analysis = environment.analyze_actions()
    rewards = analysis.action_analysis.immediate_rewards
    maximum = int(rewards[analysis.available_mask].max())

    assert maximum > 0
    assert int(rewards[edge]) == maximum


def test_greediness_zero_selects_smallest_positive_reward() -> None:
    """Greediness zero must still improve score, but as little as possible."""
    environment = _environment()
    policy = RConfigurableGreedyPolicy(
        rng=np.random.default_rng(2),
        config=RConfigurableGreedyPolicyConfig(greediness=0.0),
    )

    edge = policy.select_action(environment)
    analysis = environment.analyze_actions()
    rewards = analysis.action_analysis.immediate_rewards
    positive = rewards[analysis.available_mask & (rewards > 0)]

    assert positive.size > 0
    assert int(rewards[edge]) == int(positive.min())


def test_configurable_greedy_rejects_invalid_percentiles() -> None:
    """Greediness must remain on the closed unit interval."""
    for value in (-0.01, 1.01):
        try:
            RConfigurableGreedyPolicyConfig(greediness=value)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid greediness was accepted.")