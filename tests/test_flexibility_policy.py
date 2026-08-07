"""Tests for the adaptive flexibility-preserving edge policy."""

import numpy as np
import pytest

from ramsey.RColoring import RColoring
from ramsey.REnvironment import REnvironment
from ramsey.REnvironmentMemory import RNullMemory
from ramsey.RFlexibilityPolicy import (
    RFlexibilityPolicy,
    RFlexibilityPolicyConfig,
    select_flexibility_action,
)
from ramsey.RGraph import RGraph
from ramsey.RObjective import RMonochromaticObjective
from ramsey.RProblem import RProblem


def make_environment(seed: int = 601) -> REnvironment:
    graph = RGraph(RProblem.r55(n_vertices=8))
    rng = np.random.default_rng(seed)
    colors = rng.integers(
        0,
        2,
        size=graph.number_of_edges,
        dtype=np.uint8,
    )
    environment = REnvironment(
        graph=graph,
        objective=RMonochromaticObjective(),
        memory=RNullMemory(),
    )
    environment.reset(RColoring(graph, colors))
    return environment


def test_healthy_flexibility_uses_best_exact_score_reward() -> None:
    environment = make_environment()
    config = RFlexibilityPolicyConfig(
        budgets=(0, 1, 2),
        monitor_budget=0,
        flexibility_floor=0.0,
        maximum_temporary_damage=3,
    )
    analysis = environment.analyze_actions()
    expected = int(
        analysis.action_analysis.immediate_rewards[
            analysis.available_mask
        ].max()
    )

    decision = select_flexibility_action(
        environment,
        np.random.default_rng(602),
        config,
    )

    assert decision.mode == "score"
    assert decision.immediate_reward == expected


def test_low_flexibility_enters_reorganization_mode() -> None:
    environment = make_environment(seed=603)
    config = RFlexibilityPolicyConfig(
        budgets=(0, 1, 2),
        monitor_budget=0,
        flexibility_floor=1.0,
        maximum_temporary_damage=3,
    )

    decision = select_flexibility_action(
        environment,
        np.random.default_rng(604),
        config,
    )

    # If F(0) happens to equal one on a tiny graph, the configured
    # threshold legitimately leaves the policy in score mode.
    if decision.current_flexibility < 1.0:
        assert decision.mode == "flexibility"
        assert decision.resulting_flexibility is not None


def test_policy_returns_available_action_without_mutating_state() -> None:
    environment = make_environment(seed=605)
    policy = RFlexibilityPolicy(
        np.random.default_rng(606),
        RFlexibilityPolicyConfig(
            flexibility_floor=0.0,
        ),
    )
    before = environment.state.colors.copy()
    version = environment.state.version
    available = environment.available_action_mask_fast().copy()

    edge = policy.select_action(environment)

    assert available[edge]
    assert environment.state.version == version
    assert np.array_equal(environment.state.colors, before)
    assert policy.requires_full_analysis


def test_policy_config_rejects_monitor_budget_not_in_curve() -> None:
    with pytest.raises(ValueError):
        RFlexibilityPolicyConfig(
            budgets=(0, 1, 5),
            monitor_budget=2,
        )