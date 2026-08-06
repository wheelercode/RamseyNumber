"""Test maximum-violation-load edge selection."""

import numpy as np

from ramsey.RColoring import RColoring
from ramsey.REdgeViolationAction import (
    analyze_edge_violations,
)
from ramsey.REdgeViolationPolicy import (
    REdgeViolationPolicy,
)
from ramsey.REnvironment import REnvironment
from ramsey.REnvironmentConfig import (
    REnvironmentConfig,
    RTabuMemoryConfig,
)
from ramsey.REnvironmentMemory import RTabuMemory
from ramsey.RGraph import RGraph
from ramsey.RObjective import RDangerObjective
from ramsey.RProblem import RProblem


def make_environment(
    seed: int,
) -> REnvironment:
    graph = RGraph(
        RProblem.r55(n_vertices=10)
    )

    colors = np.random.default_rng(seed).integers(
        0,
        2,
        size=graph.number_of_edges,
        dtype=np.uint8,
    )

    environment = REnvironment(
        graph=graph,
        objective=RDangerObjective(decay=0.25),
        memory=RTabuMemory(
            graph.number_of_edges,
            RTabuMemoryConfig(
                edge_tenure=5,
                visited_state_window=50,
            ),
        ),
        config=REnvironmentConfig(max_steps=100),
    )

    environment.reset(
        RColoring(
            graph,
            colors,
        )
    )

    return environment


def test_policy_selects_maximum_available_violation_load() -> None:
    environment = make_environment(seed=411)

    policy = REdgeViolationPolicy(
        np.random.default_rng(412)
    )

    available_mask = (
        environment.available_action_mask_fast()
    )

    analysis = analyze_edge_violations(
        environment.state
    )

    expected_load = int(
        analysis.violation_loads[
            available_mask
        ].max()
    )

    edge = policy.select_action(environment)

    assert available_mask[edge]
    assert (
        analysis.violation_loads[edge]
        == expected_load
    )
    assert not policy.requires_full_analysis
    assert policy.name == "edge-violation-load"


def test_policy_selection_does_not_mutate_state() -> None:
    environment = make_environment(seed=413)

    policy = REdgeViolationPolicy(
        np.random.default_rng(414)
    )

    version = environment.state.version
    colors = environment.state.colors.copy()

    policy.select_action(environment)

    assert environment.state.version == version
    assert np.array_equal(
        environment.state.colors,
        colors,
    )


def test_policy_tie_breaking_is_reproducible() -> None:
    first_environment = make_environment(seed=415)
    second_environment = make_environment(seed=415)

    first_policy = REdgeViolationPolicy(
        np.random.default_rng(416)
    )
    second_policy = REdgeViolationPolicy(
        np.random.default_rng(416)
    )

    assert first_policy.select_action(
        first_environment
    ) == second_policy.select_action(
        second_environment
    )