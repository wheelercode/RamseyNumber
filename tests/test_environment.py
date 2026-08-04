"""Test the refactored environment against the reference environment."""

import numpy as np
import pytest

import RamseyEnvironment as reference_environment
import RamseyTypes as reference_types

from ramsey.RColoring import RColoring
from ramsey.REnvironment import REnvironment
from ramsey.REnvironmentConfig import (
    REnvironmentConfig,
    RTabuMemoryConfig,
)
from ramsey.REnvironmentMemory import RTabuMemory
from ramsey.RGraph import RGraph
from ramsey.RObjective import (
    RDangerObjective,
    RMonochromaticObjective,
)
from ramsey.RProblem import RProblem


@pytest.fixture(scope="module")
def graph() -> RGraph:
    return RGraph(RProblem.r55(n_vertices=10))


def make_environments(
    graph: RGraph,
    seed: int,
    max_steps: int = 100,
):
    rng = np.random.default_rng(seed)

    colors = rng.integers(
        0,
        2,
        size=graph.number_of_edges,
        dtype=np.uint8,
    )

    subgraph_index = graph.subgraph_index(5)

    reference = reference_environment.RamseyTabuEnvironment(
        n_vertices=10,
        edges=graph.edges,
        kn_edges=subgraph_index.clique_edges,
        edge_to_kn=subgraph_index.edge_to_cliques,
        rng=np.random.default_rng(seed + 1),
        config=reference_types.TabuConfig(
            edge_tenure=5,
            visited_state_window=50,
            use_aspiration=True,
            max_steps=max_steps,
            danger_decay=0.25,
        ),
    )

    reference.reset(colors)

    parallel = REnvironment(
        graph=graph,
        objective=RDangerObjective(decay=0.25),
        memory=RTabuMemory(
            graph.number_of_edges,
            RTabuMemoryConfig(
                edge_tenure=5,
                visited_state_window=50,
            ),
        ),
        config=REnvironmentConfig(
            max_steps=max_steps,
            use_aspiration=True,
        ),
    )

    parallel.reset(
        RColoring(
            graph,
            colors,
        )
    )

    return parallel, reference


def assert_environment_states_equal(
    parallel: REnvironment,
    reference,
) -> None:
    assert np.array_equal(
        parallel.state.colors,
        reference.state.coloring,
    )

    assert np.array_equal(
        parallel.state.color_one_counts,
        reference.state.blue_counts,
    )

    assert np.array_equal(
        parallel.state.histogram,
        reference.state.histogram,
    )

    assert parallel.state.score == reference.state.score

    assert parallel.best_score == reference.best_score

    assert parallel.step_number == reference.step_number


def test_initial_environment_state_matches_reference(
    graph: RGraph,
) -> None:
    parallel, reference = make_environments(
        graph,
        seed=81,
    )

    assert_environment_states_equal(
        parallel,
        reference,
    )


def test_full_analysis_matches_reference(
    graph: RGraph,
) -> None:
    parallel, reference = make_environments(
        graph,
        seed=82,
    )

    actual = parallel.analyze_actions(use_cache=False)

    expected = reference.analyze_actions(use_cache=False)

    assert np.array_equal(
        actual.action_analysis.profiles,
        expected.profiles,
    )

    assert np.array_equal(
        actual.action_analysis.histogram_deltas,
        expected.histogram_deltas,
    )

    assert np.array_equal(
        actual.action_analysis.immediate_rewards,
        expected.immediate_rewards,
    )

    assert np.allclose(
        actual.objective_rewards,
        expected.danger_rewards,
    )

    assert np.array_equal(
        actual.memory_status.edge_tabu_mask,
        expected.edge_tabu_mask,
    )

    assert np.array_equal(
        actual.memory_status.revisit_mask,
        expected.revisit_mask,
    )

    assert np.array_equal(
        actual.aspiration_mask,
        expected.aspiration_mask,
    )

    assert np.array_equal(
        actual.available_mask,
        expected.available_mask,
    )

    assert actual.forced_fallback == expected.forced_fallback


def test_parallel_trajectory_matches_reference(
    graph: RGraph,
) -> None:
    parallel, reference = make_environments(
        graph,
        seed=83,
    )

    for _ in range(40):
        parallel_analysis = parallel.analyze_actions(use_cache=False)

        reference_analysis = reference.analyze_actions(use_cache=False)

        assert np.array_equal(
            parallel_analysis.available_mask,
            reference_analysis.available_mask,
        )

        # Choosing the largest resulting score keeps this
        # equivalence test away from early termination.
        masked_scores = np.where(
            parallel_analysis.available_mask,
            (parallel_analysis.action_analysis.resulting_scores),
            -1,
        )

        edge = int(np.argmax(masked_scores))

        actual = parallel.step(
            edge,
            full_analysis=True,
        )

        expected = reference.step(
            edge,
            full_analysis=True,
        )

        assert actual.immediate_reward == expected.immediate_reward

        assert np.isclose(
            actual.objective_reward,
            expected.danger_reward,
        )

        assert actual.score == expected.score

        assert actual.best_score == expected.best_score

        assert actual.terminated == expected.terminated

        assert actual.truncated == expected.truncated

        assert_environment_states_equal(
            parallel,
            reference,
        )


def test_fast_and_full_masks_match_reference_over_trajectory(
    graph: RGraph,
) -> None:
    parallel, reference = make_environments(
        graph,
        seed=84,
    )

    for _ in range(30):
        parallel_fast = parallel.available_action_mask_fast(use_cache=False)

        parallel_analysis = parallel.analyze_actions(use_cache=False)

        parallel_full = parallel_analysis.available_mask

        reference_fast = reference.available_action_mask_fast(use_cache=False)

        assert np.array_equal(
            parallel_fast,
            parallel_full,
        )

        assert np.array_equal(
            parallel_fast,
            reference_fast,
        )

        masked_scores = np.where(
            parallel_fast,
            (parallel_analysis.action_analysis.resulting_scores),
            -1,
        )

        edge = int(np.argmax(masked_scores))

        parallel.step(
            edge,
            full_analysis=False,
        )

        reference.step(
            edge,
            full_analysis=False,
        )


def test_environment_does_not_select_actions(
    graph: RGraph,
) -> None:
    parallel, _ = make_environments(
        graph,
        seed=85,
    )

    assert not hasattr(
        parallel,
        "select_random_available_action",
    )

    assert not hasattr(
        parallel,
        "select_greedy_action",
    )


def test_monochromatic_objective_reward_equals_score_reduction(
    graph: RGraph,
) -> None:
    colors = np.random.default_rng(86).integers(
        0,
        2,
        size=graph.number_of_edges,
        dtype=np.uint8,
    )

    environment = REnvironment(
        graph=graph,
        objective=RMonochromaticObjective(),
        memory=RTabuMemory(
            graph.number_of_edges,
            RTabuMemoryConfig(
                edge_tenure=0,
                visited_state_window=0,
            ),
        ),
    )

    environment.reset(
        RColoring(
            graph,
            colors,
        )
    )

    edge = int(environment.available_actions()[0])

    result = environment.step(edge)

    assert result.objective_reward == result.immediate_reward


def test_environment_truncates_at_step_limit() -> None:
    graph = RGraph(RProblem.r55(n_vertices=6))

    colors = np.zeros(
        graph.number_of_edges,
        dtype=np.uint8,
    )

    environment = REnvironment(
        graph=graph,
        objective=RMonochromaticObjective(),
        memory=RTabuMemory(
            graph.number_of_edges,
            RTabuMemoryConfig(
                edge_tenure=0,
                visited_state_window=0,
            ),
        ),
        config=REnvironmentConfig(max_steps=1),
    )

    environment.reset(
        RColoring(
            graph,
            colors,
        )
    )

    result = environment.step(0)

    assert result.truncated
    assert not result.terminated

    with pytest.raises(
        RuntimeError,
        match="episode has ended",
    ):
        environment.step(1)
