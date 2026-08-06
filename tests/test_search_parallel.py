"""Test process-parallel exact-greedy search execution."""

import numpy as np
import pytest

from ramsey.RColoring import RColoring
from ramsey.REnvironment import REnvironment
from ramsey.REnvironmentConfig import (
    REnvironmentConfig,
    RTabuMemoryConfig,
)
from ramsey.REnvironmentMemory import RTabuMemory
from ramsey.RGraph import RGraph
from ramsey.RObjective import RMonochromaticObjective
from ramsey.RPolicy import RGreedyPolicy
from ramsey.RProblem import RProblem
from ramsey.RSearch import RSearch
from ramsey.RSearchParallel import (
    RExactGreedyProcessConfig,
    RExactGreedyProcessPool,
    RParallelSearchTask,
)


def test_parallel_task_owns_read_only_colors() -> None:
    colors = np.zeros(
        28,
        dtype=np.uint8,
    )

    task = RParallelSearchTask(
        task_id=3,
        colors=colors,
        action_seed=17,
    )

    colors[0] = 1

    assert task.colors[0] == 0
    assert not task.colors.flags.writeable


def test_parallel_pool_validates_worker_count() -> None:
    config = RExactGreedyProcessConfig(
        problem=RProblem.r55(n_vertices=8),
        environment=REnvironmentConfig(max_steps=5),
        memory=RTabuMemoryConfig(),
    )

    with pytest.raises(
        ValueError,
        match="max_workers",
    ):
        RExactGreedyProcessPool(
            config,
            max_workers=0,
        )


def test_one_worker_matches_serial_exact_greedy() -> None:
    problem = RProblem.r55(n_vertices=8)
    graph = RGraph(problem)

    colors = np.random.default_rng(701).integers(
        0,
        2,
        size=graph.number_of_edges,
        dtype=np.uint8,
    )

    environment_config = REnvironmentConfig(
        max_steps=20,
    )

    memory_config = RTabuMemoryConfig(
        edge_tenure=5,
        visited_state_window=50,
    )

    action_seed = 702

    serial_search = RSearch(
        environment=REnvironment(
            graph=graph,
            objective=RMonochromaticObjective(),
            memory=RTabuMemory(
                graph.number_of_edges,
                memory_config,
            ),
            config=environment_config,
        ),
        policy=RGreedyPolicy(
            rng=np.random.default_rng(
                action_seed
            ),
            use_objective_reward=False,
        ),
    )

    serial_result = serial_search.run(
        RColoring(graph, colors)
    )

    parallel_config = RExactGreedyProcessConfig(
        problem=problem,
        environment=environment_config,
        memory=memory_config,
    )

    task = RParallelSearchTask(
        task_id=11,
        colors=colors,
        action_seed=action_seed,
    )

    with RExactGreedyProcessPool(
        parallel_config,
        max_workers=1,
    ) as pool:
        parallel_result = pool.run(
            (task,)
        )[0]

    assert parallel_result.task_id == 11
    assert (
        parallel_result.initial_score
        == serial_result.initial_score
    )
    assert (
        parallel_result.final_score
        == serial_result.final_score
    )
    assert (
        parallel_result.best_score
        == serial_result.best_score
    )
    assert np.array_equal(
        parallel_result.best_colors,
        serial_result.best_coloring.colors,
    )