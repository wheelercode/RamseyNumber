"""Tests for process-parallel archive population."""

from pathlib import Path

import numpy as np
import pytest

from ramsey.RArchive import RSQLiteArchive
from ramsey.RArchiveBatchParallel import (
    RArchiveBatchParallel,
    RArchiveBatchParallelConfig,
)
from ramsey.RConstruction import RRandomConstruction
from ramsey.REnvironmentConfig import (
    REnvironmentConfig,
    RTabuMemoryConfig,
)
from ramsey.RGraph import RGraph
from ramsey.RProblem import RProblem
from ramsey.RSearchParallel import (
    RExactGreedyProcessConfig,
    RExactGreedyProcessPool,
)


def test_parallel_batch_config_validates_batch_size() -> None:
    with pytest.raises(
        ValueError,
        match="batch_size",
    ):
        RArchiveBatchParallelConfig(
            run_name="test",
            target_count=10,
            maximum_attempts=10,
            batch_size=0,
            action_seed_base=1,
        )


def test_parallel_batch_populates_archive(
    tmp_path: Path,
) -> None:
    problem = RProblem.r55(n_vertices=8)
    graph = RGraph(problem)

    process_config = RExactGreedyProcessConfig(
        problem=problem,
        environment=REnvironmentConfig(
            max_steps=5,
        ),
        memory=RTabuMemoryConfig(
            edge_tenure=2,
            visited_state_window=20,
        ),
    )

    with RSQLiteArchive(
        tmp_path / "parallel.sqlite3"
    ) as archive:
        with RExactGreedyProcessPool(
            process_config,
            max_workers=2,
        ) as pool:
            batch = RArchiveBatchParallel(
                graph=graph,
                construction=RRandomConstruction(
                    np.random.default_rng(101)
                ),
                archive=archive,
                search_pool=pool,
            )

            result = batch.populate(
                RArchiveBatchParallelConfig(
                    run_name="parallel-test",
                    target_count=3,
                    maximum_attempts=5,
                    batch_size=2,
                    action_seed_base=200,
                    maximum_score=0,
                )
            )

        assert result.target_reached
        assert result.initial_eligible_count == 0
        assert result.final_eligible_count == 3
        assert result.attempts_completed == 3
        assert len(result.passes) == 2
        assert result.best_score == 0


def test_parallel_batch_observer_receives_one_event_per_pass(
    tmp_path: Path,
) -> None:
    problem = RProblem.r55(n_vertices=8)
    graph = RGraph(problem)
    observed_passes = []

    with RSQLiteArchive(
        tmp_path / "observer.sqlite3"
    ) as archive:
        process_config = RExactGreedyProcessConfig(
            problem=problem,
            environment=REnvironmentConfig(
                max_steps=5,
            ),
            memory=RTabuMemoryConfig(),
        )

        with RExactGreedyProcessPool(
            process_config,
            max_workers=1,
        ) as pool:
            batch = RArchiveBatchParallel(
                graph=graph,
                construction=RRandomConstruction(
                    np.random.default_rng(301)
                ),
                archive=archive,
                search_pool=pool,
            )

            result = batch.populate(
                RArchiveBatchParallelConfig(
                    run_name="observer-test",
                    target_count=3,
                    maximum_attempts=4,
                    batch_size=2,
                    action_seed_base=400,
                    maximum_score=0,
                ),
                observer=observed_passes.append,
            )

    assert len(observed_passes) == len(result.passes)
    assert [
        parallel_pass.pass_number
        for parallel_pass in observed_passes
    ] == list(range(len(observed_passes)))