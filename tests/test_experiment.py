"""Test reproducible search experiment execution."""

import numpy as np
import pytest

from ramsey.RArchive import RSQLiteArchive
from ramsey.RColoring import RColoring
from ramsey.RConstruction import (
    RFixedConstruction,
)
from ramsey.REnvironment import REnvironment
from ramsey.REnvironmentConfig import (
    REnvironmentConfig,
    RTabuMemoryConfig,
)
from ramsey.REnvironmentMemory import (
    RTabuMemory,
)
from ramsey.RExperiment import (
    RExperiment,
    RExperimentConfig,
)
from ramsey.RGraph import RGraph
from ramsey.RObjective import (
    RMonochromaticObjective,
)
from ramsey.RPolicy import RPolicy
from ramsey.RProblem import RProblem
from ramsey.RSearch import RSearch


class RFirstAvailablePolicy(RPolicy):
    @property
    def name(self) -> str:
        return "first-available"

    def select_action(
        self,
        environment: REnvironment,
    ) -> int:
        return int(environment.available_actions()[0])


def make_experiment(
    n_vertices: int,
    max_steps: int,
    colors: np.ndarray,
    archive=None,
) -> RExperiment:
    graph = RGraph(RProblem.r55(n_vertices=n_vertices))

    construction = RFixedConstruction(
        RColoring(
            graph,
            colors,
        )
    )

    environment = REnvironment(
        graph=graph,
        objective=RMonochromaticObjective(),
        memory=RTabuMemory(
            graph.number_of_edges,
            RTabuMemoryConfig(
                edge_tenure=2,
                visited_state_window=20,
            ),
        ),
        config=REnvironmentConfig(max_steps=max_steps),
    )

    search = RSearch(
        environment,
        RFirstAvailablePolicy(),
    )

    return RExperiment(
        graph,
        construction,
        search,
        archive,
    )


def test_experiment_runs_searches_and_archives_best_colorings(
    tmp_path,
) -> None:
    graph = RGraph(RProblem.r55(n_vertices=10))

    colors = np.zeros(
        graph.number_of_edges,
        dtype=np.uint8,
    )

    with RSQLiteArchive(tmp_path / "experiment.sqlite3") as archive:
        experiment = make_experiment(
            10,
            2,
            colors,
            archive,
        )

        observed_iterations = []

        result = experiment.run(
            RExperimentConfig(
                run_name="fixed-seeds",
                iterations=3,
                start_iteration=7,
                record_steps=True,
            ),
            observer=observed_iterations.append,
        )

        assert result.completed_iterations == 3

        assert [item.iteration for item in result.iteration_results] == [
            7,
            8,
            9,
        ]

        assert observed_iterations == list(result.iteration_results)

        assert all(
            len(item.search_result.step_results) == 2
            for item in result.iteration_results
        )

        assert archive.coloring_count(experiment.graph) == 1

        assert result.iteration_results[0].new_archive_best

        assert not (result.iteration_results[1].new_archive_best)

        assert result.iteration_results[-1].archive_record.times_seen == 3

        assert result.best_score == (result.best_iteration.search_result.best_score)


def test_experiment_can_run_without_archive() -> None:
    graph = RGraph(RProblem.r55(n_vertices=10))

    colors = np.zeros(
        graph.number_of_edges,
        dtype=np.uint8,
    )

    experiment = make_experiment(
        10,
        1,
        colors,
    )

    result = experiment.run(
        RExperimentConfig(
            run_name="memory-only",
            iterations=2,
        )
    )

    assert result.completed_iterations == 2

    assert all(
        item.archive_record is None and not item.new_archive_best
        for item in result.iteration_results
    )


def test_experiment_stops_when_solution_is_found() -> None:
    graph = RGraph(RProblem.r55(n_vertices=5))

    colors = np.zeros(
        graph.number_of_edges,
        dtype=np.uint8,
    )

    colors[0] = 1

    experiment = make_experiment(
        5,
        10,
        colors,
    )

    result = experiment.run(
        RExperimentConfig(
            run_name="solution",
            iterations=10,
            stop_on_solution=True,
        )
    )

    assert result.completed_iterations == 1

    assert result.solved
    assert result.best_score == 0


@pytest.mark.parametrize(
    (
        "values",
        "exception_type",
    ),
    [
        (
            {
                "run_name": "",
                "iterations": 1,
            },
            ValueError,
        ),
        (
            {
                "run_name": "run",
                "iterations": 0,
            },
            ValueError,
        ),
        (
            {
                "run_name": "run",
                "iterations": 1.5,
            },
            TypeError,
        ),
        (
            {
                "run_name": "run",
                "iterations": 1,
                "start_iteration": -1,
            },
            ValueError,
        ),
        (
            {
                "run_name": "run",
                "iterations": 1,
                "record_steps": 1,
            },
            TypeError,
        ),
    ],
)
def test_experiment_config_validation(
    values,
    exception_type,
) -> None:
    with pytest.raises(exception_type):
        RExperimentConfig(**values)


def test_experiment_rejects_mismatched_search_problem() -> None:
    graph10 = RGraph(RProblem.r55(n_vertices=10))

    graph11 = RGraph(RProblem.r55(n_vertices=11))

    coloring = RColoring(
        graph10,
        np.zeros(
            graph10.number_of_edges,
            dtype=np.uint8,
        ),
    )

    environment = REnvironment(
        graph=graph11,
        objective=RMonochromaticObjective(),
        memory=RTabuMemory(graph11.number_of_edges),
    )

    search = RSearch(
        environment,
        RFirstAvailablePolicy(),
    )

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        RExperiment(
            graph10,
            RFixedConstruction(coloring),
            search,
        )
