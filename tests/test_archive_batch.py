"""Test bounded batch population of coloring archives."""

import numpy as np
import pytest

from ramsey.RArchive import RSQLiteArchive
from ramsey.RArchiveBatch import (
    RArchiveBatch,
    RArchiveBatchConfig,
)
from ramsey.RColoring import RColoring
from ramsey.RConstruction import (
    RConstruction,
    RFixedConstruction,
)
from ramsey.REnvironment import REnvironment
from ramsey.REnvironmentConfig import REnvironmentConfig
from ramsey.REnvironmentMemory import RNullMemory
from ramsey.RGraph import RGraph
from ramsey.RObjective import RMonochromaticObjective
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


class RSequenceConstruction(RConstruction):
    """Return deterministic colorings in a repeating sequence."""

    def __init__(
        self,
        colorings: tuple[RColoring, ...],
    ) -> None:
        self._colorings = colorings
        self._next = 0

    @property
    def name(self) -> str:
        return "sequence"

    def construct(
        self,
        graph: RGraph,
    ) -> RColoring:
        coloring = self._colorings[
            self._next % len(self._colorings)
        ]

        self._next += 1

        return RColoring(
            graph,
            coloring.colors,
        )


def make_graph_and_search() -> tuple[
    RGraph,
    RSearch,
]:
    graph = RGraph(RProblem.r55(n_vertices=5))

    environment = REnvironment(
        graph=graph,
        objective=RMonochromaticObjective(),
        memory=RNullMemory(),
        config=REnvironmentConfig(max_steps=2),
    )

    search = RSearch(
        environment,
        RFirstAvailablePolicy(),
    )

    return (
        graph,
        search,
    )


def nonmonochromatic_coloring(
    graph: RGraph,
    blue_edge: int,
) -> RColoring:
    colors = np.zeros(
        graph.number_of_edges,
        dtype=np.uint8,
    )

    colors[blue_edge] = 1

    return RColoring(
        graph,
        colors,
    )


def test_archive_batch_stops_when_target_pool_is_full(
    tmp_path,
) -> None:
    (
        graph,
        search,
    ) = make_graph_and_search()

    construction = RSequenceConstruction(
        (
            nonmonochromatic_coloring(graph, 0),
            nonmonochromatic_coloring(graph, 1),
        )
    )

    with RSQLiteArchive(tmp_path / "batch.sqlite3") as archive:
        batch = RArchiveBatch(
            graph=graph,
            construction=construction,
            search=search,
            archive=archive,
        )

        observed = []

        result = batch.populate(
            RArchiveBatchConfig(
                run_name="score-zero-pool",
                target_count=2,
                maximum_attempts=10,
                maximum_score=0,
                start_iteration=20,
            ),
            observer=observed.append,
        )

        assert result.target_reached

        assert result.attempts_completed == 2

        assert result.initial_eligible_count == 0

        assert result.final_eligible_count == 2

        assert result.new_eligible_colorings == 2

        assert result.best_score == 0

        assert [
            item.iteration
            for item in result.attempt_results
        ] == [
            20,
            21,
        ]

        assert observed == list(result.attempt_results)

        assert archive.coloring_count(graph) == 2


def test_archive_batch_does_not_count_duplicates_twice(
    tmp_path,
) -> None:
    (
        graph,
        search,
    ) = make_graph_and_search()

    coloring = nonmonochromatic_coloring(
        graph,
        0,
    )

    with RSQLiteArchive(tmp_path / "duplicates.sqlite3") as archive:
        batch = RArchiveBatch(
            graph=graph,
            construction=RFixedConstruction(coloring),
            search=search,
            archive=archive,
        )

        result = batch.populate(
            RArchiveBatchConfig(
                run_name="duplicates",
                target_count=2,
                maximum_attempts=3,
                maximum_score=0,
            )
        )

        assert not result.target_reached

        assert result.attempts_completed == 3

        assert result.final_eligible_count == 1

        assert result.attempt_results[-1].archive_record.times_seen == 3

        assert [
            item.new_unique_coloring
            for item in result.attempt_results
        ] == [
            True,
            False,
            False,
        ]


def test_archive_batch_can_discard_out_of_range_results(
    tmp_path,
) -> None:
    (
        graph,
        search,
    ) = make_graph_and_search()

    coloring = nonmonochromatic_coloring(
        graph,
        0,
    )

    with RSQLiteArchive(tmp_path / "discard.sqlite3") as archive:
        batch = RArchiveBatch(
            graph=graph,
            construction=RFixedConstruction(coloring),
            search=search,
            archive=archive,
        )

        result = batch.populate(
            RArchiveBatchConfig(
                run_name="discard",
                target_count=1,
                maximum_attempts=2,
                minimum_score=1,
                save_out_of_range=False,
            )
        )

        assert not result.target_reached

        assert archive.coloring_count(graph) == 0

        assert all(
            item.archive_record is None
            and not item.in_score_range
            for item in result.attempt_results
        )


def test_archive_batch_preserves_out_of_range_results_by_default(
    tmp_path,
) -> None:
    (
        graph,
        search,
    ) = make_graph_and_search()

    coloring = nonmonochromatic_coloring(
        graph,
        0,
    )

    with RSQLiteArchive(tmp_path / "preserve.sqlite3") as archive:
        batch = RArchiveBatch(
            graph=graph,
            construction=RFixedConstruction(coloring),
            search=search,
            archive=archive,
        )

        result = batch.populate(
            RArchiveBatchConfig(
                run_name="preserve",
                target_count=1,
                maximum_attempts=2,
                minimum_score=1,
            )
        )

        assert not result.target_reached

        assert result.final_eligible_count == 0

        assert archive.coloring_count(graph) == 1

        assert all(
            item.archive_record is not None
            for item in result.attempt_results
        )


def test_archive_batch_skips_work_when_target_already_exists(
    tmp_path,
) -> None:
    (
        graph,
        search,
    ) = make_graph_and_search()

    coloring = nonmonochromatic_coloring(
        graph,
        0,
    )

    with RSQLiteArchive(tmp_path / "existing.sqlite3") as archive:
        archive.save_coloring(
            coloring,
            run_name="existing",
            iteration=0,
        )

        batch = RArchiveBatch(
            graph=graph,
            construction=RFixedConstruction(coloring),
            search=search,
            archive=archive,
        )

        result = batch.populate(
            RArchiveBatchConfig(
                run_name="already-full",
                target_count=1,
                maximum_attempts=10,
                maximum_score=0,
            )
        )

        assert result.target_reached

        assert result.attempts_completed == 0

        assert result.best_score is None


@pytest.mark.parametrize(
    (
        "values",
        "exception_type",
    ),
    [
        (
            {
                "run_name": "",
                "target_count": 1,
                "maximum_attempts": 1,
            },
            ValueError,
        ),
        (
            {
                "run_name": "run",
                "target_count": 0,
                "maximum_attempts": 1,
            },
            ValueError,
        ),
        (
            {
                "run_name": "run",
                "target_count": 1.5,
                "maximum_attempts": 1,
            },
            TypeError,
        ),
        (
            {
                "run_name": "run",
                "target_count": 1,
                "maximum_attempts": 0,
            },
            ValueError,
        ),
        (
            {
                "run_name": "run",
                "target_count": 1,
                "maximum_attempts": 1,
                "start_iteration": -1,
            },
            ValueError,
        ),
        (
            {
                "run_name": "run",
                "target_count": 1,
                "maximum_attempts": 1,
                "minimum_score": 2,
                "maximum_score": 1,
            },
            ValueError,
        ),
        (
            {
                "run_name": "run",
                "target_count": 1,
                "maximum_attempts": 1,
                "record_steps": 1,
            },
            TypeError,
        ),
        (
            {
                "run_name": "run",
                "target_count": 1,
                "maximum_attempts": 1,
                "save_out_of_range": 1,
            },
            TypeError,
        ),
    ],
)
def test_archive_batch_config_validation(
    values,
    exception_type,
) -> None:
    with pytest.raises(exception_type):
        RArchiveBatchConfig(**values)