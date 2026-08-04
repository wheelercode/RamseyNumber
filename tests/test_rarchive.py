"""Test persistent coloring, provenance, and leaderboard storage."""

import numpy as np
import pytest

from RamseyDatabase import ColoringDatabase

from ramsey.RArchive import RSQLiteArchive
from ramsey.RColoring import RColoring
from ramsey.RGraph import RGraph
from ramsey.RProblem import RProblem
from ramsey.RScoring import (
    binary_histogram,
    score_coloring,
)


def random_coloring(
    graph: RGraph,
    seed: int,
) -> RColoring:
    colors = np.random.default_rng(seed).integers(
        0,
        2,
        size=graph.number_of_edges,
        dtype=np.uint8,
    )

    return RColoring(
        graph,
        colors,
    )


def test_archive_round_trip_and_duplicate_observation(
    tmp_path,
) -> None:
    graph = RGraph(RProblem.r55(n_vertices=10))

    coloring = random_coloring(
        graph,
        seed=211,
    )

    with RSQLiteArchive(tmp_path / "colorings.sqlite3") as archive:
        first = archive.save_coloring(
            coloring,
            run_name="round-trip",
            iteration=1,
        )

        second = archive.save_coloring(
            coloring,
            run_name="round-trip",
            iteration=2,
        )

        restored = archive.load_coloring(
            first.coloring_id,
            graph,
        )

        assert archive.coloring_count() == 1

        assert archive.best_score() == (score_coloring(coloring))

        assert first.coloring_id == second.coloring_id

        assert second.times_seen == 2
        assert second.iteration == 2

        assert restored.coloring.exact_equals(coloring)

        assert np.array_equal(
            restored.histogram,
            binary_histogram(coloring),
        )

        assert not (restored.histogram.flags.writeable)


def test_archive_orders_best_colorings(
    tmp_path,
) -> None:
    graph = RGraph(RProblem.r55(n_vertices=10))

    colorings = [
        random_coloring(
            graph,
            seed,
        )
        for seed in range(212, 220)
    ]

    with RSQLiteArchive(tmp_path / "leaderboard.sqlite3") as archive:
        for iteration, coloring in enumerate(colorings):
            archive.save_coloring(
                coloring,
                run_name="leaderboard",
                iteration=iteration,
            )

        records = archive.best_colorings(
            limit=5,
            graph=graph,
        )

    scores = [record.score for record in records]

    assert scores == sorted(scores)
    assert len(records) == 5


def test_archive_filters_different_problems(
    tmp_path,
) -> None:
    graph10 = RGraph(RProblem.r55(n_vertices=10))

    graph11 = RGraph(RProblem.r55(n_vertices=11))

    with RSQLiteArchive(tmp_path / "problems.sqlite3") as archive:
        archive.save_coloring(
            random_coloring(
                graph10,
                seed=221,
            ),
            run_name="k10",
            iteration=0,
        )

        archive.save_coloring(
            random_coloring(
                graph11,
                seed=222,
            ),
            run_name="k11",
            iteration=0,
        )

        assert archive.coloring_count() == 2

        assert archive.coloring_count(graph10) == 1

        assert archive.coloring_count(graph11) == 1


def test_archive_rejects_unknown_coloring_id(
    tmp_path,
) -> None:
    graph = RGraph(RProblem.r55(n_vertices=10))

    with RSQLiteArchive(tmp_path / "unknown.sqlite3") as archive:
        with pytest.raises(
            KeyError,
            match="Unknown coloring ID",
        ):
            archive.load_coloring(
                999,
                graph,
            )


@pytest.mark.parametrize(
    (
        "run_name",
        "iteration",
        "exception_type",
    ),
    [
        (
            "",
            0,
            ValueError,
        ),
        (
            "run",
            -1,
            ValueError,
        ),
        (
            "run",
            1.5,
            TypeError,
        ),
    ],
)
def test_archive_validates_provenance(
    tmp_path,
    run_name,
    iteration,
    exception_type,
) -> None:
    graph = RGraph(RProblem.r55(n_vertices=10))

    coloring = random_coloring(
        graph,
        seed=223,
    )

    with RSQLiteArchive(tmp_path / "invalid.sqlite3") as archive:
        with pytest.raises(exception_type):
            archive.save_coloring(
                coloring,
                run_name=run_name,
                iteration=iteration,
            )


def test_closed_archive_rejects_operations(
    tmp_path,
) -> None:
    archive = RSQLiteArchive(tmp_path / "closed.sqlite3")

    archive.close()

    with pytest.raises(
        RuntimeError,
        match="closed",
    ):
        archive.coloring_count()


def test_archive_is_compatible_with_reference_database(
    tmp_path,
) -> None:
    graph = RGraph(RProblem.r55(n_vertices=10))

    coloring = random_coloring(
        graph,
        seed=224,
    )

    histogram = binary_histogram(coloring)

    database_path = tmp_path / "compatible.sqlite3"

    with ColoringDatabase(database_path) as reference:
        original = reference.save_coloring(
            coloring.colors,
            histogram,
            n_vertices=10,
            k_size=5,
            run_name="reference",
            iteration=1,
        )

    with RSQLiteArchive(database_path) as archive:
        observed = archive.save_coloring(
            coloring,
            run_name="refactor",
            iteration=2,
        )

        restored = archive.load_coloring(
            original.coloring_id,
            graph,
        )

        assert archive.coloring_count() == 1

        assert observed.coloring_id == original.coloring_id

        assert observed.times_seen == 2

        assert restored.coloring.exact_equals(coloring)
