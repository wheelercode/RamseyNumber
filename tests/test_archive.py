"""Test persistent coloring storage and bounded archive queries."""

import numpy as np
import pytest

from ramsey.RArchive import RSQLiteArchive
from ramsey.RColoring import RColoring
from ramsey.RGraph import RGraph
from ramsey.RProblem import RProblem


def make_coloring(
    graph: RGraph,
    blue_edges: tuple[int, ...],
) -> RColoring:
    colors = np.zeros(
        graph.number_of_edges,
        dtype=np.uint8,
    )

    colors[list(blue_edges)] = 1

    return RColoring(
        graph,
        colors,
    )


def test_archive_round_trip_and_duplicate_observation(
    tmp_path,
) -> None:
    graph = RGraph(RProblem.r55(n_vertices=5))

    coloring = make_coloring(
        graph,
        (0,),
    )

    with RSQLiteArchive(tmp_path / "round-trip.sqlite3") as archive:
        first = archive.save_coloring(
            coloring,
            run_name="first",
            iteration=1,
        )

        second = archive.save_coloring(
            coloring,
            run_name="second",
            iteration=2,
        )

        restored = archive.load_coloring(
            first.coloring_id,
            graph,
        )

        assert first.coloring_id == second.coloring_id

        assert second.times_seen == 2

        assert second.run_name == "second"

        assert restored.coloring.exact_equals(coloring)

        assert not restored.histogram.flags.writeable


def test_archive_filters_and_counts_inclusive_score_ranges(
    tmp_path,
) -> None:
    graph = RGraph(RProblem.r55(n_vertices=5))

    all_red = make_coloring(
        graph,
        (),
    )

    mixed_zero = make_coloring(
        graph,
        (0,),
    )

    mixed_one = make_coloring(
        graph,
        (1,),
    )

    with RSQLiteArchive(tmp_path / "ranges.sqlite3") as archive:
        archive.save_coloring(
            all_red,
            run_name="range",
            iteration=0,
        )

        zero_record = archive.save_coloring(
            mixed_zero,
            run_name="range",
            iteration=1,
        )

        one_record = archive.save_coloring(
            mixed_one,
            run_name="range",
            iteration=2,
        )

        score_zero = archive.colorings_in_score_range(
            minimum_score=0,
            maximum_score=0,
            graph=graph,
        )

        assert score_zero == [
            zero_record,
            one_record,
        ]

        assert archive.coloring_count_in_score_range(
            maximum_score=0,
            graph=graph,
        ) == 2

        assert archive.coloring_count_in_score_range(
            minimum_score=1,
            graph=graph,
        ) == 1

        assert archive.colorings_in_score_range(
            maximum_score=0,
            limit=1,
            graph=graph,
        ) == [zero_record]


def test_archive_score_ranges_filter_by_problem(
    tmp_path,
) -> None:
    graph5 = RGraph(RProblem.r55(n_vertices=5))

    graph6 = RGraph(RProblem.r55(n_vertices=6))

    with RSQLiteArchive(tmp_path / "problems.sqlite3") as archive:
        archive.save_coloring(
            make_coloring(graph5, (0,)),
            run_name="k5",
            iteration=0,
        )

        archive.save_coloring(
            make_coloring(graph6, (0,)),
            run_name="k6",
            iteration=0,
        )

        assert archive.coloring_count_in_score_range(
            graph=graph5,
        ) == 1

        assert archive.coloring_count_in_score_range(
            graph=graph6,
        ) == 1

        assert archive.coloring_count_in_score_range() == 2


@pytest.mark.parametrize(
    (
        "arguments",
        "exception_type",
    ),
    [
        (
            {"minimum_score": -1},
            ValueError,
        ),
        (
            {"maximum_score": 1.5},
            TypeError,
        ),
        (
            {
                "minimum_score": 2,
                "maximum_score": 1,
            },
            ValueError,
        ),
        (
            {"limit": 0},
            ValueError,
        ),
        (
            {"limit": True},
            TypeError,
        ),
    ],
)
def test_archive_validates_score_range_arguments(
    tmp_path,
    arguments,
    exception_type,
) -> None:
    with RSQLiteArchive(tmp_path / "invalid.sqlite3") as archive:
        with pytest.raises(exception_type):
            archive.colorings_in_score_range(**arguments)


def test_closed_archive_rejects_bounded_queries(
    tmp_path,
) -> None:
    archive = RSQLiteArchive(tmp_path / "closed.sqlite3")

    archive.close()

    with pytest.raises(
        RuntimeError,
        match="closed",
    ):
        archive.coloring_count_in_score_range()