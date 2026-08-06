"""Test cyclic live-archive seed construction."""

import numpy as np

from ramsey.RArchive import RSQLiteArchive
from ramsey.RColoring import RColoring
from ramsey.RConstruction import (
    RArchiveCycleConstruction,
)
from ramsey.RGraph import RGraph
from ramsey.RProblem import RProblem


def test_archive_cycle_refreshes_only_after_exhaustion(
    tmp_path,
) -> None:
    graph = RGraph(RProblem.r55(n_vertices=5))

    with RSQLiteArchive(tmp_path / "cycle.sqlite3") as archive:
        initial_records = []

        for edge in range(3):
            colors = np.zeros(
                graph.number_of_edges,
                dtype=np.uint8,
            )

            colors[: edge + 1] = 1

            initial_records.append(
                archive.save_coloring(
                    RColoring(graph, colors),
                    run_name="cycle-source",
                    iteration=edge,
                )
            )

        construction = RArchiveCycleConstruction(
            archive=archive,
            rng=np.random.default_rng(405),
        )

        first_cycle_ids = []

        construction.construct(graph)

        assert construction.last_record is not None

        first_cycle_ids.append(
            construction.last_record.coloring_id
        )

        assert construction.cycle_number == 1
        assert construction.current_cycle_size == 3

        # Add a descendant while cycle 1 is already in progress.
        # It must not disturb that frozen cycle.
        later_colors = np.ones(
            graph.number_of_edges,
            dtype=np.uint8,
        )

        later = archive.save_coloring(
            RColoring(graph, later_colors),
            run_name="cycle-descendant",
            iteration=3,
        )

        while construction.remaining_count:
            construction.construct(graph)

            assert construction.last_record is not None

            first_cycle_ids.append(
                construction.last_record.coloring_id
            )

        assert set(first_cycle_ids) == {
            record.coloring_id
            for record in initial_records
        }

        # Cycle 2 refreshes from SQLite and must now contain the
        # descendant that was created during cycle 1.
        second_cycle_ids = []

        for _ in range(4):
            construction.construct(graph)

            assert construction.last_record is not None

            second_cycle_ids.append(
                construction.last_record.coloring_id
            )

        assert construction.cycle_number == 2
        assert construction.current_cycle_size == 4
        assert later.coloring_id in second_cycle_ids


def test_archive_cycle_limit_selects_best_live_pool(
    tmp_path,
) -> None:
    graph = RGraph(RProblem.r55(n_vertices=5))

    with RSQLiteArchive(tmp_path / "cycle-limit.sqlite3") as archive:
        for edge_count in range(4):
            colors = np.zeros(
                graph.number_of_edges,
                dtype=np.uint8,
            )

            colors[:edge_count] = 1

            archive.save_coloring(
                RColoring(graph, colors),
                run_name="cycle-limit-source",
                iteration=edge_count,
            )

        expected_ids = {
            record.coloring_id
            for record in archive.colorings_in_score_range(
                limit=2,
                graph=graph,
            )
        }

        construction = RArchiveCycleConstruction(
            archive=archive,
            rng=np.random.default_rng(406),
            limit=2,
        )

        consumed_ids = set()

        for _ in range(2):
            construction.construct(graph)

            assert construction.last_record is not None

            consumed_ids.add(
                construction.last_record.coloring_id
            )

        assert consumed_ids == expected_ids
        assert construction.remaining_count == 0