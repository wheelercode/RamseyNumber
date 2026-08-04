"""Exact coloring and metadata persistence."""

"""Characterize exact coloring and metadata persistence."""

import numpy as np

import RamseyGraph as graph
import RamseyScoring as scoring
from RamseyDatabase import ColoringDatabase


def test_database_round_trip_and_duplicate_observation(
    tmp_path,
    k10_k5_data,
) -> None:
    coloring = graph.random_coloring(
        len(k10_k5_data.edges),
        np.random.default_rng(404),
    )

    histogram = scoring.kn_histogram(
        coloring,
        k10_k5_data.kn_edges,
    )

    database_path = tmp_path / "colorings.sqlite3"

    with ColoringDatabase(database_path) as database:
        first = database.save_coloring(
            coloring,
            histogram,
            n_vertices=10,
            k_size=5,
            run_name="characterization",
            iteration=1,
        )

        second = database.save_coloring(
            coloring,
            histogram,
            n_vertices=10,
            k_size=5,
            run_name="characterization",
            iteration=2,
        )

        (
            restored_coloring,
            restored_histogram,
        ) = database.load_coloring(first.coloring_id)

        expected_score = int(histogram[0] + histogram[-1])

        assert database.coloring_count() == 1

        assert database.best_score() == expected_score

        assert first.coloring_id == second.coloring_id

        assert second.times_seen == 2

        assert np.array_equal(
            restored_coloring,
            coloring,
        )

        assert np.array_equal(
            restored_histogram,
            histogram,
        )
