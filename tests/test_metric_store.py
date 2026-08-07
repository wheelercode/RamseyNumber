"""Tests for persistent Ramsey metric snapshots."""

import numpy as np

from ramsey.RArchive import RSQLiteArchive
from ramsey.RColoring import RColoring
from ramsey.RGraph import RGraph
from ramsey.RMetric import calculate_metrics
from ramsey.RMetricStore import RMetricStore
from ramsey.RProblem import RProblem
from ramsey.RState import RSearchState


def test_metric_store_round_trip(tmp_path) -> None:
    graph = RGraph(
        RProblem.r55(
            n_vertices=6,
        )
    )
    coloring = RColoring(
        graph,
        np.zeros(
            graph.number_of_edges,
            dtype=np.uint8,
        ),
    )
    database_path = tmp_path / "metrics.sqlite3"

    with RSQLiteArchive(database_path) as archive:
        record = archive.save_coloring(
            coloring,
            run_name="metric-test",
            iteration=0,
        )

    snapshot = calculate_metrics(
        RSearchState(coloring)
    )

    with RMetricStore(database_path) as store:
        assert not store.contains(record.coloring_id)

        stored = store.save(
            record.coloring_id,
            snapshot,
        )

        assert store.contains(record.coloring_id)
        assert store.count() == 1
        assert stored.snapshot == snapshot
        assert store.load(record.coloring_id).snapshot == snapshot