"""Tests for structural Ramsey metric snapshots."""

import numpy as np

from ramsey.RColoring import RColoring
from ramsey.RGraph import RGraph
from ramsey.RMetric import RMetricSnapshot, calculate_metrics
from ramsey.RProblem import RProblem
from ramsey.RState import RSearchState


def all_red_state(n_vertices: int) -> RSearchState:
    graph = RGraph(
        RProblem.r55(
            n_vertices=n_vertices,
        )
    )
    coloring = RColoring(
        graph,
        np.zeros(
            graph.number_of_edges,
            dtype=np.uint8,
        ),
    )
    return RSearchState(coloring)


def test_metric_snapshot_keeps_red_and_blue_deficits_separate() -> None:
    state = all_red_state(5)

    metrics = calculate_metrics(state)

    assert metrics.score == 1
    assert metrics.red_violations == 1
    assert metrics.blue_violations == 0

    assert metrics.red_deficit_histogram[0] == 1
    assert metrics.blue_deficit_histogram[10] == 1
    assert sum(metrics.red_deficit_histogram) == 1
    assert sum(metrics.blue_deficit_histogram) == 1


def test_metric_snapshot_records_violation_participation() -> None:
    state = all_red_state(5)

    metrics = calculate_metrics(state)

    assert metrics.red_vertex_violation_load == (1, 1, 1, 1, 1)
    assert metrics.blue_vertex_violation_load == (0, 0, 0, 0, 0)
    assert metrics.red_edge_violation_load == (1,) * 10
    assert metrics.blue_edge_violation_load == (0,) * 10


def test_metric_snapshot_records_exact_overlap_sizes() -> None:
    # K6 contains six K5s. Every distinct pair omits a different vertex,
    # so every pair shares exactly four vertices: C(6, 2) = 15 pairs.
    state = all_red_state(6)

    metrics = calculate_metrics(state)

    assert metrics.score == 6
    assert metrics.red_red_overlap == (0, 0, 0, 0, 15)
    assert metrics.blue_blue_overlap == (0, 0, 0, 0, 0)
    assert metrics.red_blue_overlap == (0, 0, 0, 0, 0)


def test_metric_snapshot_records_complete_single_flip_landscape() -> None:
    state = all_red_state(5)

    metrics = calculate_metrics(state)

    assert metrics.single_flip_rewards == (1,) * 10
    assert metrics.best_single_flip_reward == 1
    assert metrics.improving_single_flips == 10
    assert metrics.neutral_single_flips == 0
    assert metrics.worsening_single_flips == 0


def test_metric_snapshot_round_trip_dict() -> None:
    metrics = calculate_metrics(
        all_red_state(6)
    )

    restored = RMetricSnapshot.from_dict(
        metrics.to_dict()
    )

    assert restored == metrics