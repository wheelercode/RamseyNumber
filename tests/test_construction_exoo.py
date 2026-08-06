"""Test the published Exoo zero-score K42 construction."""

import numpy as np
import pytest

from ramsey.RConstructionExoo import (
    EXOO_42_BLUE_CORRECTIONS,
    RConstructionExoo,
)
from ramsey.RGraph import RGraph
from ramsey.RProblem import RProblem
from ramsey.RScoring import (
    binary_histogram,
    score_coloring,
)


EXOO_42_HISTOGRAM = np.asarray(
    [
        0,
        5_893,
        30_924,
        98_346,
        192_346,
        225_990,
        174_280,
        88_216,
        28_482,
        6_191,
        0,
    ],
    dtype=np.int64,
)


def test_exoo_k42_has_zero_ramsey_score() -> None:
    graph = RGraph(
        RProblem.r55(
            n_vertices=42,
        )
    )

    coloring = RConstructionExoo().construct(
        graph
    )

    assert score_coloring(coloring) == 0

    assert np.array_equal(
        binary_histogram(coloring),
        EXOO_42_HISTOGRAM,
    )


def test_exoo_k42_has_published_color_corrections() -> None:
    assert len(EXOO_42_BLUE_CORRECTIONS) == 16
    assert len(set(EXOO_42_BLUE_CORRECTIONS)) == 16


def test_exoo_k42_has_expected_global_color_counts() -> None:
    graph = RGraph(
        RProblem.r55(
            n_vertices=42,
        )
    )

    coloring = RConstructionExoo().construct(
        graph
    )

    blue_edges = int(
        np.count_nonzero(
            coloring.colors == 1
        )
    )

    red_edges = int(
        np.count_nonzero(
            coloring.colors == 0
        )
    )

    assert blue_edges == 426
    assert red_edges == 435


def test_exoo_k42_rejects_other_graph_sizes() -> None:
    graph = RGraph(
        RProblem.r55(
            n_vertices=43,
        )
    )

    with pytest.raises(
        ValueError,
        match="exactly 42 vertices",
    ):
        RConstructionExoo().construct(
            graph
        )