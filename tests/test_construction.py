"""Test interchangeable seed-coloring constructions."""

import numpy as np
import pytest

from ramsey.RColoring import RColoring
from ramsey.RConstruction import (
    RCyclicConstruction,
    RFixedConstruction,
    RRandomConstruction,
)
from ramsey.RGraph import RGraph
from ramsey.RProblem import RProblem
from ramsey.RScoring import (
    binary_histogram,
    score_coloring,
)

EXOO_HISTOGRAM = np.asarray(
    [
        43,
        8_815,
        43_516,
        130_161,
        239_467,
        253_055,
        175_225,
        80_969,
        25_929,
        5_418,
        0,
    ],
    dtype=np.int64,
)


def test_random_construction_is_reproducible() -> None:
    graph = RGraph(RProblem.r55(n_vertices=10))

    first = RRandomConstruction(np.random.default_rng(201))

    second = RRandomConstruction(np.random.default_rng(201))

    first_coloring = first.construct(graph)

    second_coloring = second.construct(graph)

    assert first_coloring.exact_equals(second_coloring)

    assert first_coloring.colors.shape == (graph.number_of_edges,)


def test_cyclic_construction_uses_circular_distance() -> None:
    graph = RGraph(RProblem.r55(n_vertices=7))

    construction = RCyclicConstruction((0, 1, 0))

    coloring = construction.construct(graph)

    edge_lookup = {
        tuple(map(int, endpoints)): edge for edge, endpoints in enumerate(graph.edges)
    }

    assert coloring.color_of_edge(edge_lookup[(0, 1)]) == 0

    assert coloring.color_of_edge(edge_lookup[(0, 2)]) == 1

    assert coloring.color_of_edge(edge_lookup[(0, 3)]) == 0

    # Distance six in K7 wraps to circular distance one.
    assert coloring.color_of_edge(edge_lookup[(0, 6)]) == 0

    # Difference five wraps to circular distance two.
    assert coloring.color_of_edge(edge_lookup[(1, 6)]) == 1


def test_cyclic_construction_validates_distance_count() -> None:
    graph = RGraph(RProblem.r55(n_vertices=10))

    construction = RCyclicConstruction((0, 1))

    with pytest.raises(
        ValueError,
        match="circular edge distances",
    ):
        construction.construct(graph)


def test_exoo_construction_matches_golden_histogram() -> None:
    graph = RGraph(RProblem.r55())

    coloring = RCyclicConstruction.exoo().construct(graph)

    assert np.array_equal(
        binary_histogram(coloring),
        EXOO_HISTOGRAM,
    )

    assert score_coloring(coloring) == 43


def test_fixed_construction_rebinds_equivalent_graph() -> None:
    first_graph = RGraph(RProblem.r55(n_vertices=10))

    second_graph = RGraph(RProblem.r55(n_vertices=10))

    colors = np.random.default_rng(202).integers(
        0,
        2,
        size=first_graph.number_of_edges,
        dtype=np.uint8,
    )

    original = RColoring(
        first_graph,
        colors,
    )

    restored = RFixedConstruction(original).construct(second_graph)

    assert restored.graph is second_graph

    assert restored.exact_equals(original)


def test_fixed_construction_rejects_different_problem() -> None:
    first_graph = RGraph(RProblem.r55(n_vertices=10))

    second_graph = RGraph(RProblem.r55(n_vertices=11))

    coloring = RColoring(
        first_graph,
        np.zeros(
            first_graph.number_of_edges,
            dtype=np.uint8,
        ),
    )

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        RFixedConstruction(coloring).construct(second_graph)
"""Test interchangeable seed-coloring constructions."""

import numpy as np
import pytest

from ramsey.RArchive import RSQLiteArchive
from ramsey.RColoring import RColoring
from ramsey.RConstruction import (
    RArchiveConstruction,
    RCyclicConstruction,
    RFixedConstruction,
    RMixedConstruction,
    RRandomConstruction,
)
from ramsey.RGraph import RGraph
from ramsey.RProblem import RProblem
from ramsey.RScoring import (
    binary_histogram,
    score_coloring,
)

EXOO_HISTOGRAM = np.asarray(
    [
        43,
        8_815,
        43_516,
        130_161,
        239_467,
        253_055,
        175_225,
        80_969,
        25_929,
        5_418,
        0,
    ],
    dtype=np.int64,
)


def test_random_construction_is_reproducible() -> None:
    graph = RGraph(RProblem.r55(n_vertices=10))

    first = RRandomConstruction(np.random.default_rng(201))

    second = RRandomConstruction(np.random.default_rng(201))

    first_coloring = first.construct(graph)

    second_coloring = second.construct(graph)

    assert first_coloring.exact_equals(second_coloring)

    assert first_coloring.colors.shape == (graph.number_of_edges,)


def test_cyclic_construction_uses_circular_distance() -> None:
    graph = RGraph(RProblem.r55(n_vertices=7))

    construction = RCyclicConstruction((0, 1, 0))

    coloring = construction.construct(graph)

    edge_lookup = {
        tuple(map(int, endpoints)): edge for edge, endpoints in enumerate(graph.edges)
    }

    assert coloring.color_of_edge(edge_lookup[(0, 1)]) == 0

    assert coloring.color_of_edge(edge_lookup[(0, 2)]) == 1

    assert coloring.color_of_edge(edge_lookup[(0, 3)]) == 0

    # Distance six in K7 wraps to circular distance one.
    assert coloring.color_of_edge(edge_lookup[(0, 6)]) == 0

    # Difference five wraps to circular distance two.
    assert coloring.color_of_edge(edge_lookup[(1, 6)]) == 1


def test_cyclic_construction_validates_distance_count() -> None:
    graph = RGraph(RProblem.r55(n_vertices=10))

    construction = RCyclicConstruction((0, 1))

    with pytest.raises(
        ValueError,
        match="circular edge distances",
    ):
        construction.construct(graph)


def test_exoo_construction_matches_golden_histogram() -> None:
    graph = RGraph(RProblem.r55())

    coloring = RCyclicConstruction.exoo().construct(graph)

    assert np.array_equal(
        binary_histogram(coloring),
        EXOO_HISTOGRAM,
    )

    assert score_coloring(coloring) == 43


def test_fixed_construction_rebinds_equivalent_graph() -> None:
    first_graph = RGraph(RProblem.r55(n_vertices=10))

    second_graph = RGraph(RProblem.r55(n_vertices=10))

    colors = np.random.default_rng(202).integers(
        0,
        2,
        size=first_graph.number_of_edges,
        dtype=np.uint8,
    )

    original = RColoring(
        first_graph,
        colors,
    )

    restored = RFixedConstruction(original).construct(second_graph)

    assert restored.graph is second_graph

    assert restored.exact_equals(original)


def test_fixed_construction_rejects_different_problem() -> None:
    first_graph = RGraph(RProblem.r55(n_vertices=10))

    second_graph = RGraph(RProblem.r55(n_vertices=11))

    coloring = RColoring(
        first_graph,
        np.zeros(
            first_graph.number_of_edges,
            dtype=np.uint8,
        ),
    )

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        RFixedConstruction(coloring).construct(second_graph)


def test_archive_construction_samples_only_requested_scores(
    tmp_path,
) -> None:
    graph = RGraph(RProblem.r55(n_vertices=5))

    red = RColoring(
        graph,
        np.zeros(
            graph.number_of_edges,
            dtype=np.uint8,
        ),
    )

    mixed_colors = red.colors.copy()

    mixed_colors[0] = 1

    mixed = RColoring(
        graph,
        mixed_colors,
    )

    with RSQLiteArchive(tmp_path / "seeds.sqlite3") as archive:
        archive.save_coloring(
            red,
            run_name="red",
            iteration=0,
        )

        mixed_record = archive.save_coloring(
            mixed,
            run_name="mixed",
            iteration=1,
        )

        construction = RArchiveConstruction(
            archive=archive,
            rng=np.random.default_rng(400),
            maximum_score=0,
        )

        restored = construction.construct(graph)

        assert restored.exact_equals(mixed)

        assert construction.last_record == mixed_record

        assert construction.last_source_name == (
            "archive-score-any-to-0"
        )


def test_archive_construction_rejects_an_empty_pool(
    tmp_path,
) -> None:
    graph = RGraph(RProblem.r55(n_vertices=5))

    with RSQLiteArchive(tmp_path / "empty.sqlite3") as archive:
        construction = RArchiveConstruction(
            archive=archive,
            rng=np.random.default_rng(401),
            maximum_score=0,
        )

        with pytest.raises(
            RuntimeError,
            match="no colorings",
        ):
            construction.construct(graph)


def test_mixed_construction_reports_selected_source() -> None:
    graph = RGraph(RProblem.r55(n_vertices=5))

    colors = np.zeros(
        graph.number_of_edges,
        dtype=np.uint8,
    )

    fixed = RFixedConstruction(
        RColoring(
            graph,
            colors,
        ),
        construction_name="known-seed",
    )

    random = RRandomConstruction(
        np.random.default_rng(402)
    )

    construction = RMixedConstruction(
        constructions=(
            fixed,
            random,
        ),
        probabilities=(
            1.0,
            0.0,
        ),
        rng=np.random.default_rng(403),
        construction_name="stage-two",
    )

    coloring = construction.construct(graph)

    assert coloring.exact_equals(fixed.coloring)

    assert construction.name == "stage-two"

    assert construction.last_source_name == "known-seed"


@pytest.mark.parametrize(
    (
        "probabilities",
        "exception_type",
    ),
    [
        (
            (),
            ValueError,
        ),
        (
            (1.0,),
            ValueError,
        ),
        (
            (0.4, 0.4),
            ValueError,
        ),
        (
            (1.1, -0.1),
            ValueError,
        ),
        (
            (float("nan"), 0.0),
            ValueError,
        ),
        (
            ("one", 0.0),
            TypeError,
        ),
    ],
)
def test_mixed_construction_validates_probabilities(
    probabilities,
    exception_type,
) -> None:
    graph = RGraph(RProblem.r55(n_vertices=5))

    coloring = RColoring(
        graph,
        np.zeros(
            graph.number_of_edges,
            dtype=np.uint8,
        ),
    )

    fixed = RFixedConstruction(coloring)

    with pytest.raises(exception_type):
        RMixedConstruction(
            constructions=(
                fixed,
                fixed,
            ),
            probabilities=probabilities,
            rng=np.random.default_rng(404),
        )