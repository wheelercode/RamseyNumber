"""Test the mathematical Ramsey problem specification."""

from math import comb

import pytest

from ramsey.RProblem import RProblem


def test_r55_problem_derives_all_core_counts() -> None:
    problem = RProblem.r55()

    assert problem.n_vertices == 43

    assert problem.forbidden_clique_sizes == (5, 5)

    assert problem.n_colors == 2
    assert problem.edge_count == 903

    assert problem.required_clique_sizes == (5,)

    assert problem.is_symmetric

    assert problem.clique_count(5) == 962_598

    assert problem.edges_per_clique(5) == 10

    assert problem.cliques_per_edge(5) == 10_660


def test_asymmetric_problem_preserves_color_order() -> None:
    problem = RProblem(
        n_vertices=20,
        forbidden_clique_sizes=(
            4,
            6,
        ),
    )

    assert problem.n_colors == 2

    assert problem.forbidden_clique_size(0) == 4

    assert problem.forbidden_clique_size(1) == 6

    assert problem.required_clique_sizes == (4, 6)

    assert not problem.is_symmetric

    assert problem.edge_count == comb(
        20,
        2,
    )


def test_problem_normalizes_clique_sizes_to_tuple() -> None:
    problem = RProblem(
        n_vertices=10,
        forbidden_clique_sizes=[
            5,
            5,
        ],
    )

    assert problem.forbidden_clique_sizes == (5, 5)


@pytest.mark.parametrize(
    (
        "n_vertices",
        "clique_sizes",
    ),
    [
        (
            1,
            (2, 2),
        ),
        (
            10,
            (5,),
        ),
        (
            10,
            (1, 5),
        ),
        (
            10,
            (5, 11),
        ),
    ],
)
def test_problem_rejects_invalid_definitions(
    n_vertices: int,
    clique_sizes: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError):
        RProblem(
            n_vertices=n_vertices,
            forbidden_clique_sizes=clique_sizes,
        )


def test_problem_rejects_invalid_color_index() -> None:
    problem = RProblem.r55()

    with pytest.raises(
        IndexError,
        match="color index",
    ):
        problem.forbidden_clique_size(2)


def test_problem_rejects_noninteger_clique_size() -> None:
    with pytest.raises(
        TypeError,
        match="must be an integer",
    ):
        RProblem(
            n_vertices=10,
            forbidden_clique_sizes=(
                5,
                5.5,
            ),
        )
