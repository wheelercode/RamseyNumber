"""Exact score and histogram calculations."""

"""Characterize exact score and histogram calculations."""

from math import comb

import numpy as np

import RamseyGraph as graph
import RamseyScoring as scoring


def test_all_red_and_all_blue_colorings_are_fully_monochromatic(
    k10_k5_data,
) -> None:
    expected_cliques = comb(
        10,
        5,
    )

    for color, endpoint in (
        (0, 0),
        (1, 10),
    ):
        coloring = np.full(
            len(k10_k5_data.edges),
            color,
            dtype=np.uint8,
        )

        histogram = scoring.kn_histogram(
            coloring,
            k10_k5_data.kn_edges,
        )

        assert histogram.sum() == expected_cliques

        assert histogram[endpoint] == expected_cliques

        assert (
            scoring.score_coloring(
                coloring,
                k10_k5_data.kn_edges,
            )
            == expected_cliques
        )


def test_random_histogram_accounts_for_every_k5(
    k10_k5_data,
) -> None:
    coloring = graph.random_coloring(
        len(k10_k5_data.edges),
        np.random.default_rng(2026),
    )

    histogram = scoring.kn_histogram(
        coloring,
        k10_k5_data.kn_edges,
    )

    score = scoring.score_coloring(
        coloring,
        k10_k5_data.kn_edges,
    )

    assert histogram.shape == (11,)

    assert histogram.sum() == comb(
        10,
        5,
    )

    assert score == int(histogram[0] + histogram[10])


def test_color_complement_preserves_symmetric_r55_score(
    k10_k5_data,
) -> None:
    coloring = graph.random_coloring(
        len(k10_k5_data.edges),
        np.random.default_rng(88),
    )

    complement = np.uint8(1) - coloring

    original_histogram = scoring.kn_histogram(
        coloring,
        k10_k5_data.kn_edges,
    )

    complement_histogram = scoring.kn_histogram(
        complement,
        k10_k5_data.kn_edges,
    )

    assert np.array_equal(
        complement_histogram,
        original_histogram[::-1],
    )
