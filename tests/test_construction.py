"""Seed constructors and known constructions."""

"""Characterize known seed constructions."""

import numpy as np

import RamseyScoring as scoring
from Exoo import exoo_cyclic_coloring

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


def test_exoo_cyclic_coloring_matches_golden_histogram(
    r55_data,
) -> None:
    coloring = exoo_cyclic_coloring(
        r55_data.edges,
        n_vertices=43,
    )

    histogram = scoring.kn_histogram(
        coloring,
        r55_data.kn_edges,
    )

    assert coloring.shape == (903,)

    assert np.array_equal(
        histogram,
        EXOO_HISTOGRAM,
    )

    assert (
        scoring.score_coloring(
            coloring,
            r55_data.kn_edges,
        )
        == 43
    )
