"""Test plotting without displaying interactive windows."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ramsey.RColoring import RColoring
from ramsey.REnvironment import REnvironment
from ramsey.REnvironmentConfig import (
    REnvironmentConfig,
)
from ramsey.REnvironmentMemory import (
    RNullMemory,
)
from ramsey.RGraph import RGraph
from ramsey.RObjective import (
    RMonochromaticObjective,
)
from ramsey.RPlot import (
    plot_clique_histogram,
    plot_coloring_histogram,
    plot_kn_histogram,
    plot_search_scores,
)
from ramsey.RPolicy import RPolicy
from ramsey.RProblem import RProblem
from ramsey.RSearch import RSearch


class RFirstAvailablePolicy(RPolicy):
    @property
    def name(self) -> str:
        return "first-available"

    def select_action(
        self,
        environment: REnvironment,
    ) -> int:
        return int(environment.available_actions()[0])


def make_search_result(
    record_steps: bool,
):
    graph = RGraph(RProblem.r55(n_vertices=10))

    environment = REnvironment(
        graph=graph,
        objective=RMonochromaticObjective(),
        memory=RNullMemory(),
        config=REnvironmentConfig(max_steps=3),
    )

    search = RSearch(
        environment,
        RFirstAvailablePolicy(),
    )

    coloring = RColoring(
        graph,
        np.zeros(
            graph.number_of_edges,
            dtype=np.uint8,
        ),
    )

    return search.run(
        coloring,
        record_steps=record_steps,
    )


def test_plot_clique_histogram_draws_bars_and_counts() -> None:
    histogram = np.asarray(
        [
            2,
            10,
            20,
            30,
            40,
            50,
            40,
            30,
            20,
            10,
            1,
        ],
        dtype=np.int64,
    )

    figure, axes = plot_clique_histogram(
        histogram,
        clique_size=5,
    )

    assert len(axes.patches) == 11
    assert len(axes.texts) == 11
    assert axes.get_yscale() == "log"

    assert "3 monochromatic" in axes.get_title()

    plt.close(figure)


def test_original_histogram_name_is_preserved() -> None:
    histogram = np.ones(
        11,
        dtype=np.int64,
    )

    figure, axes = plot_kn_histogram(
        histogram,
        k_size=5,
        log_scale=False,
    )

    assert len(axes.patches) == 11

    assert axes.get_yscale() == "linear"

    plt.close(figure)


@pytest.mark.parametrize(
    "histogram",
    [
        np.ones(
            10,
            dtype=np.int64,
        ),
        np.asarray(
            [
                1,
                -1,
                *([1] * 9),
            ],
            dtype=np.int64,
        ),
    ],
)
def test_plot_clique_histogram_validates_input(
    histogram,
) -> None:
    with pytest.raises(ValueError):
        plot_clique_histogram(
            histogram,
            clique_size=5,
        )


def test_plot_coloring_histogram_uses_coloring_problem() -> None:
    graph = RGraph(RProblem.r55(n_vertices=10))

    coloring = RColoring(
        graph,
        np.zeros(
            graph.number_of_edges,
            dtype=np.uint8,
        ),
    )

    figure, axes = plot_coloring_histogram(coloring)

    assert len(axes.patches) == 11

    assert "252" in axes.get_title()

    plt.close(figure)


def test_plot_search_scores_uses_recorded_trajectory() -> None:
    result = make_search_result(record_steps=True)

    figure, axes = plot_search_scores(result)

    assert len(axes.lines) == 2

    assert np.array_equal(
        axes.lines[0].get_xdata(),
        np.arange(
            0,
            result.steps_completed + 1,
        ),
    )

    assert axes.lines[0].get_ydata()[0] == result.initial_score

    assert axes.lines[1].get_ydata()[-1] == result.best_score

    plt.close(figure)


def test_plot_search_scores_requires_step_records() -> None:
    result = make_search_result(record_steps=False)

    with pytest.raises(
        ValueError,
        match="record_steps=True",
    ):
        plot_search_scores(result)
