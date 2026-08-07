"""Visualization of colorings, searches, and experiment results.

Provides matplotlib-based plots for three kinds of data: the color-one
edge-count histogram of one coloring's cliques (how close cliques are to
monochromatic, and how many actually are), the score trajectory of one
search run, and the initial/final/best score trajectory across the
iterations of one experiment. Every plotting function returns the
``(Figure, Axes)`` pair it drew on, and accepts an optional existing
``Axes`` so plots can be composed into larger figures.
"""

from __future__ import annotations

from math import comb

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from numpy.typing import NDArray

from .RColoring import RColoring
from .RExperiment import RExperimentResult
from .RScoring import binary_histogram
from .RSearch import RSearchResult


def plot_clique_histogram(
    histogram: NDArray[np.integer],
    clique_size: int,
    *,
    ax: Axes | None = None,
    log_scale: bool = True,
    title: str | None = None,
    minimum_inside_height_pixels: float = 18.0,
) -> tuple[
    plt.Figure,
    Axes,
]:
    """
    Plot a binary clique histogram with exact count annotations.

    Draws one bar per possible color-one edge count (0 through
    ``edges_per_clique``), with bar height equal to the number of cliques
    of ``clique_size`` vertices having that many color-one edges. Bars
    are colored along a diverging colormap from one end of the axis to
    the other, so the two monochromatic extremes (all edges color zero,
    at the left, and all edges color one, at the right) sit visually
    opposite each other; those two bars additionally get a thicker,
    color-coded outline to call out the exact violation counts they
    represent. Counts appear inside sufficiently tall bars. Counts
    belonging to short or zero-height bars appear above the bar so they
    remain visible.

    Args:
        histogram (NDArray[np.integer]): Clique count by color-one edge
            count, shape ``(comb(clique_size, 2) + 1,)``.
        clique_size (int): Number of vertices per clique (e.g. 5 for K5),
            used to compute the expected number of histogram bins and
            axis labels.
        ax (Axes | None): Existing axes to draw on. If ``None``, a new
            figure and axes are created.
        log_scale (bool): If ``True`` (the default), use a logarithmic
            y-axis, with the lower limit set just below the smallest
            positive count so zero-height bars remain distinguishable
            from small ones.
        title (str | None): Plot title. If ``None``, a title reporting
            the total monochromatic count is generated.
        minimum_inside_height_pixels (float): Minimum rendered bar height,
            in pixels, for a count label to be placed inside the bar
            rather than above it. Defaults to 18.0.

    Returns:
        tuple[plt.Figure, Axes]: The figure and axes the histogram was
        drawn on.

    Raises:
        ValueError: If ``histogram`` does not have the shape implied by
            ``clique_size``, if it contains negative counts, or if
            ``minimum_inside_height_pixels`` is negative.
    """
    edges_per_clique = comb(
        clique_size,
        2,
    )

    expected_bins = edges_per_clique + 1

    histogram = np.asarray(histogram)

    if histogram.shape != (expected_bins,):
        raise ValueError(
            f"K_{clique_size} has "
            f"{edges_per_clique} edges, so the "
            "histogram must have shape "
            f"({expected_bins},). "
            f"Received {histogram.shape}."
        )

    if np.any(histogram < 0):
        raise ValueError("Histogram counts cannot be negative.")

    if minimum_inside_height_pixels < 0:
        raise ValueError("minimum_inside_height_pixels " "cannot be negative.")

    if ax is None:
        figure, ax = plt.subplots(
            figsize=(10, 5.5),
            constrained_layout=True,
        )
    else:
        figure = ax.figure

    color_one_counts = np.arange(expected_bins)

    colors = plt.colormaps["coolwarm_r"](
        np.linspace(
            0.05,
            0.95,
            expected_bins,
        )
    )

    bars = ax.bar(
        color_one_counts,
        histogram,
        color=colors,
        edgecolor="#303030",
        linewidth=0.7,
        width=0.82,
    )

    # Emphasize the two monochromatic categories.
    bars[0].set_linewidth(2.5)
    bars[-1].set_linewidth(2.5)

    bars[0].set_edgecolor("darkred")

    bars[-1].set_edgecolor("darkblue")

    ax.set_xlabel("Number of color-one edges " f"in each $K_{clique_size}$")

    ax.set_ylabel("Number of subgraphs")

    if title is None:
        monochromatic = int(histogram[0] + histogram[-1])

        title = (
            f"$K_{clique_size}$ "
            "color distribution "
            f"({monochromatic:,} "
            "monochromatic)"
        )

    ax.set_title(title)
    ax.set_xticks(color_one_counts)

    tick_labels = [str(count) for count in color_one_counts]

    tick_labels[0] = "0\nall color zero"

    tick_labels[-1] = f"{edges_per_clique}\n" "all color one"

    ax.set_xticklabels(tick_labels)

    if log_scale:
        ax.set_yscale("log")

        positive_counts = histogram[histogram > 0]

        if positive_counts.size > 0:
            ax.set_ylim(
                bottom=max(
                    0.8,
                    float(positive_counts.min()) * 0.75,
                )
            )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.3,
    )

    ax.set_axisbelow(True)

    # Render once so data coordinates can be measured in pixels.
    figure.canvas.draw()

    axis_bottom = float(ax.get_ylim()[0])

    for bar, count_value in zip(
        bars,
        histogram,
    ):
        count = int(count_value)

        center_x = bar.get_x() + bar.get_width() / 2

        if count > 0:
            visible_top = float(count)

            bottom_pixels = ax.transData.transform(
                (
                    center_x,
                    axis_bottom,
                )
            )[1]

            top_pixels = ax.transData.transform(
                (
                    center_x,
                    visible_top,
                )
            )[1]

            rendered_height = top_pixels - bottom_pixels
        else:
            visible_top = axis_bottom
            rendered_height = 0.0

        place_inside = count > 0 and rendered_height >= minimum_inside_height_pixels

        if place_inside:
            (
                red,
                green,
                blue,
                _,
            ) = bar.get_facecolor()

            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue

            text_color = "black" if luminance > 0.58 else "white"

            vertical_alignment = "top"
            vertical_offset = -3
            label_background = None

        else:
            text_color = "black"
            vertical_alignment = "bottom"
            vertical_offset = 3

            label_background = {
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
                "pad": 0.5,
            }

        ax.annotate(
            f"{count:,}",
            xy=(
                center_x,
                visible_top,
            ),
            xytext=(
                0,
                vertical_offset,
            ),
            textcoords="offset points",
            ha="center",
            va=vertical_alignment,
            fontsize=8,
            fontweight="semibold",
            color=text_color,
            bbox=label_background,
            clip_on=False,
        )

    return figure, ax


def plot_coloring_histogram(
    coloring: RColoring,
    **plot_options,
) -> tuple[
    plt.Figure,
    Axes,
]:
    """
    Calculate and plot a symmetric binary coloring histogram.

    Computes the color-one edge-count histogram for ``coloring`` via
    :func:`ramsey.RScoring.binary_histogram` and renders it with
    :func:`plot_clique_histogram`.

    Args:
        coloring (RColoring): Coloring to summarize. Its problem must use
            exactly two colors and be symmetric.
        **plot_options: Forwarded to :func:`plot_clique_histogram` (e.g.
            ``ax``, ``log_scale``, ``title``).

    Returns:
        tuple[plt.Figure, Axes]: The figure and axes the histogram was
        drawn on.

    Raises:
        ValueError: If ``coloring``'s problem is not a symmetric
            two-color problem.
    """
    problem = coloring.graph.problem

    if problem.n_colors != 2 or not problem.is_symmetric:
        raise ValueError(
            "plot_coloring_histogram requires " "a symmetric two-color problem."
        )

    return plot_clique_histogram(
        binary_histogram(coloring),
        problem.required_clique_sizes[0],
        **plot_options,
    )


def plot_search_scores(
    result: RSearchResult,
    *,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[
    plt.Figure,
    Axes,
]:
    """
    Plot current and episode-best scores across a search.

    Draws two lines against environment step number: the current
    monochromatic score at each recorded step (which may rise and fall
    as the policy explores), and the running best score seen so far
    (monotonically non-increasing). Both series start from
    ``result.initial_score`` at step 0.

    Args:
        result (RSearchResult): Search result to plot. Must have been run
            with ``record_steps=True`` if any steps were completed.
        ax (Axes | None): Existing axes to draw on. If ``None``, a new
            figure and axes are created.
        title (str | None): Plot title. If ``None``, defaults to
            ``f"{result.policy_name} search"``.

    Returns:
        tuple[plt.Figure, Axes]: The figure and axes the scores were
        drawn on.

    Raises:
        ValueError: If ``result.steps_completed`` is positive but
            ``result.step_results`` is empty (the search was not run
            with ``record_steps=True``).
    """
    if result.steps_completed > 0 and not result.step_results:
        raise ValueError(
            "Search result has no step records. " "Run with record_steps=True."
        )

    if ax is None:
        figure, ax = plt.subplots(
            figsize=(10, 5.5),
            constrained_layout=True,
        )
    else:
        figure = ax.figure

    steps = np.asarray(
        [
            0,
            *(step.step_number for step in result.step_results),
        ],
        dtype=np.int32,
    )

    scores = np.asarray(
        [
            result.initial_score,
            *(step.score for step in result.step_results),
        ],
        dtype=np.int64,
    )

    best_scores = np.asarray(
        [
            result.initial_score,
            *(step.best_score for step in result.step_results),
        ],
        dtype=np.int64,
    )

    ax.plot(
        steps,
        scores,
        label="current score",
        linewidth=1.4,
        alpha=0.8,
    )

    ax.plot(
        steps,
        best_scores,
        label="best score",
        linewidth=2.2,
    )

    ax.set_xlabel("Environment step")

    ax.set_ylabel("Monochromatic score")

    ax.set_title(title or f"{result.policy_name} search")

    ax.grid(
        linestyle="--",
        alpha=0.3,
    )

    ax.legend()

    return figure, ax


def plot_experiment_scores(
    result: RExperimentResult,
    *,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[
    plt.Figure,
    Axes,
]:
    """
    Plot initial, final, and best scores by experiment iteration.

    Draws three lines against experiment iteration number: each
    iteration's starting score, its ending score, and the best score
    reached during that iteration's search. Useful for comparing outcomes
    across the independent restarts or reconfigurations that make up one
    experiment.

    Args:
        result (RExperimentResult): Experiment result to plot.
        ax (Axes | None): Existing axes to draw on. If ``None``, a new
            figure and axes are created.
        title (str | None): Plot title. If ``None``, defaults to
            ``result.run_name``.

    Returns:
        tuple[plt.Figure, Axes]: The figure and axes the scores were
        drawn on.
    """
    if ax is None:
        figure, ax = plt.subplots(
            figsize=(10, 5.5),
            constrained_layout=True,
        )
    else:
        figure = ax.figure

    iteration_numbers = np.asarray(
        [item.iteration for item in result.iteration_results],
        dtype=np.int64,
    )

    initial_scores = np.asarray(
        [item.search_result.initial_score for item in result.iteration_results],
        dtype=np.int64,
    )

    final_scores = np.asarray(
        [item.search_result.final_score for item in result.iteration_results],
        dtype=np.int64,
    )

    best_scores = np.asarray(
        [item.search_result.best_score for item in result.iteration_results],
        dtype=np.int64,
    )

    ax.plot(
        iteration_numbers,
        initial_scores,
        label="initial",
        alpha=0.65,
    )

    ax.plot(
        iteration_numbers,
        final_scores,
        label="final",
        alpha=0.8,
    )

    ax.plot(
        iteration_numbers,
        best_scores,
        label="best",
        linewidth=2.2,
    )

    ax.set_xlabel("Experiment iteration")

    ax.set_ylabel("Monochromatic score")

    ax.set_title(title or result.run_name)

    ax.grid(
        linestyle="--",
        alpha=0.3,
    )

    ax.legend()

    return figure, ax


def plot_kn_histogram(
    histogram: NDArray[np.integer],
    k_size: int,
    **plot_options,
) -> tuple[
    plt.Figure,
    Axes,
]:
    """
    Compatibility alias for the original histogram function name.

    Args:
        histogram (NDArray[np.integer]): Clique count by color-one edge
            count, forwarded to :func:`plot_clique_histogram`.
        k_size (int): Number of vertices per clique, forwarded to
            :func:`plot_clique_histogram` as ``clique_size``.
        **plot_options: Forwarded to :func:`plot_clique_histogram`.

    Returns:
        tuple[plt.Figure, Axes]: The figure and axes the histogram was
        drawn on.
    """
    return plot_clique_histogram(
        histogram,
        clique_size=k_size,
        **plot_options,
    )
