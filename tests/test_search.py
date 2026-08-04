"""Test complete search lifecycle and immutable results."""

import numpy as np
import pytest

from ramsey.RColoring import RColoring
from ramsey.REnvironment import REnvironment
from ramsey.REnvironmentConfig import (
    REnvironmentConfig,
    RTabuMemoryConfig,
)
from ramsey.REnvironmentMemory import (
    RTabuMemory,
)
from ramsey.RGraph import RGraph
from ramsey.RObjective import (
    RMonochromaticObjective,
)
from ramsey.RPolicy import RPolicy
from ramsey.RProblem import RProblem
from ramsey.RSearch import RSearch


class RFirstAvailablePolicy(RPolicy):
    """
    Deterministic test policy selecting the first available edge.
    """

    @property
    def name(self) -> str:
        return "first-available"

    def select_action(
        self,
        environment: REnvironment,
    ) -> int:
        return int(environment.available_actions()[0])


def make_search(
    n_vertices: int,
    max_steps: int,
) -> tuple[
    RGraph,
    RSearch,
]:
    graph = RGraph(RProblem.r55(n_vertices=n_vertices))

    environment = REnvironment(
        graph=graph,
        objective=RMonochromaticObjective(),
        memory=RTabuMemory(
            graph.number_of_edges,
            RTabuMemoryConfig(
                edge_tenure=2,
                visited_state_window=20,
            ),
        ),
        config=REnvironmentConfig(max_steps=max_steps),
    )

    search = RSearch(
        environment,
        RFirstAvailablePolicy(),
    )

    return graph, search


def test_search_runs_until_environment_truncation() -> None:
    graph, search = make_search(
        n_vertices=10,
        max_steps=5,
    )

    seed = RColoring(
        graph,
        np.zeros(
            graph.number_of_edges,
            dtype=np.uint8,
        ),
    )

    result = search.run(
        seed,
        record_steps=True,
    )

    assert result.steps_completed == 5
    assert result.truncated
    assert not result.terminated

    assert len(result.step_results) == 5

    assert result.policy_name == "first-available"

    assert result.objective_name == "monochromatic"

    assert result.final_score == result.step_results[-1].score

    assert result.best_score <= result.initial_score

    assert result.best_score <= result.final_score

    assert result.final_coloring.exact_equals(
        search.environment.state.coloring_snapshot()
    )


def test_search_can_omit_step_records() -> None:
    graph, search = make_search(
        n_vertices=10,
        max_steps=3,
    )

    seed = RColoring(
        graph,
        np.zeros(
            graph.number_of_edges,
            dtype=np.uint8,
        ),
    )

    result = search.run(seed)

    assert result.steps_completed == 3
    assert result.step_results == ()


def test_score_zero_seed_terminates_without_an_action() -> None:
    graph, search = make_search(
        n_vertices=5,
        max_steps=10,
    )

    colors = np.zeros(
        graph.number_of_edges,
        dtype=np.uint8,
    )

    # A K5 containing both colors is not monochromatic.
    colors[0] = 1

    seed = RColoring(
        graph,
        colors,
    )

    result = search.run(
        seed,
        record_steps=True,
    )

    assert result.initial_score == 0
    assert result.final_score == 0
    assert result.best_score == 0

    assert result.steps_completed == 0
    assert result.terminated
    assert not result.truncated
    assert result.step_results == ()


def test_search_result_reports_score_reductions() -> None:
    graph, search = make_search(
        n_vertices=10,
        max_steps=4,
    )

    seed = RColoring(
        graph,
        np.zeros(
            graph.number_of_edges,
            dtype=np.uint8,
        ),
    )

    result = search.run(seed)

    assert result.score_reduction == (result.initial_score - result.final_score)

    assert result.best_score_reduction == (result.initial_score - result.best_score)


def test_search_validates_record_steps() -> None:
    graph, search = make_search(
        n_vertices=10,
        max_steps=2,
    )

    seed = RColoring(
        graph,
        np.zeros(
            graph.number_of_edges,
            dtype=np.uint8,
        ),
    )

    with pytest.raises(
        TypeError,
        match="record_steps",
    ):
        search.run(
            seed,
            record_steps=1,
        )
