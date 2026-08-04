"""Test null, tabu, and exact visited-state memory."""

import numpy as np
import pytest

from ramsey.RColoring import RColoring
from ramsey.REnvironmentConfig import RTabuMemoryConfig
from ramsey.REnvironmentMemory import (
    RNullMemory,
    RTabuMemory,
)
from ramsey.RGraph import RGraph
from ramsey.RProblem import RProblem
from ramsey.RState import RSearchState


@pytest.fixture(scope="module")
def graph() -> RGraph:
    return RGraph(RProblem.r55(n_vertices=10))


def make_state(
    graph: RGraph,
    seed: int,
) -> RSearchState:
    rng = np.random.default_rng(seed)

    colors = rng.integers(
        0,
        2,
        size=graph.number_of_edges,
        dtype=np.uint8,
    )

    return RSearchState(
        RColoring(
            graph,
            colors,
        )
    )


def test_null_memory_never_blocks(
    graph: RGraph,
) -> None:
    state = make_state(
        graph,
        seed=71,
    )

    memory = RNullMemory()
    memory.reset(state)

    status = memory.status(
        state,
        step_number=0,
    )

    assert not np.any(status.edge_tabu_mask)

    assert not np.any(status.revisit_mask)

    assert not np.any(status.blocked_mask)


def test_tabu_memory_blocks_reverse_action_as_tabu_and_revisit(
    graph: RGraph,
) -> None:
    state = make_state(
        graph,
        seed=72,
    )

    memory = RTabuMemory(
        graph.number_of_edges,
        RTabuMemoryConfig(
            edge_tenure=5,
            visited_state_window=20,
        ),
    )

    memory.reset(state)

    state.apply_edge_flip(0)

    memory.record_transition(
        edge=0,
        state=state,
        step_number=1,
    )

    status = memory.status(
        state,
        step_number=1,
    )

    assert status.edge_tabu_mask[0]
    assert status.revisit_mask[0]
    assert status.blocked_mask[0]

    assert memory.tabu_until[0] == 6
    assert memory.visited_state_count == 2


def test_edge_becomes_available_after_exact_tenure(
    graph: RGraph,
) -> None:
    state = make_state(
        graph,
        seed=73,
    )

    memory = RTabuMemory(
        graph.number_of_edges,
        RTabuMemoryConfig(
            edge_tenure=5,
            visited_state_window=0,
        ),
    )

    memory.reset(state)

    state.apply_edge_flip(3)

    memory.record_transition(
        edge=3,
        state=state,
        step_number=1,
    )

    assert memory.status(
        state,
        step_number=5,
    ).edge_tabu_mask[3]

    assert not memory.status(
        state,
        step_number=6,
    ).edge_tabu_mask[3]


def test_visited_state_queue_is_bounded(
    graph: RGraph,
) -> None:
    state = make_state(
        graph,
        seed=74,
    )

    memory = RTabuMemory(
        graph.number_of_edges,
        RTabuMemoryConfig(
            edge_tenure=0,
            visited_state_window=3,
        ),
    )

    memory.reset(state)

    edges = [0, 1, 2, 3, 4]

    for step_number, edge in enumerate(
        edges,
        start=1,
    ):
        state.apply_edge_flip(edge)

        memory.record_transition(
            edge=edge,
            state=state,
            step_number=step_number,
        )

    assert memory.visited_state_count == 3


def test_memory_masks_are_read_only(
    graph: RGraph,
) -> None:
    state = make_state(
        graph,
        seed=75,
    )

    memory = RTabuMemory(graph.number_of_edges)

    memory.reset(state)

    status = memory.status(
        state,
        step_number=0,
    )

    assert not (status.edge_tabu_mask.flags.writeable)

    assert not (status.revisit_mask.flags.writeable)

    assert not (status.blocked_mask.flags.writeable)
