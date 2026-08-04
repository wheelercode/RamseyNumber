"""Search-history restrictions used by Ramsey environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .REnvironmentConfig import RTabuMemoryConfig
from .RState import RSearchState


def _read_only_bool_array(
    values: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """
    Return an owned, read-only boolean array.
    """
    result = np.asarray(
        values,
        dtype=np.bool_,
    ).copy()

    result.flags.writeable = False

    return result


@dataclass(
    frozen=True,
    slots=True,
    eq=False,
)
class RMemoryStatus:
    """
    Action restrictions produced by search history.

    edge_tabu_mask:
        True for actions blocked by edge tenure.

    revisit_mask:
        True for actions whose successor state is contained in
        visited-state memory.

    blocked_mask:
        The union of the edge-tabu and revisit masks.
    """

    edge_tabu_mask: NDArray[np.bool_]
    revisit_mask: NDArray[np.bool_]
    blocked_mask: NDArray[np.bool_]

    def __post_init__(self) -> None:
        edge_tabu_mask = _read_only_bool_array(self.edge_tabu_mask)

        revisit_mask = _read_only_bool_array(self.revisit_mask)

        blocked_mask = _read_only_bool_array(self.blocked_mask)

        if not (edge_tabu_mask.shape == revisit_mask.shape == blocked_mask.shape):
            raise ValueError("All memory masks must have the same shape.")

        if edge_tabu_mask.ndim != 1:
            raise ValueError("Memory masks must be one-dimensional.")

        expected_blocked_mask = edge_tabu_mask | revisit_mask

        if not np.array_equal(
            blocked_mask,
            expected_blocked_mask,
        ):
            raise ValueError(
                "blocked_mask must equal " "edge_tabu_mask | revisit_mask."
            )

        object.__setattr__(
            self,
            "edge_tabu_mask",
            edge_tabu_mask,
        )

        object.__setattr__(
            self,
            "revisit_mask",
            revisit_mask,
        )

        object.__setattr__(
            self,
            "blocked_mask",
            blocked_mask,
        )


class RMemory(ABC):
    """
    Interface for environment search-history memory.

    Memory decides which actions history normally blocks.
    Aspiration and deadlock fallback remain responsibilities
    of REnvironment.
    """

    @abstractmethod
    def reset(
        self,
        state: RSearchState,
    ) -> None:
        """
        Forget the previous episode and remember the new
        episode's initial state.
        """
        ...

    @abstractmethod
    def status(
        self,
        state: RSearchState,
        step_number: int,
    ) -> RMemoryStatus:
        """
        Return the actions blocked by current search history.
        """
        ...

    @abstractmethod
    def record_transition(
        self,
        edge: int,
        state: RSearchState,
        step_number: int,
    ) -> None:
        """
        Record one completed transition and its resulting state.
        """
        ...


class RNullMemory(RMemory):
    """
    Memory implementation that never blocks an action.

    This is useful for algorithms that do not use tabu search.
    """

    def reset(
        self,
        state: RSearchState,
    ) -> None:
        return None

    def status(
        self,
        state: RSearchState,
        step_number: int,
    ) -> RMemoryStatus:
        empty = np.zeros(
            state.number_of_edges,
            dtype=np.bool_,
        )

        return RMemoryStatus(
            edge_tabu_mask=empty,
            revisit_mask=empty,
            blocked_mask=empty,
        )

    def record_transition(
        self,
        edge: int,
        state: RSearchState,
        step_number: int,
    ) -> None:
        return None


class RTabuMemory(RMemory):
    """
    Edge-tenure and exact bounded visited-state memory.

    A recently flipped edge is temporarily blocked. Candidate
    successor colorings are also blocked if they occur in the
    configured visited-state window.
    """

    def __init__(
        self,
        number_of_edges: int,
        config: RTabuMemoryConfig | None = None,
    ) -> None:
        if isinstance(number_of_edges, bool) or not isinstance(
            number_of_edges,
            (int, np.integer),
        ):
            raise TypeError("number_of_edges must be an integer.")

        if number_of_edges <= 0:
            raise ValueError("number_of_edges must be positive.")

        self._number_of_edges = int(number_of_edges)

        self._config = config if config is not None else RTabuMemoryConfig()

        # tabu_until[e] is the first step at which edge e
        # becomes available again.
        self._tabu_until = np.zeros(
            self._number_of_edges,
            dtype=np.int32,
        )

        # A queue preserves insertion order for the bounded window.
        self._visited_queue: deque[bytes] = deque()

        # Counts are necessary because the same state can occur more
        # than once while it remains in the bounded queue.
        self._visited_counts: dict[bytes, int] = {}

    @property
    def config(self) -> RTabuMemoryConfig:
        """
        Return the immutable memory configuration.
        """
        return self._config

    @property
    def number_of_edges(self) -> int:
        """
        Return the number of edge actions tracked by memory.
        """
        return self._number_of_edges

    @property
    def tabu_until(self) -> NDArray[np.int32]:
        """
        Return a read-only view of edge expiration steps.
        """
        view = self._tabu_until.view()
        view.flags.writeable = False

        return view

    @property
    def visited_state_count(self) -> int:
        """
        Return the number of states in the bounded history queue.
        """
        return len(self._visited_queue)

    def reset(
        self,
        state: RSearchState,
    ) -> None:
        """
        Clear all memory and remember the initial state.
        """
        self._require_compatible_state(state)

        self._tabu_until.fill(0)
        self._visited_queue.clear()
        self._visited_counts.clear()

        self._remember_state(state)

    def status(
        self,
        state: RSearchState,
        step_number: int,
    ) -> RMemoryStatus:
        """
        Return exact edge-tenure and revisit restrictions.
        """
        self._require_compatible_state(state)
        self._require_nonnegative_step(step_number)

        edge_tabu_mask = self._tabu_until > step_number

        revisit_mask = self._candidate_revisit_mask(state)

        blocked_mask = edge_tabu_mask | revisit_mask

        return RMemoryStatus(
            edge_tabu_mask=edge_tabu_mask,
            revisit_mask=revisit_mask,
            blocked_mask=blocked_mask,
        )

    def record_transition(
        self,
        edge: int,
        state: RSearchState,
        step_number: int,
    ) -> None:
        """
        Make the selected edge tabu and remember the new state.

        step_number is the environment's step number after the
        transition has been applied.
        """
        self._require_compatible_state(state)

        if isinstance(edge, bool) or not isinstance(
            edge,
            (int, np.integer),
        ):
            raise TypeError("edge must be an integer.")

        edge = int(edge)

        if edge < 0 or edge >= self._number_of_edges:
            raise IndexError(f"Invalid edge index: {edge}")

        if isinstance(step_number, bool) or not isinstance(
            step_number,
            (int, np.integer),
        ):
            raise TypeError("step_number must be an integer.")

        step_number = int(step_number)

        if step_number <= 0:
            raise ValueError("step_number must be positive after " "a transition.")

        self._tabu_until[edge] = step_number + self._config.edge_tenure

        self._remember_state(state)

    def _candidate_revisit_mask(
        self,
        state: RSearchState,
    ) -> NDArray[np.bool_]:
        """
        Return successors found in exact visited-state memory.

        Every possible successor differs from the current coloring
        by exactly one edge bit.
        """
        if self._config.visited_state_window == 0 or not self._visited_counts:
            return np.zeros(
                self._number_of_edges,
                dtype=np.bool_,
            )

        packed = np.packbits(
            state.colors,
            bitorder="little",
        )

        # Create packed copies of every possible successor state.
        candidates = np.tile(
            packed,
            (
                self._number_of_edges,
                1,
            ),
        )

        action_indices = np.arange(
            self._number_of_edges,
            dtype=np.int32,
        )

        byte_indices = action_indices // 8

        bit_indices = action_indices % 8

        bit_masks = np.left_shift(
            np.uint8(1),
            bit_indices.astype(np.uint8),
        )

        candidates[
            action_indices,
            byte_indices,
        ] ^= bit_masks

        return np.fromiter(
            (candidate.tobytes() in self._visited_counts for candidate in candidates),
            dtype=np.bool_,
            count=self._number_of_edges,
        )

    def _remember_state(
        self,
        state: RSearchState,
    ) -> None:
        """
        Add one exact state key to the bounded history.
        """
        window = self._config.visited_state_window

        if window == 0:
            return

        key = self._state_key(state)

        self._visited_queue.append(key)

        self._visited_counts[key] = self._visited_counts.get(key, 0) + 1

        while len(self._visited_queue) > window:
            removed = self._visited_queue.popleft()

            remaining = self._visited_counts[removed] - 1

            if remaining == 0:
                del self._visited_counts[removed]
            else:
                self._visited_counts[removed] = remaining

    @staticmethod
    def _state_key(
        state: RSearchState,
    ) -> bytes:
        """
        Pack a binary coloring into an exact compact key.

        This is exact representation, not probabilistic hashing.
        """
        return np.packbits(
            state.colors,
            bitorder="little",
        ).tobytes()

    def _require_compatible_state(
        self,
        state: RSearchState,
    ) -> None:
        """
        Require the configured number of edge actions.
        """
        if state.number_of_edges != self._number_of_edges:
            raise ValueError("Search-state edge count does not " "match memory.")

    @staticmethod
    def _require_nonnegative_step(
        step_number: int,
    ) -> None:
        """
        Validate an environment step number.
        """
        if isinstance(step_number, bool) or not isinstance(
            step_number,
            (int, np.integer),
        ):
            raise TypeError("step_number must be an integer.")

        if step_number < 0:
            raise ValueError("step_number cannot be negative.")
