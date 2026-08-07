"""Structural analysis of edge participation in Ramsey violations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .RState import RSearchState


def _owned_read_only(
    values: NDArray,
) -> NDArray:
    """Return an owned, read-only copy of an array.

    Args:
        values (numpy.ndarray): Source array or array-like value to copy.

    Returns:
        numpy.ndarray: A new array that owns its memory, with the
        ``writeable`` flag cleared.
    """
    result = np.asarray(values).copy()

    result.flags.writeable = False

    return result


@dataclass(
    frozen=True,
    slots=True,
    eq=False,
)
class REdgeViolationAnalysis:
    """
    Monochromatic-clique participation of every edge.

    violation_loads[e] is the number of currently monochromatic
    forbidden cliques containing edge e.  This deliberately measures
    only current structural involvement; it does not estimate the
    score change, danger, or other consequences of flipping the edge.

    Attributes:
        source_state (RSearchState): Search state this analysis
            describes.
        state_version (int): Version of ``source_state`` at the time of
            analysis, used by :meth:`applies_to` to detect staleness.
        violation_loads (numpy.ndarray): Read-only ``int32`` array of
            shape ``(number_of_edges,)`` giving the number of currently
            monochromatic forbidden cliques containing each edge.
    """

    source_state: RSearchState
    state_version: int
    violation_loads: NDArray[np.int32]

    def __post_init__(self) -> None:
        """Validate the shape and non-negativity of ``violation_loads``.

        Raises:
            ValueError: If ``violation_loads`` does not have one value
                per edge, or if any value is negative.
        """
        expected_shape = (
            self.source_state.number_of_edges,
        )

        loads = np.asarray(
            self.violation_loads,
            dtype=np.int32,
        )

        if loads.shape != expected_shape:
            raise ValueError(
                "violation_loads has the wrong shape."
            )

        if np.any(loads < 0):
            raise ValueError(
                "violation_loads cannot be negative."
            )

        object.__setattr__(
            self,
            "state_version",
            int(self.state_version),
        )

        object.__setattr__(
            self,
            "violation_loads",
            _owned_read_only(loads),
        )

    @property
    def maximum_load(self) -> int:
        """int: Greatest current violation load of any edge, or zero if there are no edges."""
        if self.violation_loads.size == 0:
            return 0

        return int(self.violation_loads.max())

    @property
    def total_load(self) -> int:
        """int: Total edge participation summed across all violations."""
        return int(
            self.violation_loads.sum(
                dtype=np.int64,
            )
        )

    def applies_to(
        self,
        state: RSearchState,
    ) -> bool:
        """Return whether this analysis describes the current state.

        Args:
            state (RSearchState): Candidate state to check.

        Returns:
            bool: ``True`` if ``state`` is the same object as
            :attr:`source_state` and has not been mutated since (its
            version still matches :attr:`state_version`).
        """
        return (
            self.source_state is state
            and self.state_version == state.version
        )


def edge_violation_loads(
    state: RSearchState,
) -> NDArray[np.int32]:
    """
    Count current monochromatic forbidden cliques per edge.

    For the symmetric two-color K5 problem, profile bin 0 contains
    all-red K5s and the final profile bin contains all-blue K5s.
    Their sum is therefore exactly the number of violations containing
    each edge.

    Args:
        state (RSearchState): Search state whose edges are counted.

    Returns:
        numpy.ndarray: ``int32`` array of shape ``(number_of_edges,)``
        with the number of currently monochromatic forbidden cliques
        containing each edge.
    """
    profiles = state.action_profiles

    return (
        profiles[:, 0].astype(np.int32)
        + profiles[:, -1].astype(np.int32)
    )


def analyze_edge_violations(
    state: RSearchState,
) -> REdgeViolationAnalysis:
    """Analyze current monochromatic-clique participation by edge.

    Args:
        state (RSearchState): Search state to analyze.

    Returns:
        REdgeViolationAnalysis: Wrapped violation-load counts for
        ``state``, tagged with its version for staleness checks.
    """
    return REdgeViolationAnalysis(
        source_state=state,
        state_version=state.version,
        violation_loads=edge_violation_loads(state),
    )