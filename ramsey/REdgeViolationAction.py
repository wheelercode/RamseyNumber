"""Structural analysis of edge participation in Ramsey violations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .RState import RSearchState


def _owned_read_only(
    values: NDArray,
) -> NDArray:
    """Return an owned, read-only copy of an array."""
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
    """

    source_state: RSearchState
    state_version: int
    violation_loads: NDArray[np.int32]

    def __post_init__(self) -> None:
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
        """Return the greatest current violation load of any edge."""
        if self.violation_loads.size == 0:
            return 0

        return int(self.violation_loads.max())

    @property
    def total_load(self) -> int:
        """Return total edge participation across all violations."""
        return int(
            self.violation_loads.sum(
                dtype=np.int64,
            )
        )

    def applies_to(
        self,
        state: RSearchState,
    ) -> bool:
        """Return whether this analysis describes the current state."""
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
    """
    profiles = state.action_profiles

    return (
        profiles[:, 0].astype(np.int32)
        + profiles[:, -1].astype(np.int32)
    )


def analyze_edge_violations(
    state: RSearchState,
) -> REdgeViolationAnalysis:
    """Analyze current monochromatic-clique participation by edge."""
    return REdgeViolationAnalysis(
        source_state=state,
        state_version=state.version,
        violation_loads=edge_violation_loads(state),
    )