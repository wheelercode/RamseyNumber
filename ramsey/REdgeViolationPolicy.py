"""Edge-selection policy based only on current violation load."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .REdgeViolationAction import (
    analyze_edge_violations,
)
from .REnvironment import REnvironment
from .RPolicy import RPolicy


@dataclass(slots=True)
class REdgeViolationPolicy(RPolicy):
    """
    Flip an available edge having maximum violation participation.

    This deliberately ignores the consequences of the flip and ranks
    edges only by how many current monochromatic cliques contain them.
    """

    rng: np.random.Generator

    @property
    def name(self) -> str:
        return "edge-violation-load"

    def select_action(
        self,
        environment: REnvironment,
    ) -> int:
        available_mask = (
            environment.available_action_mask_fast()
        )

        analysis = analyze_edge_violations(
            environment.state
        )

        masked_loads = np.where(
            available_mask,
            analysis.violation_loads,
            -1,
        )

        maximum_load = int(
            masked_loads.max()
        )

        candidates = np.flatnonzero(
            masked_loads == maximum_load
        ).astype(np.int32)

        if candidates.size == 0:
            raise RuntimeError(
                "No edge-violation actions are available."
            )

        return int(
            self.rng.choice(candidates)
        )