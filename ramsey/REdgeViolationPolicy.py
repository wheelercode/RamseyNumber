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

    Attributes:
        rng (numpy.random.Generator): Random source used to break ties
            uniformly among edges that share the maximum violation load.
    """

    rng: np.random.Generator

    @property
    def name(self) -> str:
        """str: Stable identifier for this policy, ``"edge-violation-load"``."""
        return "edge-violation-load"

    def select_action(
        self,
        environment: REnvironment,
    ) -> int:
        """Select an available edge with the greatest violation load.

        Args:
            environment (REnvironment): Environment supplying the
                current search state and the mask of currently
                available actions.

        Returns:
            int: Index of an available edge whose violation load
            (:func:`REdgeViolationAction.analyze_edge_violations`) is
            maximal, chosen uniformly at random among ties.

        Raises:
            RuntimeError: If no edge is available to select.
        """
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