"""Greedy edge selection restricted by local K5 histogram structure."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from .REnvironment import REnvironment
from .RPolicy import RPolicy
from .RState import RSearchState


def histogram_band_loads(
    state: RSearchState,
    h_min: int,
    h_max: int,
) -> NDArray[np.int64]:
    """Count incident K5s in a selected histogram band for every edge.

    Args:
        state (RSearchState): Current Ramsey search state.
        h_min (int): Lowest included color-one histogram bin.
        h_max (int): Highest included color-one histogram bin.

    Returns:
        NDArray[np.int64]: One structural band-load count per host edge.

    Raises:
        TypeError: If a histogram bound is not an integer.
        ValueError: If the requested band is invalid for the state.
    """
    h_min = _require_integer("h_min", h_min)
    h_max = _require_integer("h_max", h_max)

    if h_min < 0:
        raise ValueError("h_min cannot be negative.")

    if h_max < h_min:
        raise ValueError("h_max cannot be less than h_min.")

    if h_max > state.edges_per_clique:
        raise ValueError(
            f"h_max cannot exceed H{state.edges_per_clique} "
            "for the current clique size."
        )

    return state.action_profiles[
        :,
        h_min : h_max + 1,
    ].sum(
        axis=1,
        dtype=np.int64,
    )


@dataclass(frozen=True, slots=True)
class RHistogramBandPolicyConfig:
    """Configuration for structural-pool exact-score greedy search.

    The policy first ranks currently available edges by the number of
    incident K5s whose color-one counts fall in ``H[h_min:h_max]``.
    It keeps at least ``candidate_pool_size`` of the most strongly
    represented edges. Ties at the structural cutoff are retained.
    Exact immediate Ramsey-score reward is used only after that pool
    has been formed.

    Args:
        h_min (int): Lowest included K5 histogram bin.
        h_max (int): Highest included K5 histogram bin.
        candidate_pool_size (int): Minimum number of structurally
            preferred available edges retained for greedy ranking.
    """

    h_min: int = 3
    h_max: int = 7
    candidate_pool_size: int | None = None

    def __post_init__(self) -> None:
        h_min = _require_integer(
            "h_min",
            self.h_min,
        )

        h_max = _require_integer(
            "h_max",
            self.h_max,
        )

        candidate_pool_size = (
            self.candidate_pool_size
        )

        if candidate_pool_size is not None:
            candidate_pool_size = _require_integer(
                "candidate_pool_size",
                candidate_pool_size,
            )

        if h_min < 0:
            raise ValueError(
                "h_min cannot be negative."
            )

        if h_max < h_min:
            raise ValueError(
                "h_max cannot be less than h_min."
            )

        if (
            candidate_pool_size is not None
            and candidate_pool_size <= 0
        ):
            raise ValueError(
                "candidate_pool_size must be positive."
            )

        object.__setattr__(
            self,
            "h_min",
            h_min,
        )

        object.__setattr__(
            self,
            "h_max",
            h_max,
        )

        object.__setattr__(
            self,
            "candidate_pool_size",
            candidate_pool_size,
        )


@dataclass(frozen=True, slots=True)
class RHistogramBandDecision:
    """Observable details of one histogram-band policy decision."""

    edge: int
    exact_reward: int
    band_load: int
    cutoff_band_load: int
    candidate_count: int


@dataclass(slots=True)
class RHistogramBandGreedyPolicy(RPolicy):
    """Choose exact-score-greedy actions from a structural H-band pool."""

    rng: np.random.Generator
    config: RHistogramBandPolicyConfig = field(
        default_factory=RHistogramBandPolicyConfig
    )
    last_decision: RHistogramBandDecision | None = field(
        init=False,
        default=None,
    )

    @property
    def name(self) -> str:
        """Return a stable name describing the configured histogram band."""
        return (
            f"histogram-band-H{self.config.h_min}-H{self.config.h_max}"
            f"-greedy-exact"
        )

    @property
    def requires_full_analysis(self) -> bool:
        """Return true because selection reuses complete action analysis."""
        return True

    def select_action(
        self,
        environment: REnvironment,
    ) -> int:
        """Return the best exact-score action in the structural pool.

        Args:
            environment (REnvironment): Active search environment.

        Returns:
            int: Selected host-edge index.

        Raises:
            RuntimeError: If the environment exposes no available action.
        """
        analysis = environment.analyze_actions()

        band_loads = histogram_band_loads(
            environment.state,
            self.config.h_min,
            self.config.h_max,
        )

        available_edges = np.flatnonzero(
            analysis.available_mask
        ).astype(np.int32)

        if available_edges.size == 0:
            raise RuntimeError("Environment exposes no available action.")

        available_loads = band_loads[available_edges]

        if self.config.candidate_pool_size is None:
            # Intrinsic pool:
            # retain edges having at least the
            # mean H-band exposure.
            cutoff_band_load = int(
                np.ceil(
                    available_loads.mean()
                )
            )
        else:
            target_size = min(
                self.config.candidate_pool_size,
                int(available_edges.size),
            )

            cutoff_index = (
                int(available_edges.size)
                - target_size
            )

            cutoff_band_load = int(
                np.partition(
                    available_loads,
                    cutoff_index,
                )[cutoff_index]
            )

        candidate_mask = (
            analysis.available_mask
            & (band_loads >= cutoff_band_load)
        )

        exact_rewards = analysis.action_analysis.immediate_rewards
        candidate_edges = np.flatnonzero(candidate_mask).astype(np.int32)
        candidate_rewards = exact_rewards[candidate_edges]

        best_reward = int(candidate_rewards.max())
        best_edges = candidate_edges[candidate_rewards == best_reward]
        edge = int(self.rng.choice(best_edges))

        self.last_decision = RHistogramBandDecision(
            edge=edge,
            exact_reward=best_reward,
            band_load=int(band_loads[edge]),
            cutoff_band_load=cutoff_band_load,
            candidate_count=int(candidate_edges.size),
        )

        return edge


def _require_integer(
    name: str,
    value: int,
) -> int:
    """Return a validated built-in integer."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")

    return int(value)