"""Directional structure surrounding every single-edge flip."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .RAction import RActionAnalysis, analyze_actions
from .RState import RSearchState


def _read_only_copy(array: NDArray) -> NDArray:
    result = np.asarray(array).copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True, eq=False)
class RActionLandscape:
    """
    Preserve the information hidden by a scalar edge-flip reward.

    Rows correspond to host-graph edges.  Deficit columns run from
    zero (already monochromatic) through ``edges_per_clique``.

    ``red_profiles[e, d]`` counts K5s containing edge ``e`` that are
    currently ``d`` edge recolorings away from all red.  The blue
    profile has the same meaning in the opposite color direction.

    The delta arrays describe the exact global change to those two
    distributions if the corresponding edge is flipped.
    """

    source_state: RSearchState
    state_version: int
    analysis: RActionAnalysis

    red_profiles: NDArray[np.uint16]
    blue_profiles: NDArray[np.uint16]
    red_deficit_deltas: NDArray[np.int32]
    blue_deficit_deltas: NDArray[np.int32]

    red_violations_destroyed: NDArray[np.int32]
    red_violations_created: NDArray[np.int32]
    blue_violations_destroyed: NDArray[np.int32]
    blue_violations_created: NDArray[np.int32]

    def __post_init__(self) -> None:
        number_of_edges = self.source_state.number_of_edges
        number_of_bins = self.source_state.edges_per_clique + 1

        matrix_shape = (number_of_edges, number_of_bins)
        vector_shape = (number_of_edges,)

        for name in (
            "red_profiles",
            "blue_profiles",
            "red_deficit_deltas",
            "blue_deficit_deltas",
        ):
            value = np.asarray(getattr(self, name))

            if value.shape != matrix_shape:
                raise ValueError(
                    f"{name} must have shape {matrix_shape}."
                )

            object.__setattr__(
                self,
                name,
                _read_only_copy(value),
            )

        for name in (
            "red_violations_destroyed",
            "red_violations_created",
            "blue_violations_destroyed",
            "blue_violations_created",
        ):
            value = np.asarray(getattr(self, name), dtype=np.int32)

            if value.shape != vector_shape:
                raise ValueError(
                    f"{name} must have shape {vector_shape}."
                )

            object.__setattr__(
                self,
                name,
                _read_only_copy(value),
            )

        if not self.analysis.applies_to(self.source_state):
            raise ValueError(
                "Action analysis does not describe the source state."
            )

        if self.state_version != self.source_state.version:
            raise ValueError(
                "Action landscape state version is stale."
            )

    @property
    def exact_rewards(self) -> NDArray[np.int32]:
        """Return the exact monochromatic-score reward of every flip."""
        return self.analysis.immediate_rewards

    @property
    def red_deficit_one_deltas(self) -> NDArray[np.int32]:
        """Return each flip's change in red deficit-one K5 count."""
        return self.red_deficit_deltas[:, 1]

    @property
    def blue_deficit_one_deltas(self) -> NDArray[np.int32]:
        """Return each flip's change in blue deficit-one K5 count."""
        return self.blue_deficit_deltas[:, 1]

    @property
    def red_deficit_two_deltas(self) -> NDArray[np.int32]:
        """Return each flip's change in red deficit-two K5 count."""
        return self.red_deficit_deltas[:, 2]

    @property
    def blue_deficit_two_deltas(self) -> NDArray[np.int32]:
        """Return each flip's change in blue deficit-two K5 count."""
        return self.blue_deficit_deltas[:, 2]

    def applies_to(self, state: RSearchState) -> bool:
        """Return whether this landscape still describes ``state``."""
        return (
            self.source_state is state
            and self.state_version == state.version
        )

    def summary(self) -> dict[str, int | float]:
        """Return objective-neutral scalar descriptors of the landscape."""
        rewards = self.exact_rewards
        best_reward = int(rewards.max())
        best_mask = rewards == best_reward
        neutral_mask = rewards == 0

        return {
            "best_reward": best_reward,
            "worst_reward": int(rewards.min()),
            "improving_actions": int(np.count_nonzero(rewards > 0)),
            "neutral_actions": int(np.count_nonzero(neutral_mask)),
            "worsening_actions": int(np.count_nonzero(rewards < 0)),
            "mean_reward": float(rewards.mean()),
            "std_reward": float(rewards.std()),
            "best_red_deficit_1_delta_min": int(
                self.red_deficit_one_deltas[best_mask].min()
            ),
            "best_red_deficit_1_delta_max": int(
                self.red_deficit_one_deltas[best_mask].max()
            ),
            "best_blue_deficit_1_delta_min": int(
                self.blue_deficit_one_deltas[best_mask].min()
            ),
            "best_blue_deficit_1_delta_max": int(
                self.blue_deficit_one_deltas[best_mask].max()
            ),
            "neutral_red_deficit_1_delta_min": _masked_min(
                self.red_deficit_one_deltas,
                neutral_mask,
            ),
            "neutral_red_deficit_1_delta_max": _masked_max(
                self.red_deficit_one_deltas,
                neutral_mask,
            ),
            "neutral_blue_deficit_1_delta_min": _masked_min(
                self.blue_deficit_one_deltas,
                neutral_mask,
            ),
            "neutral_blue_deficit_1_delta_max": _masked_max(
                self.blue_deficit_one_deltas,
                neutral_mask,
            ),
        }


def calculate_action_landscape(
    state: RSearchState,
    analysis: RActionAnalysis | None = None,
) -> RActionLandscape:
    """Calculate the complete directional single-edge action landscape."""
    if analysis is None:
        analysis = analyze_actions(state)
    elif not analysis.applies_to(state):
        raise ValueError(
            "Action analysis does not describe the supplied state."
        )

    # profiles[:, k] uses the number of blue edges as k.  That is
    # exactly the red completion deficit.  Reversing the columns gives
    # the number of red edges, which is the blue completion deficit.
    red_profiles = analysis.profiles
    blue_profiles = analysis.profiles[:, ::-1]

    red_deficit_deltas = analysis.histogram_deltas
    blue_deficit_deltas = analysis.histogram_deltas[:, ::-1]

    colors = state.colors
    red_edges = colors == 0
    blue_edges = colors == 1

    number_of_edges = state.number_of_edges
    red_destroyed = np.zeros(number_of_edges, dtype=np.int32)
    red_created = np.zeros(number_of_edges, dtype=np.int32)
    blue_destroyed = np.zeros(number_of_edges, dtype=np.int32)
    blue_created = np.zeros(number_of_edges, dtype=np.int32)

    # Red -> blue destroys all-red K5s and can create all-blue K5s.
    red_destroyed[red_edges] = analysis.profiles[red_edges, 0]
    blue_created[red_edges] = analysis.profiles[
        red_edges,
        state.edges_per_clique - 1,
    ]

    # Blue -> red is the color-reversed situation.
    blue_destroyed[blue_edges] = analysis.profiles[
        blue_edges,
        state.edges_per_clique,
    ]
    red_created[blue_edges] = analysis.profiles[blue_edges, 1]

    return RActionLandscape(
        source_state=state,
        state_version=state.version,
        analysis=analysis,
        red_profiles=red_profiles,
        blue_profiles=blue_profiles,
        red_deficit_deltas=red_deficit_deltas,
        blue_deficit_deltas=blue_deficit_deltas,
        red_violations_destroyed=red_destroyed,
        red_violations_created=red_created,
        blue_violations_destroyed=blue_destroyed,
        blue_violations_created=blue_created,
    )


def _masked_min(
    values: NDArray[np.integer],
    mask: NDArray[np.bool_],
) -> int:
    selected = values[mask]
    return int(selected.min()) if selected.size else 0


def _masked_max(
    values: NDArray[np.integer],
    mask: NDArray[np.bool_],
) -> int:
    selected = values[mask]
    return int(selected.max()) if selected.size else 0
