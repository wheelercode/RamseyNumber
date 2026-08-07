"""Directional structure surrounding every single-edge flip.

:mod:`ramsey.RAction` collapses each edge flip's consequences into a
single scalar reward, discarding the direction and magnitude of the
underlying change. This module reconstructs that hidden structure: for
every edge it derives, per color, how many K5s are at each "deficit"
(number of recolorings needed to become monochromatic in that color) and
how that deficit distribution would change if the edge were flipped, plus
how many K5s would be outright destroyed or created in each color. This
supports action selection strategies that care about more than the
immediate score change, such as favoring flips that also reduce the
number of near-complete (low-deficit) K5s.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .RAction import RActionAnalysis, analyze_actions
from .RState import RSearchState


def _read_only_copy(array: NDArray) -> NDArray:
    """Copy an array, mark the copy read-only, and return it.

    Args:
        array (NDArray): Source array; may be any array-like value.

    Returns:
        NDArray: An independently owned, read-only copy.
    """
    result = np.asarray(array).copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True, eq=False)
class RActionLandscape:
    """Preserve the information hidden by a scalar edge-flip reward.

    Rows correspond to host-graph edges. Deficit columns run from
    zero (already monochromatic) through ``edges_per_clique``.

    ``red_profiles[e, d]`` counts K5s containing edge ``e`` that are
    currently ``d`` edge recolorings away from all red. The blue
    profile has the same meaning in the opposite color direction.

    The delta arrays describe the exact global change to those two
    distributions if the corresponding edge is flipped.

    Attributes:
        source_state (RSearchState): Search state the landscape was
            computed from.
        state_version (int): ``source_state.version`` at computation
            time, used to detect staleness.
        analysis (RActionAnalysis): Underlying action analysis this
            landscape was derived from.
        red_profiles (NDArray[np.uint16]): Shape
            ``(number_of_edges, edges_per_clique + 1)``. ``[e, d]`` is
            the number of K5s containing edge ``e`` currently ``d``
            edge recolorings away from being all red.
        blue_profiles (NDArray[np.uint16]): Same shape and meaning as
            ``red_profiles``, but measuring deficit from all blue.
        red_deficit_deltas (NDArray[np.int32]): Same shape as
            ``red_profiles``; the exact change to the red-deficit
            distribution if the corresponding edge is flipped.
        blue_deficit_deltas (NDArray[np.int32]): Same shape as
            ``blue_profiles``; the exact change to the blue-deficit
            distribution if the corresponding edge is flipped.
        red_violations_destroyed (NDArray[np.int32]): Shape
            ``(number_of_edges,)``. Number of all-red K5s destroyed by
            flipping edge ``e`` (nonzero only for currently red edges).
        red_violations_created (NDArray[np.int32]): Number of new
            all-red K5s created by flipping edge ``e`` (nonzero only for
            currently blue edges).
        blue_violations_destroyed (NDArray[np.int32]): Number of
            all-blue K5s destroyed by flipping edge ``e`` (nonzero only
            for currently blue edges).
        blue_violations_created (NDArray[np.int32]): Number of new
            all-blue K5s created by flipping edge ``e`` (nonzero only
            for currently red edges).
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
        """Validate array shapes, freeze array fields, and check freshness.

        Raises:
            ValueError: If any profile or delta array does not have
                shape ``(number_of_edges, edges_per_clique + 1)``, if
                any violation-count array does not have shape
                ``(number_of_edges,)``, if ``analysis`` does not apply
                to ``source_state``, or if ``state_version`` is stale
                relative to ``source_state``.
        """
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
        """NDArray[np.int32]: Exact monochromatic-score reward of every flip."""
        return self.analysis.immediate_rewards

    @property
    def red_deficit_one_deltas(self) -> NDArray[np.int32]:
        """NDArray[np.int32]: Each flip's change in red deficit-one K5 count.

        Deficit-one K5s are one edge recoloring away from all red; this
        tracks how many enter or leave that state per flip.
        """
        return self.red_deficit_deltas[:, 1]

    @property
    def blue_deficit_one_deltas(self) -> NDArray[np.int32]:
        """NDArray[np.int32]: Each flip's change in blue deficit-one K5 count."""
        return self.blue_deficit_deltas[:, 1]

    @property
    def red_deficit_two_deltas(self) -> NDArray[np.int32]:
        """NDArray[np.int32]: Each flip's change in red deficit-two K5 count."""
        return self.red_deficit_deltas[:, 2]

    @property
    def blue_deficit_two_deltas(self) -> NDArray[np.int32]:
        """NDArray[np.int32]: Each flip's change in blue deficit-two K5 count."""
        return self.blue_deficit_deltas[:, 2]

    def applies_to(self, state: RSearchState) -> bool:
        """Return whether this landscape still describes ``state``.

        Args:
            state (RSearchState): Candidate state to check freshness
                against.

        Returns:
            bool: ``True`` if ``state`` is the exact object this
            landscape was computed from and has not been mutated since.
        """
        return (
            self.source_state is state
            and self.state_version == state.version
        )

    def summary(self) -> dict[str, int | float]:
        """Return objective-neutral scalar descriptors of the landscape.

        Summarizes the reward distribution across all actions (best,
        worst, mean, standard deviation, and counts of improving/
        neutral/worsening actions), plus, among the best-reward actions
        and separately among the neutral (reward == 0) actions, the
        range of red/blue deficit-one deltas they produce. That range
        distinguishes among reward-tied actions by their secondary
        effect on near-complete K5s.

        Returns:
            dict[str, int | float]: Scalar descriptors keyed by name
            (``best_reward``, ``worst_reward``, ``improving_actions``,
            ``neutral_actions``, ``worsening_actions``, ``mean_reward``,
            ``std_reward``, and the
            ``best_``/``neutral_`` ``red``/``blue`` ``_deficit_1_delta_``
            ``min``/``max`` combinations). The neutral-group values are
            ``0`` when no neutral actions exist.
        """
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
    """Calculate the complete directional single-edge action landscape.

    Derives red/blue deficit profiles and their flip deltas from the
    scalar action analysis's ``profiles``/``histogram_deltas`` (the blue
    view is simply the red view with bins reversed, since a K5's blue
    edge count is ``edges_per_clique`` minus its red edge count), and
    computes exact destroyed/created violation counts per edge from the
    boundary bins of ``profiles`` and the edge's current color.

    Args:
        state (RSearchState): Search state to analyze.
        analysis (RActionAnalysis | None): Action analysis for
            ``state``. If ``None``, it is computed via
            :func:`ramsey.RAction.analyze_actions`.

    Returns:
        RActionLandscape: Directional per-color, per-deficit structure
        underlying every single-edge flip in ``state``.

    Raises:
        ValueError: If ``analysis`` does not apply to ``state``.
    """
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
    """Return the minimum of ``values`` selected by ``mask``, or zero.

    Args:
        values (NDArray[np.integer]): Values to select from.
        mask (NDArray[np.bool_]): Boolean mask of the same shape as
            ``values``.

    Returns:
        int: Minimum of the selected values, or ``0`` if ``mask``
        selects nothing.
    """
    selected = values[mask]
    return int(selected.min()) if selected.size else 0


def _masked_max(
    values: NDArray[np.integer],
    mask: NDArray[np.bool_],
) -> int:
    """Return the maximum of ``values`` selected by ``mask``, or zero.

    Args:
        values (NDArray[np.integer]): Values to select from.
        mask (NDArray[np.bool_]): Boolean mask of the same shape as
            ``values``.

    Returns:
        int: Maximum of the selected values, or ``0`` if ``mask``
        selects nothing.
    """
    selected = values[mask]
    return int(selected.max()) if selected.size else 0
