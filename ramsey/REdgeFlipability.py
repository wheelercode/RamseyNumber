"""Measure how structurally free each edge is to change color."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .RAction import RActionAnalysis
from .RState import RSearchState


def _read_only_copy(array: NDArray) -> NDArray:
    result = np.asarray(array).copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True, eq=False)
class REdgeFlipabilityAnalysis:
    """Per-edge K5 role profiles and discrete flipability totals."""

    source_state: RSearchState
    state_version: int

    # role_profiles[e, k - 1] is the number of K5s containing edge e
    # in which exactly k of the ten K5 edges have e's current color.
    role_profiles: NDArray[np.uint16]

    # Sum of discrete local flipability across every K5 containing
    # each edge. Local flipability is k - 1, where k is the number of
    # K5 edges having the edge's current color.
    flipabilities: NDArray[np.int64]

    def __post_init__(self) -> None:
        number_of_edges = self.source_state.number_of_edges
        edges_per_clique = self.source_state.edges_per_clique

        expected_profile_shape = (
            number_of_edges,
            edges_per_clique,
        )

        if self.role_profiles.shape != expected_profile_shape:
            raise ValueError(
                "role_profiles must have shape "
                f"{expected_profile_shape}."
            )

        if self.flipabilities.shape != (number_of_edges,):
            raise ValueError(
                "flipabilities must have one value per edge."
            )

        if np.any(self.flipabilities < 0):
            raise ValueError(
                "flipabilities cannot be negative."
            )

        object.__setattr__(
            self,
            "role_profiles",
            _read_only_copy(self.role_profiles),
        )
        object.__setattr__(
            self,
            "flipabilities",
            _read_only_copy(self.flipabilities),
        )

    def applies_to(self, state: RSearchState) -> bool:
        """Return whether this analysis still describes ``state``."""
        return (
            self.source_state is state
            and self.state_version == state.version
        )

    def summary(self) -> dict[str, int | float]:
        """Return compact distribution statistics for the state."""
        values = self.flipabilities

        return {
            "flipability_mean": float(values.mean()),
            "flipability_std": float(values.std()),
            "flipability_min": int(values.min()),
            "flipability_max": int(values.max()),
            "flipability_p05": float(np.quantile(values, 0.05)),
            "flipability_p50": float(np.quantile(values, 0.50)),
            "flipability_p95": float(np.quantile(values, 0.95)),
        }


def calculate_edge_flipability(
    state: RSearchState,
    analysis: RActionAnalysis | None = None,
) -> REdgeFlipabilityAnalysis:
    """
    Calculate the structural flipability of every host-graph edge.

    For one K5 containing edge ``e``, let ``k`` be the number of K5
    edges having ``e``'s current color. The discrete local flipability
    is

        k - 1.

    This is literally the number of same-color edges that remain in
    the K5 if ``e`` is flipped. A unique minority-color edge therefore
    has local flipability zero, while an edge in a monochromatic K5 has
    local flipability nine.

    Global edge flipability is the integer sum of those local values.
    Every edge of the same complete host graph participates in the same
    number of K5s, so the totals are directly comparable without an
    arbitrary continuous weighting.
    """
    if state.graph.problem.n_colors != 2:
        raise ValueError(
            "Edge flipability currently requires exactly two colors."
        )

    if analysis is not None and not analysis.applies_to(state):
        raise ValueError(
            "Action analysis does not describe the supplied state."
        )

    profiles = (
        state.action_profiles
        if analysis is None
        else analysis.profiles
    )

    edges_per_clique = state.edges_per_clique
    number_of_edges = state.number_of_edges

    role_profiles = np.empty(
        (number_of_edges, edges_per_clique),
        dtype=np.uint16,
    )

    red_edges = state.colors == 0
    blue_edges = state.colors == 1

    # profiles[e, b] counts K5s containing e with exactly b blue
    # edges. For a red edge, its same-color count is 10 - b, so bins
    # 9..0 correspond to same-color counts 1..10. For a blue edge,
    # bins 1..10 already correspond to same-color counts 1..10.
    role_profiles[red_edges] = profiles[
        red_edges,
        edges_per_clique - 1 :: -1,
    ]
    role_profiles[blue_edges] = profiles[
        blue_edges,
        1 : edges_per_clique + 1,
    ]

    cliques_per_edge = state.index.cliques_per_edge
    profile_totals = role_profiles.sum(
        axis=1,
        dtype=np.uint32,
    )

    if np.any(profile_totals != cliques_per_edge):
        raise RuntimeError(
            "Edge role profiles do not contain every incident clique."
        )

    local_flipabilities = np.arange(
        edges_per_clique,
        dtype=np.int64,
    )

    flipabilities = (
        role_profiles.astype(np.int64)
        @ local_flipabilities
    )

    return REdgeFlipabilityAnalysis(
        source_state=state,
        state_version=state.version,
        role_profiles=role_profiles,
        flipabilities=flipabilities,
    )