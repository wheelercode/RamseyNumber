"""Objective-neutral vertex color-balance transfer actions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .RAction import analyze_actions
from .RState import RSearchState


def _read_only_copy(
    values: NDArray,
) -> NDArray:
    result = np.asarray(values).copy()
    result.flags.writeable = False
    return result


def vertex_blue_degrees(
    state: RSearchState,
) -> NDArray[np.int16]:
    """Return the number of blue (color-one) edges at every vertex."""
    n_vertices = state.graph.problem.n_vertices

    degrees = np.zeros(
        n_vertices,
        dtype=np.int16,
    )

    blue_edges = state.graph.edges[state.colors == 1]

    np.add.at(
        degrees,
        blue_edges[:, 0],
        1,
    )
    np.add.at(
        degrees,
        blue_edges[:, 1],
        1,
    )

    return degrees


def vertex_color_imbalances(
    state: RSearchState,
) -> NDArray[np.int16]:
    """Return blue degree minus red degree for every vertex."""
    n_vertices = state.graph.problem.n_vertices

    blue_degrees = vertex_blue_degrees(state)

    return (
        2 * blue_degrees
        - (n_vertices - 1)
    ).astype(np.int16)


def vertex_balance_energy(
    state: RSearchState,
) -> int:
    """Return sum of squared vertex color imbalances; lower is better."""
    imbalances = vertex_color_imbalances(
        state,
    ).astype(np.int32)

    return int(
        np.sum(
            imbalances * imbalances,
            dtype=np.int64,
        )
    )


@dataclass(
    frozen=True,
    slots=True,
)
class RVertexBalanceTransfer:
    """One blue-degree transfer from donor to recipient through a pivot."""

    donor_vertex: int
    recipient_vertex: int
    pivot_vertex: int
    donor_edge: int
    recipient_edge: int


@dataclass(
    frozen=True,
    slots=True,
    eq=False,
)
class RVertexBalanceAnalysis:
    """Exact structural consequences of all legal balance transfers."""

    source_state: RSearchState
    state_version: int

    donor_vertices: NDArray[np.uint8]
    recipient_vertices: NDArray[np.uint8]
    pivot_vertices: NDArray[np.uint8]
    donor_edges: NDArray[np.uint16]
    recipient_edges: NDArray[np.uint16]

    balance_rewards: NDArray[np.int32]
    histogram_deltas: NDArray[np.int32]
    exact_rewards: NDArray[np.int32]
    resulting_scores: NDArray[np.int32]

    def __post_init__(self) -> None:
        arrays = {
            "donor_vertices": np.asarray(
                self.donor_vertices,
                dtype=np.uint8,
            ),
            "recipient_vertices": np.asarray(
                self.recipient_vertices,
                dtype=np.uint8,
            ),
            "pivot_vertices": np.asarray(
                self.pivot_vertices,
                dtype=np.uint8,
            ),
            "donor_edges": np.asarray(
                self.donor_edges,
                dtype=np.uint16,
            ),
            "recipient_edges": np.asarray(
                self.recipient_edges,
                dtype=np.uint16,
            ),
            "balance_rewards": np.asarray(
                self.balance_rewards,
                dtype=np.int32,
            ),
            "exact_rewards": np.asarray(
                self.exact_rewards,
                dtype=np.int32,
            ),
            "resulting_scores": np.asarray(
                self.resulting_scores,
                dtype=np.int32,
            ),
        }

        number_of_actions = len(
            arrays["balance_rewards"]
        )

        for name, values in arrays.items():
            if values.shape != (number_of_actions,):
                raise ValueError(
                    f"{name} has the wrong shape."
                )

        histogram_deltas = np.asarray(
            self.histogram_deltas,
            dtype=np.int32,
        )

        expected_delta_shape = (
            number_of_actions,
            self.source_state.edges_per_clique + 1,
        )

        if histogram_deltas.shape != expected_delta_shape:
            raise ValueError(
                "histogram_deltas has the wrong shape."
            )

        object.__setattr__(
            self,
            "state_version",
            int(self.state_version),
        )

        for name, values in arrays.items():
            object.__setattr__(
                self,
                name,
                _read_only_copy(values),
            )

        object.__setattr__(
            self,
            "histogram_deltas",
            _read_only_copy(histogram_deltas),
        )

    @property
    def number_of_actions(self) -> int:
        return len(self.balance_rewards)

    def transfer(
        self,
        action_index: int,
    ) -> RVertexBalanceTransfer:
        if action_index < 0 or action_index >= self.number_of_actions:
            raise IndexError(
                "action_index is out of range."
            )

        return RVertexBalanceTransfer(
            donor_vertex=int(
                self.donor_vertices[action_index]
            ),
            recipient_vertex=int(
                self.recipient_vertices[action_index]
            ),
            pivot_vertex=int(
                self.pivot_vertices[action_index]
            ),
            donor_edge=int(
                self.donor_edges[action_index]
            ),
            recipient_edge=int(
                self.recipient_edges[action_index]
            ),
        )

    def applies_to(
        self,
        state: RSearchState,
    ) -> bool:
        return (
            self.source_state is state
            and self.state_version == state.version
        )


def _edge_index_matrix(
    state: RSearchState,
) -> NDArray[np.int32]:
    n_vertices = state.graph.problem.n_vertices

    result = np.full(
        (n_vertices, n_vertices),
        -1,
        dtype=np.int32,
    )

    edge_indices = np.arange(
        state.number_of_edges,
        dtype=np.int32,
    )

    first = state.graph.edges[:, 0]
    second = state.graph.edges[:, 1]

    result[first, second] = edge_indices
    result[second, first] = edge_indices

    return result


def _candidate_transfers(
    state: RSearchState,
    imbalances: NDArray[np.int16],
) -> tuple[
    NDArray[np.uint8],
    NDArray[np.uint8],
    NDArray[np.uint8],
    NDArray[np.uint16],
    NDArray[np.uint16],
    NDArray[np.int32],
]:
    """Enumerate transfers that strictly reduce squared imbalance."""
    donors = np.flatnonzero(
        imbalances > 0
    )
    recipients = np.flatnonzero(
        imbalances < 0
    )

    edge_index = _edge_index_matrix(
        state,
    )

    donor_vertices: list[int] = []
    recipient_vertices: list[int] = []
    pivot_vertices: list[int] = []
    donor_edges: list[int] = []
    recipient_edges: list[int] = []
    balance_rewards: list[int] = []

    n_vertices = state.graph.problem.n_vertices

    for donor in donors:
        donor = int(donor)

        for recipient in recipients:
            recipient = int(recipient)

            # Moving one blue degree from donor to recipient changes
            # their imbalances by -2 and +2 respectively. The pivot's
            # degree is unchanged. This is the exact reduction in
            # squared imbalance energy.
            balance_reward = int(
                4
                * (
                    int(imbalances[donor])
                    - int(imbalances[recipient])
                    - 2
                )
            )

            if balance_reward <= 0:
                continue

            for pivot in range(n_vertices):
                if pivot in (
                    donor,
                    recipient,
                ):
                    continue

                donor_edge = int(
                    edge_index[donor, pivot]
                )
                recipient_edge = int(
                    edge_index[recipient, pivot]
                )

                if (
                    state.colors[donor_edge] == 1
                    and state.colors[recipient_edge] == 0
                ):
                    donor_vertices.append(donor)
                    recipient_vertices.append(recipient)
                    pivot_vertices.append(pivot)
                    donor_edges.append(donor_edge)
                    recipient_edges.append(
                        recipient_edge
                    )
                    balance_rewards.append(
                        balance_reward
                    )

    return (
        np.asarray(
            donor_vertices,
            dtype=np.uint8,
        ),
        np.asarray(
            recipient_vertices,
            dtype=np.uint8,
        ),
        np.asarray(
            pivot_vertices,
            dtype=np.uint8,
        ),
        np.asarray(
            donor_edges,
            dtype=np.uint16,
        ),
        np.asarray(
            recipient_edges,
            dtype=np.uint16,
        ),
        np.asarray(
            balance_rewards,
            dtype=np.int32,
        ),
    )


def analyze_vertex_balance_transfers(
    state: RSearchState,
) -> RVertexBalanceAnalysis:
    """
    Analyze every pivot transfer that moves both endpoint imbalances
    toward zero.

    For donor u, recipient v, and pivot w the required colors are:

        u-w = blue (1)
        v-w = red  (0)

    Recoloring those edges to red and blue transfers one unit of blue
    degree from u to v while leaving w and the global blue-edge count
    unchanged.
    """
    imbalances = vertex_color_imbalances(
        state,
    )

    (
        donor_vertices,
        recipient_vertices,
        pivot_vertices,
        donor_edges,
        recipient_edges,
        balance_rewards,
    ) = _candidate_transfers(
        state,
        imbalances,
    )

    number_of_actions = len(
        balance_rewards
    )

    number_of_bins = (
        state.edges_per_clique + 1
    )

    if number_of_actions == 0:
        return RVertexBalanceAnalysis(
            source_state=state,
            state_version=state.version,
            donor_vertices=donor_vertices,
            recipient_vertices=recipient_vertices,
            pivot_vertices=pivot_vertices,
            donor_edges=donor_edges,
            recipient_edges=recipient_edges,
            balance_rewards=balance_rewards,
            histogram_deltas=np.empty(
                (0, number_of_bins),
                dtype=np.int32,
            ),
            exact_rewards=np.empty(
                0,
                dtype=np.int32,
            ),
            resulting_scores=np.empty(
                0,
                dtype=np.int32,
            ),
        )

    single_analysis = analyze_actions(
        state,
    )

    histogram_deltas = (
        single_analysis.histogram_deltas[
            donor_edges
        ]
        + single_analysis.histogram_deltas[
            recipient_edges
        ]
    ).astype(np.int32)

    # The two candidate edges share the pivot. Any forbidden clique
    # containing both sees one blue->red and one red->blue mutation,
    # so its final blue count is unchanged. The sum of the two
    # independently computed edge deltas counted that clique twice;
    # remove those two artificial one-edge transitions exactly.
    for action_index, (
        donor_edge,
        recipient_edge,
    ) in enumerate(
        zip(
            donor_edges,
            recipient_edges,
        )
    ):
        shared_cliques = np.intersect1d(
            state.index.edge_to_cliques[
                int(donor_edge)
            ],
            state.index.edge_to_cliques[
                int(recipient_edge)
            ],
            assume_unique=True,
        )

        shared_counts = state.color_one_counts[
            shared_cliques
        ]

        shared_histogram = np.bincount(
            shared_counts,
            minlength=number_of_bins,
        ).astype(np.int32)

        histogram_deltas[
            action_index,
            :,
        ] += 2 * shared_histogram

        histogram_deltas[
            action_index,
            :-1,
        ] -= shared_histogram[1:]

        histogram_deltas[
            action_index,
            1:,
        ] -= shared_histogram[:-1]

    exact_rewards = -(
        histogram_deltas[:, 0]
        + histogram_deltas[:, -1]
    ).astype(np.int32)

    resulting_scores = (
        state.score - exact_rewards
    ).astype(np.int32)

    return RVertexBalanceAnalysis(
        source_state=state,
        state_version=state.version,
        donor_vertices=donor_vertices,
        recipient_vertices=recipient_vertices,
        pivot_vertices=pivot_vertices,
        donor_edges=donor_edges,
        recipient_edges=recipient_edges,
        balance_rewards=balance_rewards,
        histogram_deltas=histogram_deltas,
        exact_rewards=exact_rewards,
        resulting_scores=resulting_scores,
    )


def apply_vertex_balance_transfer(
    state: RSearchState,
    transfer: RVertexBalanceTransfer,
) -> int:
    """Apply one validated pivot transfer and return exact score reward."""
    n_vertices = state.graph.problem.n_vertices

    vertices = (
        transfer.donor_vertex,
        transfer.recipient_vertex,
        transfer.pivot_vertex,
    )

    if any(
        vertex < 0 or vertex >= n_vertices
        for vertex in vertices
    ):
        raise IndexError(
            "transfer contains an invalid vertex."
        )

    if len(set(vertices)) != 3:
        raise ValueError(
            "donor, recipient, and pivot must be distinct."
        )

    edge_index = _edge_index_matrix(
        state,
    )

    expected_donor_edge = int(
        edge_index[
            transfer.donor_vertex,
            transfer.pivot_vertex,
        ]
    )
    expected_recipient_edge = int(
        edge_index[
            transfer.recipient_vertex,
            transfer.pivot_vertex,
        ]
    )

    if (
        transfer.donor_edge
        != expected_donor_edge
        or transfer.recipient_edge
        != expected_recipient_edge
    ):
        raise ValueError(
            "transfer edge indexes do not match its vertices."
        )

    if state.colors[transfer.donor_edge] != 1:
        raise ValueError(
            "donor-pivot edge must currently be blue."
        )

    if state.colors[transfer.recipient_edge] != 0:
        raise ValueError(
            "recipient-pivot edge must currently be red."
        )

    imbalances = vertex_color_imbalances(
        state,
    )

    if imbalances[transfer.donor_vertex] <= 0:
        raise ValueError(
            "donor vertex must currently be blue-heavy."
        )

    if imbalances[transfer.recipient_vertex] >= 0:
        raise ValueError(
            "recipient vertex must currently be red-heavy."
        )

    return state.apply_edge_recoloring(
        np.asarray(
            (
                transfer.donor_edge,
                transfer.recipient_edge,
            ),
            dtype=np.int32,
        ),
        np.asarray(
            (0, 1),
            dtype=np.uint8,
        ),
    )