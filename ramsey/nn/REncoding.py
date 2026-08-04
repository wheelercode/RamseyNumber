""" "Conversion of Ramsey search information into neural model inputs."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import torch

from ..RGraph import RGraph
from ..RState import RSearchState

PAIR_INPUT_CHANNELS = 3

EDGE_COLOR_CHANNEL = 0
DIAGONAL_CHANNEL = 1
AVAILABILITY_CHANNEL = 2


def encode_pair_features(
    state: RSearchState,
    available_mask: NDArray[np.bool_],
) -> NDArray[np.float32]:
    """
    Encode one state as a symmetric vertex-pair feature array.

    Channels
    --------
    0:
        Edge color represented as -1 for color zero and +1 for
        color one.

    1:
        Diagonal indicator. This distinguishes diagonal positions
        from real edge positions.

    2:
        Action availability represented as zero or one.
    """
    available_mask = _validate_available_mask(
        available_mask,
        state.number_of_edges,
    )

    graph = state.graph
    n_vertices = graph.problem.n_vertices

    pair_features = np.zeros(
        (
            n_vertices,
            n_vertices,
            PAIR_INPUT_CHANNELS,
        ),
        dtype=np.float32,
    )

    i = graph.edges[:, 0].astype(np.int64)

    j = graph.edges[:, 1].astype(np.int64)

    signed_colors = state.colors.astype(np.float32) * 2.0 - 1.0

    pair_features[
        i,
        j,
        EDGE_COLOR_CHANNEL,
    ] = signed_colors

    pair_features[
        j,
        i,
        EDGE_COLOR_CHANNEL,
    ] = signed_colors

    diagonal = np.arange(n_vertices)

    pair_features[
        diagonal,
        diagonal,
        DIAGONAL_CHANNEL,
    ] = 1.0

    availability = available_mask.astype(np.float32)

    pair_features[
        i,
        j,
        AVAILABILITY_CHANNEL,
    ] = availability

    pair_features[
        j,
        i,
        AVAILABILITY_CHANNEL,
    ] = availability

    return pair_features


def build_network_input(
    state: RSearchState,
    available_mask: NDArray[np.bool_],
    device: torch.device | str | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """
    Build one batched pair-input tensor and action-mask tensor.

    Returns
    -------
    pair_tensor:
        Shape:

            (1, n_vertices, n_vertices, 3)

    mask_tensor:
        Shape:

            (1, number_of_edges)
    """
    normalized_mask = _validate_available_mask(
        available_mask,
        state.number_of_edges,
    )

    pair_features = encode_pair_features(
        state,
        normalized_mask,
    )

    pair_tensor = torch.as_tensor(
        pair_features,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    mask_tensor = torch.tensor(
        normalized_mask,
        dtype=torch.bool,
        device=device,
    ).unsqueeze(0)

    return (
        pair_tensor,
        mask_tensor,
    )


def edge_permutation_indices(
    graph: RGraph,
    vertex_permutation: NDArray[np.integer],
) -> NDArray[np.int64]:
    """
    Map old edge indexes to new indexes after vertex renaming.

    If the returned array is named edge_mapping, then:

        new_values[edge_mapping] = old_values

    applies the vertex permutation to an edge-indexed array.
    """
    n_vertices = graph.problem.n_vertices

    permutation = np.asarray(vertex_permutation)

    expected_shape = (n_vertices,)

    if permutation.shape != expected_shape:
        raise ValueError(
            "Expected permutation shape "
            f"{expected_shape}, received "
            f"{permutation.shape}."
        )

    if not np.issubdtype(
        permutation.dtype,
        np.integer,
    ):
        raise TypeError("vertex_permutation must " "contain integers.")

    permutation = permutation.astype(
        np.int64,
        copy=False,
    )

    if not np.array_equal(
        np.sort(permutation),
        np.arange(n_vertices),
    ):
        raise ValueError(
            "vertex_permutation must contain " "every vertex exactly once."
        )

    edge_lookup = np.full(
        (
            n_vertices,
            n_vertices,
        ),
        -1,
        dtype=np.int64,
    )

    edge_numbers = np.arange(
        graph.number_of_edges,
        dtype=np.int64,
    )

    i = graph.edges[:, 0].astype(np.int64)

    j = graph.edges[:, 1].astype(np.int64)

    edge_lookup[
        i,
        j,
    ] = edge_numbers

    edge_lookup[
        j,
        i,
    ] = edge_numbers

    renamed_i = permutation[i]
    renamed_j = permutation[j]

    mapping = edge_lookup[
        renamed_i,
        renamed_j,
    ]

    mapping.flags.writeable = False

    return mapping


def _validate_available_mask(
    available_mask: NDArray[np.bool_],
    number_of_edges: int,
) -> NDArray[np.bool_]:
    """
    Return a validated boolean action mask without copying it.
    """
    available_mask = np.asarray(available_mask)

    expected_shape = (number_of_edges,)

    if available_mask.shape != expected_shape:
        raise ValueError(
            "Expected available-mask shape "
            f"{expected_shape}, received "
            f"{available_mask.shape}."
        )

    if available_mask.dtype != np.bool_:
        raise TypeError("available_mask must have boolean dtype.")

    if not np.any(available_mask):
        raise ValueError("At least one action must be available.")

    return available_mask
