"""Conversion of Ramsey search information into neural model inputs.

Encodes an :class:`~ramsey.RState.RSearchState` (a coloring of the
host graph together with which edges are candidate actions) as dense
NumPy/PyTorch tensors suitable for the pairwise policy/value network
in :mod:`ramsey.nn.RModel`. Also provides the vertex-permutation
bookkeeping used to keep encoded edge order consistent when the host
graph's vertices are renamed.
"""

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
    """Encode one search state as a symmetric vertex-pair feature array.

    Builds a dense ``(n_vertices, n_vertices, PAIR_INPUT_CHANNELS)``
    array indexed by ordered vertex pairs ``(i, j)``, which the
    pairwise policy/value network consumes directly. Because the host
    graph is undirected, every real edge's features are written
    symmetrically at both ``[i, j]`` and ``[j, i]``; vertex-diagonal
    positions ``[k, k]`` carry no edge information but are flagged by
    ``DIAGONAL_CHANNEL`` so the network can distinguish them from real
    edges. Positions corresponding to a pair of vertices with no edge
    in the host graph (i.e. not covered by ``state.graph.edges``) are
    left at zero in every channel.

    Channels (last axis, indexed by the module constants):

    - ``EDGE_COLOR_CHANNEL`` (0): The edge's color, signed to -1.0 for
      color zero and +1.0 for color one. Zero at non-edge and diagonal
      positions.
    - ``DIAGONAL_CHANNEL`` (1): 1.0 at vertex-diagonal positions
      ``[k, k]``, 0.0 everywhere else.
    - ``AVAILABILITY_CHANNEL`` (2): 1.0 if the edge is a currently
      available action per ``available_mask``, 0.0 otherwise
      (including at non-edge and diagonal positions).

    Args:
        state (RSearchState): Search state supplying the host graph
            and current edge coloring to encode.
        available_mask (numpy.typing.NDArray[numpy.bool_]): Boolean
            array of shape ``(state.number_of_edges,)``, aligned with
            ``state.graph.edges``, where ``True`` marks an edge as a
            currently available action.

    Returns:
        numpy.typing.NDArray[numpy.float32]: Newly allocated array of
        shape ``(n_vertices, n_vertices, PAIR_INPUT_CHANNELS)`` and
        dtype ``float32``, symmetric in its first two axes.

    Raises:
        ValueError: If ``available_mask`` does not have shape
            ``(state.number_of_edges,)`` or contains no available
            actions.
        TypeError: If ``available_mask`` does not have boolean dtype.
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
    """Build one batched pair-input tensor and action-mask tensor.

    Calls :func:`encode_pair_features` to build the NumPy feature
    array, then converts it (and the action mask) to PyTorch tensors
    with an added leading batch dimension of size 1, ready to pass
    directly to :class:`~ramsey.nn.RModel.RPairPolicyValueNetwork`.

    Args:
        state (RSearchState): Search state supplying the host graph
            and current edge coloring to encode.
        available_mask (numpy.typing.NDArray[numpy.bool_]): Boolean
            array of shape ``(state.number_of_edges,)`` marking which
            edges are currently available actions.
        device (torch.device | str | None): Device on which the
            returned tensors are allocated. ``None`` uses PyTorch's
            default device placement.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A pair
        ``(pair_tensor, mask_tensor)`` where ``pair_tensor`` has shape
        ``(1, n_vertices, n_vertices, PAIR_INPUT_CHANNELS)`` and dtype
        ``torch.float32``, and ``mask_tensor`` has shape
        ``(1, number_of_edges)`` and dtype ``torch.bool``.

    Raises:
        ValueError: If ``available_mask`` does not have shape
            ``(state.number_of_edges,)`` or contains no available
            actions.
        TypeError: If ``available_mask`` does not have boolean dtype.
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
    """Map old edge indexes to new indexes after vertex renaming.

    Given a permutation of vertex labels, computes, for each original
    edge index, the index that the same edge (now between renamed
    vertices) occupies in ``graph.edges``. If the returned array is
    named ``edge_mapping``, then::

        new_values[edge_mapping] = old_values

    applies the vertex permutation to any edge-indexed array (such as
    a coloring or action mask), reordering it to match the renamed
    vertex labeling.

    Args:
        graph (RGraph): Host graph whose edge ordering defines both
            the old and new edge index spaces.
        vertex_permutation (numpy.typing.NDArray[numpy.integer]):
            Array of shape ``(n_vertices,)`` giving, for each new
            vertex label, the original vertex it corresponds to (i.e.
            a permutation of ``range(n_vertices)``).

    Returns:
        numpy.typing.NDArray[numpy.int64]: Read-only array of shape
        ``(graph.number_of_edges,)`` mapping each original edge index
        to its index after the permutation is applied.

    Raises:
        ValueError: If ``vertex_permutation`` does not have shape
            ``(n_vertices,)`` or is not a permutation of every vertex
            exactly once.
        TypeError: If ``vertex_permutation`` does not contain integers.
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
    """Return a validated boolean action mask without copying it.

    Args:
        available_mask (numpy.typing.NDArray[numpy.bool_]): Candidate
            action-availability mask to validate.
        number_of_edges (int): Expected length of ``available_mask``,
            i.e. the host graph's number of edges.

    Returns:
        numpy.typing.NDArray[numpy.bool_]: ``available_mask`` converted
        via :func:`numpy.asarray` (a view when possible, otherwise a
        copy), unchanged in content.

    Raises:
        ValueError: If ``available_mask`` does not have shape
            ``(number_of_edges,)`` or contains no available actions.
        TypeError: If ``available_mask`` does not have boolean dtype.
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
