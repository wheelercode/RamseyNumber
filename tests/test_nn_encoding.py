"""Test conversion of search states into neural pair inputs."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")


from ramsey.RColoring import RColoring
from ramsey.RGraph import RGraph
from ramsey.RProblem import RProblem
from ramsey.RState import RSearchState
from ramsey.nn.REncoding import (
    AVAILABILITY_CHANNEL,
    DIAGONAL_CHANNEL,
    EDGE_COLOR_CHANNEL,
    build_network_input,
    edge_permutation_indices,
    encode_pair_features,
)


@pytest.fixture(scope="module")
def encoded_state():
    graph = RGraph(RProblem.r55(n_vertices=7))

    colors = np.random.default_rng(301).integers(
        0,
        2,
        size=graph.number_of_edges,
        dtype=np.uint8,
    )

    state = RSearchState(
        RColoring(
            graph,
            colors,
        )
    )

    available = np.ones(
        graph.number_of_edges,
        dtype=np.bool_,
    )

    available[
        [
            1,
            4,
            8,
        ]
    ] = False

    return (
        graph,
        state,
        available,
    )


def test_pair_features_have_exact_channels(
    encoded_state,
) -> None:
    (
        graph,
        state,
        available,
    ) = encoded_state

    features = encode_pair_features(
        state,
        available,
    )

    n_vertices = graph.problem.n_vertices

    i = graph.edges[:, 0].astype(np.int64)

    j = graph.edges[:, 1].astype(np.int64)

    assert features.shape == (
        n_vertices,
        n_vertices,
        3,
    )

    assert features.dtype == np.float32

    assert np.array_equal(
        features,
        features.transpose(
            1,
            0,
            2,
        ),
    )

    assert np.array_equal(
        features[
            i,
            j,
            EDGE_COLOR_CHANNEL,
        ],
        (state.colors.astype(np.float32) * 2.0 - 1.0),
    )

    assert np.array_equal(
        features[
            i,
            j,
            AVAILABILITY_CHANNEL,
        ],
        available.astype(np.float32),
    )

    assert np.array_equal(
        np.diag(
            features[
                :,
                :,
                DIAGONAL_CHANNEL,
            ]
        ),
        np.ones(
            n_vertices,
            dtype=np.float32,
        ),
    )

    assert (
        np.count_nonzero(
            features[
                :,
                :,
                DIAGONAL_CHANNEL,
            ]
        )
        == n_vertices
    )


def test_build_network_input_adds_batch_and_device(
    encoded_state,
) -> None:
    (
        graph,
        state,
        available,
    ) = encoded_state

    (
        pair_tensor,
        mask_tensor,
    ) = build_network_input(
        state,
        available,
        device="cpu",
    )

    assert pair_tensor.shape == (
        1,
        7,
        7,
        3,
    )

    assert mask_tensor.shape == (
        1,
        graph.number_of_edges,
    )

    assert pair_tensor.dtype == torch.float32

    assert mask_tensor.dtype == torch.bool

    assert pair_tensor.device.type == "cpu"

    assert np.array_equal(
        mask_tensor[0].numpy(),
        available,
    )


@pytest.mark.parametrize(
    (
        "available",
        "exception_type",
    ),
    [
        (
            np.ones(
                20,
                dtype=np.bool_,
            ),
            ValueError,
        ),
        (
            np.ones(
                21,
                dtype=np.uint8,
            ),
            TypeError,
        ),
        (
            np.zeros(
                21,
                dtype=np.bool_,
            ),
            ValueError,
        ),
    ],
)
def test_encoding_validates_available_mask(
    encoded_state,
    available,
    exception_type,
) -> None:
    _, state, _ = encoded_state

    with pytest.raises(exception_type):
        encode_pair_features(
            state,
            available,
        )


def test_edge_permutation_mapping_matches_renamed_endpoints(
    encoded_state,
) -> None:
    graph, _, _ = encoded_state

    permutation = np.asarray(
        [
            3,
            0,
            6,
            2,
            1,
            5,
            4,
        ],
        dtype=np.int64,
    )

    mapping = edge_permutation_indices(
        graph,
        permutation,
    )

    edge_lookup = {
        tuple(map(int, endpoints)): edge for edge, endpoints in enumerate(graph.edges)
    }

    for old_edge, (
        i,
        j,
    ) in enumerate(graph.edges):
        renamed = sorted(
            (
                permutation[int(i)],
                permutation[int(j)],
            )
        )

        assert mapping[old_edge] == (edge_lookup[tuple(renamed)])

    assert sorted(mapping.tolist()) == list(range(graph.number_of_edges))

    assert not mapping.flags.writeable


@pytest.mark.parametrize(
    "permutation",
    [
        np.asarray(
            [
                0,
                1,
                2,
            ],
            dtype=np.int64,
        ),
        np.asarray(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                5,
            ],
            dtype=np.int64,
        ),
    ],
)
def test_edge_permutation_rejects_invalid_permutations(
    encoded_state,
    permutation,
) -> None:
    graph, _, _ = encoded_state

    with pytest.raises(ValueError):
        edge_permutation_indices(
            graph,
            permutation,
        )
