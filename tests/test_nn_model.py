"""Test the permutation-equivariant pair policy/value model."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")


from ramsey.RColoring import RColoring
from ramsey.RGraph import RGraph
from ramsey.RProblem import RProblem
from ramsey.RState import RSearchState
from ramsey.nn.REncoding import (
    build_network_input,
    edge_permutation_indices,
)
from ramsey.nn.RModel import (
    RModelConfig,
    RPairMessageLayer,
    RPairPolicyValueNetwork,
)


def make_input(
    seed: int = 311,
):
    graph = RGraph(RProblem.r55(n_vertices=7))

    colors = np.random.default_rng(seed).integers(
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
            2,
            6,
        ]
    ] = False

    (
        pair_input,
        mask,
    ) = build_network_input(
        state,
        available,
    )

    return (
        graph,
        colors,
        available,
        pair_input,
        mask,
    )


@pytest.mark.parametrize(
    (
        "values",
        "exception_type",
    ),
    [
        (
            {"input_size": 0},
            ValueError,
        ),
        (
            {"hidden_size": 0},
            ValueError,
        ),
        (
            {"number_of_layers": -1},
            ValueError,
        ),
        (
            {"hidden_size": 4.5},
            TypeError,
        ),
        (
            {"dropout": -0.1},
            ValueError,
        ),
        (
            {"dropout": 1.0},
            ValueError,
        ),
    ],
)
def test_model_config_validation(
    values,
    exception_type,
) -> None:
    with pytest.raises(exception_type):
        RModelConfig(**values)


def test_message_layer_explicitly_restores_pair_symmetry() -> None:
    torch.manual_seed(312)

    layer = RPairMessageLayer(hidden_size=8)

    pair_features = torch.randn(
        2,
        7,
        7,
        8,
    )

    output = layer(pair_features)

    assert output.shape == pair_features.shape

    assert torch.allclose(
        output,
        output.transpose(
            1,
            2,
        ),
        atol=1.0e-6,
    )


def test_model_output_shapes_masking_and_finiteness() -> None:
    torch.manual_seed(313)

    (
        graph,
        _,
        available,
        pair_input,
        mask,
    ) = make_input()

    model = RPairPolicyValueNetwork(
        graph,
        RModelConfig(
            hidden_size=16,
            number_of_layers=2,
        ),
    )

    logits, values = model(
        pair_input,
        mask,
    )

    assert logits.shape == (
        1,
        graph.number_of_edges,
    )

    assert values.shape == (1,)

    assert torch.isfinite(values).all()

    assert torch.isfinite(logits[mask]).all()

    assert torch.all(logits[~mask] == model.unavailable_logit)

    assert model.trainable_parameter_count == sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    assert np.array_equal(
        model.edge_vertices.cpu().numpy(),
        graph.edges,
    )


def test_model_accepts_batches() -> None:
    (
        graph,
        _,
        _,
        pair_input,
        mask,
    ) = make_input(seed=314)

    model = RPairPolicyValueNetwork(
        graph,
        RModelConfig(
            hidden_size=8,
            number_of_layers=1,
        ),
    )

    batch_input = pair_input.repeat(
        3,
        1,
        1,
        1,
    )

    batch_mask = mask.repeat(
        3,
        1,
    )

    logits, values = model(
        batch_input,
        batch_mask,
    )

    assert logits.shape == (
        3,
        graph.number_of_edges,
    )

    assert values.shape == (3,)


def test_model_is_equivariant_to_vertex_renaming() -> None:
    torch.manual_seed(315)

    (
        graph,
        colors,
        available,
        original_input,
        original_mask,
    ) = make_input(seed=316)

    model = RPairPolicyValueNetwork(
        graph,
        RModelConfig(
            hidden_size=16,
            number_of_layers=2,
            dropout=0.0,
        ),
    )

    model.eval()

    permutation = np.asarray(
        [
            4,
            1,
            6,
            0,
            3,
            5,
            2,
        ],
        dtype=np.int64,
    )

    edge_mapping = edge_permutation_indices(
        graph,
        permutation,
    )

    edge_mapping_tensor = torch.as_tensor(
        edge_mapping.copy(),
        dtype=torch.long,
        device=original_input.device,
    )

    permuted_colors = np.empty_like(colors)

    permuted_colors[edge_mapping] = colors

    permuted_available = np.empty_like(available)

    permuted_available[edge_mapping] = available

    permuted_state = RSearchState(
        RColoring(
            graph,
            permuted_colors,
        )
    )

    (
        permuted_input,
        permuted_mask,
    ) = build_network_input(
        permuted_state,
        permuted_available,
    )

    with torch.no_grad():
        (
            original_logits,
            original_value,
        ) = model(
            original_input,
            original_mask,
        )

        (
            permuted_logits,
            permuted_value,
        ) = model(
            permuted_input,
            permuted_mask,
        )

    assert torch.allclose(
        original_logits[0],
        permuted_logits[
            0,
            edge_mapping_tensor,
        ],
        atol=1.0e-5,
        rtol=1.0e-5,
    )

    assert torch.allclose(
        original_value,
        permuted_value,
        atol=1.0e-5,
        rtol=1.0e-5,
    )


def test_model_validates_input_shapes_and_mask_dtype() -> None:
    (
        graph,
        _,
        _,
        pair_input,
        mask,
    ) = make_input(seed=317)

    model = RPairPolicyValueNetwork(
        graph,
        RModelConfig(
            hidden_size=8,
            number_of_layers=0,
        ),
    )

    with pytest.raises(
        ValueError,
        match="pair_input",
    ):
        model(
            pair_input[:, :-1],
            mask,
        )

    with pytest.raises(
        ValueError,
        match="available_mask",
    ):
        model(
            pair_input,
            mask[:, :-1],
        )

    with pytest.raises(
        TypeError,
        match="boolean",
    ):
        model(
            pair_input,
            mask.float(),
        )
