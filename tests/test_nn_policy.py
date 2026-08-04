"""Test the neural adapter through the core policy interface."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")


from torch.distributions import Categorical

from ramsey.RColoring import RColoring
from ramsey.REnvironment import REnvironment
from ramsey.REnvironmentConfig import (
    REnvironmentConfig,
    RTabuMemoryConfig,
)
from ramsey.REnvironmentMemory import (
    RTabuMemory,
)
from ramsey.RGraph import RGraph
from ramsey.RObjective import (
    RMonochromaticObjective,
)
from ramsey.RProblem import RProblem
from ramsey.nn.REncoding import (
    build_network_input,
)
from ramsey.nn.RModel import (
    RModelConfig,
    RPairPolicyValueNetwork,
)
from ramsey.nn.RNeuralPolicy import (
    RNeuralPolicy,
)


def make_components(
    n_vertices: int = 7,
):
    graph = RGraph(RProblem.r55(n_vertices=n_vertices))

    environment = REnvironment(
        graph=graph,
        objective=RMonochromaticObjective(),
        memory=RTabuMemory(
            graph.number_of_edges,
            RTabuMemoryConfig(
                edge_tenure=3,
                visited_state_window=20,
            ),
        ),
        config=REnvironmentConfig(max_steps=10),
    )

    coloring = RColoring(
        graph,
        np.zeros(
            graph.number_of_edges,
            dtype=np.uint8,
        ),
    )

    environment.reset(coloring)

    network = RPairPolicyValueNetwork(
        graph,
        RModelConfig(
            hidden_size=8,
            number_of_layers=1,
        ),
    )

    return (
        graph,
        environment,
        network,
    )


def test_neural_policy_returns_complete_available_decision() -> None:
    torch.manual_seed(401)

    (
        _,
        environment,
        network,
    ) = make_components()

    policy = RNeuralPolicy(
        network,
        "cpu",
        greedy=False,
    )

    version = environment.state.version

    decision = policy.evaluate(environment)

    assert environment.available_action_mask_fast()[decision.edge]

    assert np.isfinite(decision.log_probability)

    assert np.isfinite(decision.value)

    assert np.isfinite(decision.entropy)

    assert environment.state.version == version

    assert policy.select_action(environment) in environment.available_actions()

    assert policy.name == "neural-sampling"

    assert not (policy.requires_full_analysis)


def test_greedy_neural_policy_selects_largest_available_logit() -> None:
    torch.manual_seed(402)

    (
        _,
        environment,
        network,
    ) = make_components()

    environment.step(
        0,
        full_analysis=False,
    )

    available = environment.available_action_mask_fast()

    (
        pair_input,
        mask,
    ) = build_network_input(
        environment.state,
        available,
    )

    network.eval()

    with torch.no_grad():
        logits, _ = network(
            pair_input,
            mask,
        )

    expected = int(
        torch.argmax(
            logits,
            dim=-1,
        ).item()
    )

    policy = RNeuralPolicy(
        network,
        "cpu",
        greedy=True,
    )

    decision = policy.evaluate(environment)

    assert decision.edge == expected
    assert available[decision.edge]
    assert decision.edge != 0

    assert policy.name == "neural-greedy"

    distribution = Categorical(logits=logits)

    expected_log_probability = distribution.log_prob(torch.as_tensor([expected]))

    assert np.isclose(
        decision.log_probability,
        float(expected_log_probability.item()),
    )


def test_neural_policy_rejects_mismatched_environment() -> None:
    _, _, network = make_components(n_vertices=7)

    (
        _,
        other_environment,
        _,
    ) = make_components(n_vertices=8)

    policy = RNeuralPolicy(
        network,
        "cpu",
    )

    with pytest.raises(
        ValueError,
        match="do not match",
    ):
        policy.evaluate(other_environment)


def test_neural_policy_rejects_network_device_mismatch() -> None:
    (
        _,
        environment,
        network,
    ) = make_components()

    policy = RNeuralPolicy(
        network,
        "meta",
    )

    with pytest.raises(
        RuntimeError,
        match="Network is on",
    ):
        policy.evaluate(environment)


def test_neural_policy_validates_greedy_flag() -> None:
    _, _, network = make_components()

    with pytest.raises(
        TypeError,
        match="greedy",
    ):
        RNeuralPolicy(
            network,
            "cpu",
            greedy=1,
        )
