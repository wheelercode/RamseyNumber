"""Test Proximal Policy Optimization updates and diagnostics."""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch.distributions import Categorical

from ramsey.RColoring import RColoring
from ramsey.RGraph import RGraph
from ramsey.RProblem import RProblem
from ramsey.nn.RModel import (
    RModelConfig,
    RPairPolicyValueNetwork,
)
from ramsey.nn.RPPO import (
    RPPOConfig,
    create_optimizer,
    ppo_update,
)
from ramsey.nn.RRollout import RRolloutBatch


def make_training_batch(
    graph: RGraph,
    network: RPairPolicyValueNetwork,
    number_of_steps: int = 8,
) -> RRolloutBatch:
    """Create a small, internally consistent synthetic rollout."""

    generator = torch.Generator().manual_seed(731)

    pair_inputs = torch.randn(
        number_of_steps,
        graph.problem.n_vertices,
        graph.problem.n_vertices,
        network.config.input_size,
        generator=generator,
    )

    # Pair features represent unordered vertex pairs and should
    # therefore be symmetric across their two vertex dimensions.
    pair_inputs = 0.5 * (
        pair_inputs
        + pair_inputs.transpose(
            1,
            2,
        )
    )

    available_masks = torch.ones(
        number_of_steps,
        graph.number_of_edges,
        dtype=torch.bool,
    )

    network.eval()

    with torch.no_grad():
        logits, old_values = network(
            pair_inputs,
            available_masks,
        )

        distribution = Categorical(
            logits=logits,
        )

        actions = distribution.sample()

        old_log_probabilities = distribution.log_prob(actions)

    advantages = torch.linspace(
        -1.0,
        1.0,
        number_of_steps,
    )

    returns = old_values + torch.linspace(
        0.5,
        -0.5,
        number_of_steps,
    )

    coloring = RColoring(
        graph,
        np.zeros(
            graph.number_of_edges,
            dtype=np.uint8,
        ),
    )

    return RRolloutBatch(
        pair_inputs=pair_inputs,
        available_masks=available_masks,
        actions=actions,
        old_log_probabilities=old_log_probabilities,
        rewards=torch.zeros(
            number_of_steps,
            dtype=torch.float32,
        ),
        old_values=old_values,
        advantages=advantages,
        returns=returns,
        initial_score=10,
        final_score=8,
        best_score=7,
        final_coloring=coloring,
        best_coloring=coloring,
        terminated=False,
        truncated=True,
    )


def make_empty_batch(
    graph: RGraph,
    network: RPairPolicyValueNetwork,
) -> RRolloutBatch:
    """Create a valid rollout containing no environment steps."""

    coloring = RColoring(
        graph,
        np.zeros(
            graph.number_of_edges,
            dtype=np.uint8,
        ),
    )

    return RRolloutBatch(
        pair_inputs=torch.empty(
            (
                0,
                graph.problem.n_vertices,
                graph.problem.n_vertices,
                network.config.input_size,
            ),
            dtype=torch.float32,
        ),
        available_masks=torch.empty(
            (
                0,
                graph.number_of_edges,
            ),
            dtype=torch.bool,
        ),
        actions=torch.empty(
            0,
            dtype=torch.long,
        ),
        old_log_probabilities=torch.empty(
            0,
            dtype=torch.float32,
        ),
        rewards=torch.empty(
            0,
            dtype=torch.float32,
        ),
        old_values=torch.empty(
            0,
            dtype=torch.float32,
        ),
        advantages=torch.empty(
            0,
            dtype=torch.float32,
        ),
        returns=torch.empty(
            0,
            dtype=torch.float32,
        ),
        initial_score=0,
        final_score=0,
        best_score=0,
        final_coloring=coloring,
        best_coloring=coloring,
        terminated=True,
        truncated=False,
    )


@pytest.mark.parametrize(
    (
        "values",
        "exception_type",
    ),
    [
        (
            {
                "update_epochs": 0,
            },
            ValueError,
        ),
        (
            {
                "minibatch_size": 1.5,
            },
            TypeError,
        ),
        (
            {
                "clip_ratio": 0.0,
            },
            ValueError,
        ),
        (
            {
                "value_loss_weight": -1.0,
            },
            ValueError,
        ),
        (
            {
                "entropy_weight": -1.0,
            },
            ValueError,
        ),
        (
            {
                "maximum_gradient_norm": 0.0,
            },
            ValueError,
        ),
        (
            {
                "learning_rate": 0.0,
            },
            ValueError,
        ),
        (
            {
                "target_kl": 0.0,
            },
            ValueError,
        ),
    ],
)
def test_ppo_config_validation(
    values,
    exception_type,
) -> None:
    with pytest.raises(exception_type):
        RPPOConfig(**values)


def test_create_optimizer_uses_configured_learning_rate() -> None:
    graph = RGraph(
        RProblem.r55(
            n_vertices=7,
        )
    )

    network = RPairPolicyValueNetwork(
        graph,
        RModelConfig(
            hidden_size=8,
            number_of_layers=1,
        ),
    )

    optimizer = create_optimizer(
        network,
        RPPOConfig(
            learning_rate=2.5e-4,
        ),
    )

    assert isinstance(
        optimizer,
        torch.optim.Adam,
    )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(2.5e-4)


def test_ppo_update_changes_parameters_and_reports_metrics() -> None:
    torch.manual_seed(732)

    graph = RGraph(
        RProblem.r55(
            n_vertices=7,
        )
    )

    network = RPairPolicyValueNetwork(
        graph,
        RModelConfig(
            hidden_size=8,
            number_of_layers=1,
        ),
    )

    rollout = make_training_batch(
        graph,
        network,
    )

    config = RPPOConfig(
        update_epochs=2,
        minibatch_size=4,
        value_loss_weight=0.5,
        entropy_weight=0.01,
        learning_rate=1.0e-3,
    )

    optimizer = create_optimizer(
        network,
        config,
    )

    before = [parameter.detach().clone() for parameter in network.parameters()]

    metrics = ppo_update(
        network,
        optimizer,
        rollout,
        device="cpu",
        config=config,
    )

    after = list(network.parameters())

    assert any(
        not torch.equal(
            old_parameter,
            new_parameter,
        )
        for (
            old_parameter,
            new_parameter,
        ) in zip(
            before,
            after,
        )
    )

    assert metrics.minibatch_updates == 4

    assert metrics.epochs_completed == 2

    assert not metrics.early_stopped

    assert 0.0 <= metrics.clipped_fraction <= 1.0

    # Floating-point roundoff may produce an extremely small
    # negative number even though the estimator is theoretically
    # nonnegative.
    assert metrics.approximate_kl >= -1.0e-6

    assert metrics.gradient_norm >= 0.0

    assert all(
        math.isfinite(value)
        for value in (
            metrics.policy_loss,
            metrics.value_loss,
            metrics.entropy,
            metrics.approximate_kl,
            metrics.clipped_fraction,
            metrics.gradient_norm,
        )
    )

    metrics_dictionary = metrics.as_dict()

    assert metrics_dictionary["minibatch_updates"] == 4


def test_ppo_update_rejects_empty_rollout() -> None:
    graph = RGraph(
        RProblem.r55(
            n_vertices=7,
        )
    )

    network = RPairPolicyValueNetwork(
        graph,
        RModelConfig(
            hidden_size=8,
            number_of_layers=1,
        ),
    )

    rollout = make_empty_batch(
        graph,
        network,
    )

    config = RPPOConfig()

    with pytest.raises(
        ValueError,
        match="empty rollout",
    ):
        ppo_update(
            network,
            create_optimizer(
                network,
                config,
            ),
            rollout,
            device="cpu",
            config=config,
        )
