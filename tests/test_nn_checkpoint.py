"""Test versioned neural-training checkpoints."""

from dataclasses import dataclass

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ramsey.RGraph import RGraph
from ramsey.RProblem import RProblem
from ramsey.nn.RCheckpoint import (
    CHECKPOINT_VERSION,
    load_training_checkpoint,
    save_training_checkpoint,
)
from ramsey.nn.RModel import (
    RModelConfig,
    RPairPolicyValueNetwork,
)
from ramsey.nn.RPPO import (
    RPPOConfig,
    create_optimizer,
)
from ramsey.nn.RRollout import (
    RRolloutConfig,
    RRolloutReward,
)


@dataclass(frozen=True)
class LegacyPPOConfig:
    """
    Reproduce the original combined checkpoint configuration.

    The old PPOConfig contained both rollout and optimization
    settings in one frozen dataclass.
    """

    rollout_steps: int = 128
    update_epochs: int = 3
    minibatch_size: int = 8
    discount: float = 0.99
    gae_lambda: float = 0.90
    clip_ratio: float = 0.15
    value_loss_weight: float = 0.25
    entropy_weight: float = 0.02
    maximum_gradient_norm: float = 0.75
    reward_scale: float = 20.0
    learning_rate: float = 1.0e-4


def make_components():
    """Create a small model with populated Adam optimizer state."""

    graph = RGraph(
        RProblem.r55(
            n_vertices=7,
        )
    )

    model_config = RModelConfig(
        hidden_size=8,
        number_of_layers=1,
    )

    network = RPairPolicyValueNetwork(
        graph,
        model_config,
    )

    rollout_config = RRolloutConfig(
        rollout_steps=16,
        reward_source=RRolloutReward.OBJECTIVE,
    )

    ppo_config = RPPOConfig(
        update_epochs=2,
        minibatch_size=4,
        learning_rate=1.0e-3,
    )

    optimizer = create_optimizer(
        network,
        ppo_config,
    )

    # Perform one artificial update so Adam has momentum and
    # variance tensors to save and restore.
    optimizer.zero_grad(
        set_to_none=True,
    )

    artificial_loss = sum(
        parameter.square().sum() for parameter in network.parameters()
    )

    artificial_loss.backward()

    optimizer.step()

    return (
        graph,
        network,
        optimizer,
        model_config,
        rollout_config,
        ppo_config,
    )


def test_checkpoint_round_trip_restores_training_and_rng_state(
    tmp_path,
) -> None:
    torch.manual_seed(811)

    (
        graph,
        network,
        optimizer,
        model_config,
        rollout_config,
        ppo_config,
    ) = make_components()

    rng = np.random.default_rng(812)

    saved_parameters = {
        name: value.detach().clone() for name, value in network.state_dict().items()
    }

    checkpoint_path = tmp_path / "run.pt"

    returned_path = save_training_checkpoint(
        checkpoint_path,
        network=network,
        optimizer=optimizer,
        graph=graph,
        rollout_config=rollout_config,
        ppo_config=ppo_config,
        completed_iteration=12,
        rng=rng,
        metadata={
            "run_name": "test-run",
        },
    )

    # These are the values that should appear immediately after
    # restoring the saved RNG states.
    expected_numpy = rng.integers(
        0,
        10_000,
        size=5,
    )

    expected_torch = torch.rand(5)

    # Disturb the live state before loading.
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()

    rng.random(50)
    torch.rand(50)

    restored = load_training_checkpoint(
        checkpoint_path,
        graph=graph,
        device="cpu",
        rng=rng,
    )

    assert returned_path == checkpoint_path

    assert restored.completed_iteration == 12

    assert restored.model_config == model_config

    assert restored.rollout_config == rollout_config

    assert restored.ppo_config == ppo_config

    assert restored.metadata == {
        "run_name": "test-run",
    }

    # Adam should have restored its moment estimates.
    assert restored.optimizer.state

    for (
        name,
        value,
    ) in restored.network.state_dict().items():
        assert torch.equal(
            value,
            saved_parameters[name],
        )

    actual_numpy = rng.integers(
        0,
        10_000,
        size=5,
    )

    assert np.array_equal(
        actual_numpy,
        expected_numpy,
    )

    actual_torch = torch.rand(5)

    assert torch.equal(
        actual_torch,
        expected_torch,
    )


def test_checkpoint_learning_rate_override_updates_config_and_optimizer(
    tmp_path,
) -> None:
    (
        graph,
        network,
        optimizer,
        _,
        rollout_config,
        ppo_config,
    ) = make_components()

    checkpoint_path = tmp_path / "override.pt"

    save_training_checkpoint(
        checkpoint_path,
        network=network,
        optimizer=optimizer,
        graph=graph,
        rollout_config=rollout_config,
        ppo_config=ppo_config,
        completed_iteration=2,
        rng=np.random.default_rng(813),
    )

    restored = load_training_checkpoint(
        checkpoint_path,
        graph=graph,
        device="cpu",
        rng=np.random.default_rng(814),
        new_learning_rate=2.0e-4,
    )

    assert restored.ppo_config.learning_rate == pytest.approx(2.0e-4)

    assert all(
        parameter_group["lr"] == pytest.approx(2.0e-4)
        for parameter_group in restored.optimizer.param_groups
    )


def test_checkpoint_rejects_a_different_problem(
    tmp_path,
) -> None:
    (
        graph,
        network,
        optimizer,
        _,
        rollout_config,
        ppo_config,
    ) = make_components()

    checkpoint_path = tmp_path / "problem.pt"

    save_training_checkpoint(
        checkpoint_path,
        network=network,
        optimizer=optimizer,
        graph=graph,
        rollout_config=rollout_config,
        ppo_config=ppo_config,
        completed_iteration=1,
        rng=np.random.default_rng(815),
    )

    other_graph = RGraph(
        RProblem.r55(
            n_vertices=8,
        )
    )

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        load_training_checkpoint(
            checkpoint_path,
            graph=other_graph,
            device="cpu",
            rng=np.random.default_rng(816),
        )


def test_checkpoint_rejects_unknown_version(
    tmp_path,
) -> None:
    checkpoint_path = tmp_path / "future.pt"

    torch.save(
        {
            "checkpoint_version": (CHECKPOINT_VERSION + 1),
        },
        checkpoint_path,
    )

    graph = RGraph(
        RProblem.r55(
            n_vertices=7,
        )
    )

    with pytest.raises(
        ValueError,
        match=("Unsupported checkpoint version"),
    ):
        load_training_checkpoint(
            checkpoint_path,
            graph=graph,
            device="cpu",
            rng=np.random.default_rng(817),
        )


def test_checkpoint_loads_legacy_frozen_dataclass_config(
    tmp_path,
) -> None:
    (
        graph,
        network,
        optimizer,
        model_config,
        _,
        _,
    ) = make_components()

    rng = np.random.default_rng(818)

    checkpoint_path = tmp_path / "legacy.pt"

    torch.save(
        {
            "checkpoint_version": 1,
            "iteration": 41,
            "network_state_dict": (network.state_dict()),
            "optimizer_state_dict": (optimizer.state_dict()),
            "ppo_config": (LegacyPPOConfig()),
            "network_architecture": {
                "n_vertices": (graph.problem.n_vertices),
                "input_size": (model_config.input_size),
                "hidden_size": (model_config.hidden_size),
                "number_of_layers": (model_config.number_of_layers),
                "dropout": (model_config.dropout),
            },
            "numpy_rng_state": (rng.bit_generator.state),
            "torch_rng_state": (torch.get_rng_state()),
        },
        checkpoint_path,
    )

    restored = load_training_checkpoint(
        checkpoint_path,
        graph=graph,
        device="cpu",
        rng=np.random.default_rng(819),
    )

    assert restored.completed_iteration == 41

    assert restored.rollout_config.rollout_steps == 128

    assert restored.rollout_config.reward_scale == pytest.approx(20.0)

    assert restored.ppo_config.update_epochs == 3

    assert restored.ppo_config.learning_rate == pytest.approx(1.0e-4)

    assert restored.metadata == {}


def test_checkpoint_requires_an_existing_file(
    tmp_path,
) -> None:
    graph = RGraph(
        RProblem.r55(
            n_vertices=7,
        )
    )

    with pytest.raises(FileNotFoundError):
        load_training_checkpoint(
            tmp_path / "missing.pt",
            graph=graph,
            device="cpu",
            rng=np.random.default_rng(820),
        )


@pytest.mark.parametrize(
    (
        "iteration",
        "exception_type",
    ),
    [
        (
            -1,
            ValueError,
        ),
        (
            1.5,
            TypeError,
        ),
    ],
)
def test_save_checkpoint_validates_iteration(
    tmp_path,
    iteration,
    exception_type,
) -> None:
    (
        graph,
        network,
        optimizer,
        _,
        rollout_config,
        ppo_config,
    ) = make_components()

    with pytest.raises(exception_type):
        save_training_checkpoint(
            tmp_path / "invalid.pt",
            network=network,
            optimizer=optimizer,
            graph=graph,
            rollout_config=rollout_config,
            ppo_config=ppo_config,
            completed_iteration=iteration,
            rng=np.random.default_rng(821),
        )
