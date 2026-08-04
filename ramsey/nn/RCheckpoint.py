"""Versioned persistence and restoration of neural training state."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..RGraph import RGraph
from .RModel import (
    RModelConfig,
    RPairPolicyValueNetwork,
)
from .RPPO import (
    RPPOConfig,
    create_optimizer,
)
from .RRollout import (
    RRolloutConfig,
    RRolloutReward,
)

CHECKPOINT_VERSION = 2


@dataclass(frozen=True, slots=True)
class RTrainingCheckpoint:
    """
    Restored model, optimizer, configurations, and run position.

    This object collects everything a training loop needs to
    continue from a saved checkpoint.
    """

    network: RPairPolicyValueNetwork
    optimizer: torch.optim.Optimizer
    model_config: RModelConfig
    rollout_config: RRolloutConfig
    ppo_config: RPPOConfig
    completed_iteration: int
    metadata: dict[str, Any]


def save_training_checkpoint(
    checkpoint_path: str | Path,
    *,
    network: RPairPolicyValueNetwork,
    optimizer: torch.optim.Optimizer,
    graph: RGraph,
    rollout_config: RRolloutConfig,
    ppo_config: RPPOConfig,
    completed_iteration: int,
    rng: np.random.Generator,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """
    Atomically save everything required to resume training.

    The temporary file is written first and then renamed over the
    destination. This avoids leaving a partially written checkpoint
    if saving is interrupted.
    """

    if isinstance(
        completed_iteration,
        bool,
    ) or not isinstance(
        completed_iteration,
        Integral,
    ):
        raise TypeError("completed_iteration must be an integer.")

    completed_iteration = int(completed_iteration)

    if completed_iteration < 0:
        raise ValueError("completed_iteration cannot be negative.")

    if (
        graph.problem.n_vertices != network.n_vertices
        or graph.number_of_edges != network.number_of_edges
    ):
        raise ValueError("Graph dimensions do not match network.")

    if not isinstance(
        rng,
        np.random.Generator,
    ):
        raise TypeError("rng must be a NumPy Generator.")

    if metadata is not None and not isinstance(
        metadata,
        dict,
    ):
        raise TypeError("metadata must be a dictionary or None.")

    checkpoint_path = Path(checkpoint_path)

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rollout_values = asdict(rollout_config)

    # Store the enum as a plain string so the serialized format
    # does not unnecessarily depend on the Python enum object.
    rollout_values["reward_source"] = rollout_config.reward_source.value

    checkpoint: dict[str, Any] = {
        "checkpoint_version": (CHECKPOINT_VERSION),
        "completed_iteration": (completed_iteration),
        "problem": {
            "n_vertices": (graph.problem.n_vertices),
            "forbidden_clique_sizes": (graph.problem.forbidden_clique_sizes),
        },
        "model_config": asdict(network.config),
        "rollout_config": (rollout_values),
        "ppo_config": asdict(ppo_config),
        "network_state_dict": (network.state_dict()),
        "optimizer_state_dict": (optimizer.state_dict()),
        "numpy_rng_state": (rng.bit_generator.state),
        "torch_rng_state": (torch.get_rng_state()),
        "metadata": dict(metadata or {}),
    }

    if torch.cuda.is_available():
        checkpoint["cuda_rng_states"] = torch.cuda.get_rng_state_all()

    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")

    torch.save(
        checkpoint,
        temporary_path,
    )

    temporary_path.replace(checkpoint_path)

    return checkpoint_path


def load_training_checkpoint(
    checkpoint_path: str | Path,
    *,
    graph: RGraph,
    device: torch.device | str,
    rng: np.random.Generator,
    new_learning_rate: float | None = None,
) -> RTrainingCheckpoint:
    """
    Restore a current or legacy training checkpoint.

    Parameters
    ----------
    checkpoint_path:
        Path to a trusted local PyTorch checkpoint.

    graph:
        Graph scaffolding used to rebuild the network.

    device:
        Device on which the restored network and optimizer should
        operate.

    rng:
        Existing NumPy generator whose saved state will be restored.

    new_learning_rate:
        Optional learning-rate override. It is applied after loading
        the optimizer state because optimizer.load_state_dict()
        restores the old learning rate.
    """

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    if not isinstance(
        rng,
        np.random.Generator,
    ):
        raise TypeError("rng must be a NumPy Generator.")

    checkpoint = _trusted_torch_load(checkpoint_path)

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise TypeError("Checkpoint contents must be a dictionary.")

    version = int(
        checkpoint.get(
            "checkpoint_version",
            1,
        )
    )

    if version == 1:
        restored_configuration = _read_legacy_configuration(checkpoint)
    elif version == CHECKPOINT_VERSION:
        restored_configuration = _read_current_configuration(checkpoint)
    else:
        raise ValueError("Unsupported checkpoint version: " f"{version}")

    (
        model_config,
        rollout_config,
        ppo_config,
        completed_iteration,
    ) = restored_configuration

    _require_matching_problem(
        checkpoint,
        graph,
        version,
    )

    if new_learning_rate is not None:
        if isinstance(
            new_learning_rate,
            bool,
        ) or not isinstance(
            new_learning_rate,
            Real,
        ):
            raise TypeError("new_learning_rate must be " "numeric or None.")

        new_learning_rate = float(new_learning_rate)

        if new_learning_rate <= 0.0:
            raise ValueError("new_learning_rate must be positive.")

        ppo_values = asdict(ppo_config)

        ppo_values["learning_rate"] = new_learning_rate

        ppo_config = RPPOConfig(**ppo_values)

    device = torch.device(device)

    network = RPairPolicyValueNetwork(
        graph,
        model_config,
    ).to(device)

    network.load_state_dict(checkpoint["network_state_dict"])

    optimizer = create_optimizer(
        network,
        ppo_config,
    )

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    _move_optimizer_state(
        optimizer,
        device,
    )

    # optimizer.load_state_dict() restores the learning rate that
    # was saved in the checkpoint. Therefore, an explicit override
    # must be applied after the optimizer state is loaded.
    if new_learning_rate is not None:
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = new_learning_rate

    rng.bit_generator.state = checkpoint["numpy_rng_state"]

    torch.set_rng_state(checkpoint["torch_rng_state"])

    if torch.cuda.is_available() and "cuda_rng_states" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])

    return RTrainingCheckpoint(
        network=network,
        optimizer=optimizer,
        model_config=model_config,
        rollout_config=rollout_config,
        ppo_config=ppo_config,
        completed_iteration=(completed_iteration),
        metadata=dict(
            checkpoint.get(
                "metadata",
                {},
            )
        ),
    )


def _read_current_configuration(
    checkpoint: dict[str, Any],
) -> tuple[
    RModelConfig,
    RRolloutConfig,
    RPPOConfig,
    int,
]:
    """Read configuration values from a version-two checkpoint."""

    return (
        RModelConfig(**dict(checkpoint["model_config"])),
        RRolloutConfig(**dict(checkpoint["rollout_config"])),
        RPPOConfig(**dict(checkpoint["ppo_config"])),
        int(checkpoint["completed_iteration"]),
    )


def _read_legacy_configuration(
    checkpoint: dict[str, Any],
) -> tuple[
    RModelConfig,
    RRolloutConfig,
    RPPOConfig,
    int,
]:
    """
    Convert the original combined PPO configuration.

    Older checkpoints stored rollout and optimization settings in
    one PPOConfig object. That object could be either a dictionary
    or a frozen dataclass.
    """

    legacy_config = checkpoint["ppo_config"]

    if hasattr(
        legacy_config,
        "__dataclass_fields__",
    ):
        legacy_values = asdict(legacy_config)
    else:
        legacy_values = dict(legacy_config)

    architecture = dict(checkpoint["network_architecture"])

    model_config = RModelConfig(
        input_size=architecture.get(
            "input_size",
            3,
        ),
        hidden_size=architecture.get(
            "hidden_size",
            64,
        ),
        number_of_layers=(
            architecture.get(
                "number_of_layers",
                4,
            )
        ),
        dropout=architecture.get(
            "dropout",
            0.0,
        ),
    )

    rollout_config = RRolloutConfig(
        rollout_steps=legacy_values.get(
            "rollout_steps",
            256,
        ),
        discount=legacy_values.get(
            "discount",
            0.995,
        ),
        gae_lambda=legacy_values.get(
            "gae_lambda",
            0.95,
        ),
        reward_scale=legacy_values.get(
            "reward_scale",
            10.0,
        ),
        reward_source=(RRolloutReward.EXACT_SCORE),
        normalize_advantages=True,
    )

    ppo_config = RPPOConfig(
        update_epochs=legacy_values.get(
            "update_epochs",
            4,
        ),
        minibatch_size=legacy_values.get(
            "minibatch_size",
            16,
        ),
        clip_ratio=legacy_values.get(
            "clip_ratio",
            0.20,
        ),
        value_loss_weight=(
            legacy_values.get(
                "value_loss_weight",
                0.0,
            )
        ),
        entropy_weight=(
            legacy_values.get(
                "entropy_weight",
                0.0,
            )
        ),
        maximum_gradient_norm=(
            legacy_values.get(
                "maximum_gradient_norm",
                0.50,
            )
        ),
        learning_rate=legacy_values.get(
            "learning_rate",
            5.0e-4,
        ),
    )

    return (
        model_config,
        rollout_config,
        ppo_config,
        int(checkpoint["iteration"]),
    )


def _require_matching_problem(
    checkpoint: dict[str, Any],
    graph: RGraph,
    version: int,
) -> None:
    """Require the supplied graph to match the saved problem."""

    if version == 1:
        expected_vertices = int(checkpoint["network_architecture"]["n_vertices"])

        if graph.problem.n_vertices != expected_vertices:
            raise ValueError("Checkpoint problem does not match graph.")

        return

    problem = dict(checkpoint["problem"])

    expected_clique_sizes = tuple(problem["forbidden_clique_sizes"])

    if graph.problem.n_vertices != int(problem["n_vertices"]) or (
        graph.problem.forbidden_clique_sizes != expected_clique_sizes
    ):
        raise ValueError("Checkpoint problem does not match graph.")


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    """Move every tensor stored by the optimizer to one device."""

    for optimizer_state in optimizer.state.values():
        for key, value in optimizer_state.items():
            if torch.is_tensor(value):
                optimizer_state[key] = value.to(device)


def _trusted_torch_load(
    checkpoint_path: Path,
) -> dict[str, Any]:
    """
    Load a trusted local checkpoint across PyTorch versions.

    weights_only=False is intentional because the checkpoint contains
    Python and NumPy RNG metadata. Do not use this function on an
    untrusted checkpoint file.
    """

    try:
        return torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        # PyTorch versions predating the weights_only parameter.
        return torch.load(
            checkpoint_path,
            map_location="cpu",
        )
