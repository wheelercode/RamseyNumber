"""Neural rollout collection and generalized advantage calculation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray
import torch
from torch.distributions import Categorical

from ..RColoring import RColoring
from ..REnvironment import REnvironment
from .REncoding import build_network_input
from .RModel import RPairPolicyValueNetwork


class RRolloutReward(str, Enum):
    """
    State-transition quantity used as the training reward.
    """

    EXACT_SCORE = "exact-score"
    OBJECTIVE = "objective"


@dataclass(frozen=True, slots=True)
class RRolloutConfig:
    """
    Immutable settings for one policy rollout.
    """

    rollout_steps: int = 256

    discount: float = 0.995
    gae_lambda: float = 0.95

    reward_scale: float = 10.0

    reward_source: RRolloutReward = RRolloutReward.EXACT_SCORE

    normalize_advantages: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.rollout_steps, bool) or not isinstance(
            self.rollout_steps,
            Integral,
        ):
            raise TypeError("rollout_steps must be an integer.")

        rollout_steps = int(self.rollout_steps)

        if rollout_steps <= 0:
            raise ValueError("rollout_steps must be positive.")

        object.__setattr__(
            self,
            "rollout_steps",
            rollout_steps,
        )

        for name in (
            "discount",
            "gae_lambda",
        ):
            value = getattr(
                self,
                name,
            )

            if isinstance(value, bool) or not isinstance(
                value,
                Real,
            ):
                raise TypeError(f"{name} must be numeric.")

            value = float(value)

            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between " "zero and one.")

            object.__setattr__(
                self,
                name,
                value,
            )

        if isinstance(self.reward_scale, bool) or not isinstance(
            self.reward_scale,
            Real,
        ):
            raise TypeError("reward_scale must be numeric.")

        reward_scale = float(self.reward_scale)

        if reward_scale <= 0.0:
            raise ValueError("reward_scale must be positive.")

        object.__setattr__(
            self,
            "reward_scale",
            reward_scale,
        )

        try:
            reward_source = RRolloutReward(self.reward_source)
        except ValueError as error:
            raise ValueError(
                "Unknown reward source: " f"{self.reward_source}"
            ) from error

        object.__setattr__(
            self,
            "reward_source",
            reward_source,
        )

        if not isinstance(
            self.normalize_advantages,
            bool,
        ):
            raise TypeError("normalize_advantages must " "be boolean.")


@dataclass(frozen=True, slots=True)
class RRolloutBatch:
    """
    CPU-resident experiences and summary data from one rollout.
    """

    pair_inputs: torch.Tensor
    available_masks: torch.Tensor

    actions: torch.Tensor
    old_log_probabilities: torch.Tensor

    rewards: torch.Tensor
    old_values: torch.Tensor

    advantages: torch.Tensor
    returns: torch.Tensor

    initial_score: int
    final_score: int
    best_score: int

    final_coloring: RColoring
    best_coloring: RColoring

    terminated: bool
    truncated: bool

    def __post_init__(self) -> None:
        number_of_steps = len(self.actions)

        step_tensors = (
            self.pair_inputs,
            self.available_masks,
            self.old_log_probabilities,
            self.rewards,
            self.old_values,
            self.advantages,
            self.returns,
        )

        if any(len(tensor) != number_of_steps for tensor in step_tensors):
            raise ValueError("Every rollout tensor must " "have equal length.")

    @property
    def number_of_steps(self) -> int:
        return len(self.actions)

    @property
    def total_scaled_reward(self) -> float:
        return float(self.rewards.sum().item())


def calculate_advantages(
    rewards: NDArray[np.floating],
    values: NDArray[np.floating],
    terminated: NDArray[np.bool_],
    last_value: float,
    discount: float,
    gae_lambda: float,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float32],
]:
    """
    Calculate generalized advantages and value-return targets.

    A true terminal transition prevents value bootstrapping.
    A time-limit truncation does not.
    """
    rewards = np.asarray(
        rewards,
        dtype=np.float32,
    )

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    terminated = np.asarray(terminated)

    if rewards.ndim != 1 or values.ndim != 1 or terminated.ndim != 1:
        raise ValueError("Rollout arrays must be " "one-dimensional.")

    if not (len(rewards) == len(values) == len(terminated)):
        raise ValueError("Rollout arrays must have " "equal lengths.")

    if terminated.dtype != np.bool_:
        raise TypeError("terminated must have boolean dtype.")

    if not 0.0 <= discount <= 1.0:
        raise ValueError("discount must be between " "zero and one.")

    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be between " "zero and one.")

    advantages = np.zeros(
        len(rewards),
        dtype=np.float32,
    )

    running_advantage = 0.0

    for step in reversed(range(len(rewards))):
        if step == len(rewards) - 1:
            next_value = last_value
        else:
            next_value = float(values[step + 1])

        can_continue = 0.0 if terminated[step] else 1.0

        prediction_error = (
            rewards[step] + discount * next_value * can_continue - values[step]
        )

        running_advantage = (
            prediction_error + discount * gae_lambda * can_continue * running_advantage
        )

        advantages[step] = running_advantage

    returns = advantages + values

    return (
        advantages,
        returns.astype(
            np.float32,
            copy=False,
        ),
    )


def collect_rollout(
    network: RPairPolicyValueNetwork,
    environment: REnvironment,
    coloring: RColoring,
    device: torch.device | str,
    config: RRolloutConfig,
) -> RRolloutBatch:
    """
    Collect one rollout from an explicit seed coloring.

    The returned tensors remain on the CPU. Minibatches are moved to
    the training device later by PPO.
    """
    device = _resolve_device(device)

    _require_network_device(
        network,
        device,
    )

    _require_compatible_environment(
        network,
        environment,
    )

    environment.reset(coloring)

    initial_score = environment.state.score

    pair_inputs: list[np.ndarray] = []

    available_masks: list[np.ndarray] = []

    actions: list[int] = []
    log_probabilities: list[float] = []
    rewards: list[float] = []
    values: list[float] = []
    terminated_flags: list[bool] = []

    network.eval()

    for _ in range(config.rollout_steps):
        if environment.terminated or environment.truncated:
            break

        available_actions = environment.available_action_mask_fast()

        (
            pair_input,
            available_mask,
        ) = build_network_input(
            environment.state,
            available_actions,
            device=device,
        )

        with torch.no_grad():
            logits, value = network(
                pair_input,
                available_mask,
            )

            distribution = Categorical(logits=logits)

            action_tensor = distribution.sample()

            log_probability = distribution.log_prob(action_tensor)

        action = int(action_tensor.item())

        pair_inputs.append(pair_input[0].detach().cpu().numpy().copy())

        available_masks.append(available_mask[0].detach().cpu().numpy().copy())

        actions.append(action)

        log_probabilities.append(float(log_probability.item()))

        values.append(float(value.item()))

        result = environment.step(
            action,
            full_analysis=False,
        )

        if config.reward_source is RRolloutReward.EXACT_SCORE:
            raw_reward = float(result.immediate_reward)
        else:
            raw_reward = float(result.objective_reward)

        rewards.append(raw_reward / config.reward_scale)

        terminated_flags.append(result.terminated)

        if result.terminated or result.truncated:
            break

    if environment.terminated:
        last_value = 0.0
    else:
        final_available = environment.available_action_mask_fast()

        (
            final_input,
            final_mask,
        ) = build_network_input(
            environment.state,
            final_available,
            device=device,
        )

        with torch.no_grad():
            _, final_value = network(
                final_input,
                final_mask,
            )

        last_value = float(final_value.item())

    reward_array = np.asarray(
        rewards,
        dtype=np.float32,
    )

    value_array = np.asarray(
        values,
        dtype=np.float32,
    )

    terminated_array = np.asarray(
        terminated_flags,
        dtype=np.bool_,
    )

    (
        advantages,
        returns,
    ) = calculate_advantages(
        rewards=reward_array,
        values=value_array,
        terminated=terminated_array,
        last_value=last_value,
        discount=config.discount,
        gae_lambda=config.gae_lambda,
    )

    if config.normalize_advantages and advantages.size > 0:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)

    n_vertices = environment.graph.problem.n_vertices

    number_of_edges = environment.graph.number_of_edges

    if pair_inputs:
        pair_tensor = torch.as_tensor(
            np.stack(pair_inputs),
            dtype=torch.float32,
        )

        mask_tensor = torch.as_tensor(
            np.stack(available_masks),
            dtype=torch.bool,
        )
    else:
        pair_tensor = torch.empty(
            (
                0,
                n_vertices,
                n_vertices,
                network.config.input_size,
            ),
            dtype=torch.float32,
        )

        mask_tensor = torch.empty(
            (
                0,
                number_of_edges,
            ),
            dtype=torch.bool,
        )

    return RRolloutBatch(
        pair_inputs=pair_tensor,
        available_masks=mask_tensor,
        actions=torch.as_tensor(
            actions,
            dtype=torch.long,
        ),
        old_log_probabilities=(
            torch.as_tensor(
                log_probabilities,
                dtype=torch.float32,
            )
        ),
        rewards=torch.as_tensor(
            reward_array,
            dtype=torch.float32,
        ),
        old_values=torch.as_tensor(
            value_array,
            dtype=torch.float32,
        ),
        advantages=torch.as_tensor(
            advantages,
            dtype=torch.float32,
        ),
        returns=torch.as_tensor(
            returns,
            dtype=torch.float32,
        ),
        initial_score=initial_score,
        final_score=environment.state.score,
        best_score=environment.best_score,
        final_coloring=environment.state.coloring_snapshot(),
        best_coloring=environment.best_coloring,
        terminated=environment.terminated,
        truncated=environment.truncated,
    )


def _require_compatible_environment(
    network: RPairPolicyValueNetwork,
    environment: REnvironment,
) -> None:
    graph = environment.graph

    if (
        graph.problem.n_vertices != network.n_vertices
        or graph.number_of_edges != network.number_of_edges
    ):
        raise ValueError("Environment graph dimensions " "do not match network.")


def _require_network_device(
    network: RPairPolicyValueNetwork,
    device: torch.device,
) -> None:
    try:
        actual_device = next(network.parameters()).device
    except StopIteration:
        actual_device = network.edge_vertices.device

    if actual_device != device:
        raise RuntimeError(
            f"Network is on {actual_device}, " "but rollout device is " f"{device}."
        )


def _resolve_device(
    device: torch.device | str,
) -> torch.device:
    """
    Resolve an unindexed CUDA device to the current CUDA device.

    For example, torch.device("cuda") becomes
    torch.device("cuda:0") when CUDA device zero is current.
    """
    resolved_device = torch.device(device)

    if (
        resolved_device.type == "cuda"
        and resolved_device.index is None
        and torch.cuda.is_available()
    ):
        resolved_device = torch.device(
            "cuda",
            torch.cuda.current_device(),
        )

    return resolved_device
