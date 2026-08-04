"""Proximal Policy Optimization configuration and parameter updates."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .RModel import RPairPolicyValueNetwork
from .RRollout import RRolloutBatch


@dataclass(frozen=True, slots=True)
class RPPOConfig:
    """Immutable hyperparameters for PPO parameter updates."""

    update_epochs: int = 4
    minibatch_size: int = 16
    clip_ratio: float = 0.20
    value_loss_weight: float = 0.0
    entropy_weight: float = 0.0
    maximum_gradient_norm: float = 0.50
    learning_rate: float = 5.0e-4
    target_kl: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "update_epochs",
            "minibatch_size",
        ):
            value = getattr(self, name)

            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer.")

            value = int(value)

            if value <= 0:
                raise ValueError(f"{name} must be positive.")

            object.__setattr__(
                self,
                name,
                value,
            )

        numeric_fields = (
            "clip_ratio",
            "value_loss_weight",
            "entropy_weight",
            "maximum_gradient_norm",
            "learning_rate",
        )

        for name in numeric_fields:
            value = getattr(self, name)

            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be numeric.")

            value = float(value)

            if value < 0.0:
                raise ValueError(f"{name} cannot be negative.")

            object.__setattr__(
                self,
                name,
                value,
            )

        if self.clip_ratio <= 0.0:
            raise ValueError("clip_ratio must be positive.")

        if self.maximum_gradient_norm <= 0.0:
            raise ValueError("maximum_gradient_norm must be positive.")

        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")

        if self.target_kl is not None:
            if isinstance(self.target_kl, bool) or not isinstance(
                self.target_kl,
                Real,
            ):
                raise TypeError("target_kl must be numeric or None.")

            target_kl = float(self.target_kl)

            if target_kl <= 0.0:
                raise ValueError("target_kl must be positive.")

            object.__setattr__(
                self,
                "target_kl",
                target_kl,
            )


@dataclass(frozen=True, slots=True)
class RPPOMetrics:
    """Aggregated diagnostics from one PPO update."""

    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clipped_fraction: float
    gradient_norm: float
    minibatch_updates: int
    epochs_completed: int
    early_stopped: bool

    def as_dict(
        self,
    ) -> dict[str, float | int | bool]:
        """Return metrics in a logging-friendly dictionary."""

        return {
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "entropy": self.entropy,
            "approximate_kl": self.approximate_kl,
            "clipped_fraction": self.clipped_fraction,
            "gradient_norm": self.gradient_norm,
            "minibatch_updates": self.minibatch_updates,
            "epochs_completed": self.epochs_completed,
            "early_stopped": self.early_stopped,
        }


def create_optimizer(
    network: RPairPolicyValueNetwork,
    config: RPPOConfig,
) -> torch.optim.Adam:
    """Create the standard Adam optimizer used by PPO."""

    return torch.optim.Adam(
        network.parameters(),
        lr=config.learning_rate,
    )


def ppo_update(
    network: RPairPolicyValueNetwork,
    optimizer: torch.optim.Optimizer,
    rollout: RRolloutBatch,
    device: torch.device | str,
    config: RPPOConfig,
) -> RPPOMetrics:
    """
    Train the policy/value network using one collected rollout.

    The rollout remains CPU-resident while the tensors required for
    this update are copied to the requested training device.
    """

    number_of_samples = rollout.number_of_steps

    if number_of_samples == 0:
        raise ValueError("Cannot perform a PPO update " "with an empty rollout.")

    device = torch.device(device)

    pair_inputs = rollout.pair_inputs.to(device)

    available_masks = rollout.available_masks.to(device)

    actions = rollout.actions.to(device)

    old_log_probabilities = rollout.old_log_probabilities.to(device)

    advantages = rollout.advantages.to(device)

    returns = rollout.returns.to(device)

    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    approximate_kls: list[float] = []
    clipped_fractions: list[float] = []
    gradient_norms: list[float] = []

    epochs_completed = 0
    early_stopped = False

    network.train()

    for _ in range(config.update_epochs):
        epoch_approximate_kls: list[float] = []

        permutation = torch.randperm(
            number_of_samples,
            device=device,
        )

        for start in range(
            0,
            number_of_samples,
            config.minibatch_size,
        ):
            indices = permutation[start : start + config.minibatch_size]

            logits, predicted_values = network(
                pair_inputs[indices],
                available_masks[indices],
            )

            distribution = Categorical(
                logits=logits,
            )

            new_log_probabilities = distribution.log_prob(actions[indices])

            entropy = distribution.entropy().mean()

            log_probability_ratio = (
                new_log_probabilities - old_log_probabilities[indices]
            )

            probability_ratio = torch.exp(log_probability_ratio)

            unclipped_objective = probability_ratio * advantages[indices]

            clipped_ratio = torch.clamp(
                probability_ratio,
                1.0 - config.clip_ratio,
                1.0 + config.clip_ratio,
            )

            clipped_objective = clipped_ratio * advantages[indices]

            policy_loss = -torch.minimum(
                unclipped_objective,
                clipped_objective,
            ).mean()

            value_loss = torch.mean((predicted_values - returns[indices]) ** 2)

            total_loss = (
                policy_loss
                + (config.value_loss_weight * value_loss)
                - (config.entropy_weight * entropy)
            )

            optimizer.zero_grad(
                set_to_none=True,
            )

            total_loss.backward()

            gradient_norm = nn.utils.clip_grad_norm_(
                network.parameters(),
                config.maximum_gradient_norm,
            )

            optimizer.step()

            with torch.no_grad():
                # This is a nonnegative approximation of
                # KL(old_policy || new_policy):
                #
                #     ratio - 1 - log(ratio)
                #
                # Unlike mean(old_log_prob - new_log_prob),
                # this estimator is nonnegative for each sample.
                approximate_kl = (
                    probability_ratio - 1.0 - log_probability_ratio
                ).mean()

                clipped_fraction = (
                    (torch.abs(probability_ratio - 1.0) > config.clip_ratio)
                    .float()
                    .mean()
                )

            policy_loss_value = float(policy_loss.item())

            value_loss_value = float(value_loss.item())

            entropy_value = float(entropy.item())

            approximate_kl_value = float(approximate_kl.item())

            clipped_fraction_value = float(clipped_fraction.item())

            gradient_norm_value = float(gradient_norm.item())

            policy_losses.append(policy_loss_value)

            value_losses.append(value_loss_value)

            entropies.append(entropy_value)

            approximate_kls.append(approximate_kl_value)

            epoch_approximate_kls.append(approximate_kl_value)

            clipped_fractions.append(clipped_fraction_value)

            gradient_norms.append(gradient_norm_value)

        epochs_completed += 1

        if (
            config.target_kl is not None
            and epoch_approximate_kls
            and float(np.mean(epoch_approximate_kls)) > config.target_kl
        ):
            early_stopped = True
            break

    return RPPOMetrics(
        policy_loss=float(np.mean(policy_losses)),
        value_loss=float(np.mean(value_losses)),
        entropy=float(np.mean(entropies)),
        approximate_kl=float(np.mean(approximate_kls)),
        clipped_fraction=float(np.mean(clipped_fractions)),
        gradient_norm=float(np.mean(gradient_norms)),
        minibatch_updates=len(policy_losses),
        epochs_completed=epochs_completed,
        early_stopped=early_stopped,
    )
