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
    """
    Immutable hyperparameters for PPO parameter updates.

    Attributes:
        update_epochs (int): Number of full passes over one rollout's
            samples performed by :func:`ppo_update`, each pass using
            a fresh random minibatch permutation.
        minibatch_size (int): Number of rollout steps per gradient
            step within an epoch.
        clip_ratio (float): PPO clipping parameter ``epsilon``
            bounding the probability ratio to
            ``[1 - clip_ratio, 1 + clip_ratio]`` in the clipped
            surrogate objective.
        value_loss_weight (float): Coefficient applied to the mean
            squared-error value loss when it is added to the policy
            loss to form the total loss. Defaults to ``0.0``, which
            disables the value loss term entirely.
        entropy_weight (float): Coefficient applied to the policy
            entropy bonus, subtracted from the total loss to
            encourage exploration. Defaults to ``0.0``, which
            disables the entropy bonus.
        maximum_gradient_norm (float): Maximum global gradient norm
            used to clip gradients before each optimizer step.
        learning_rate (float): Learning rate passed to the Adam
            optimizer created by :func:`create_optimizer`.
        target_kl (float | None): Optional approximate-KL threshold.
            When the mean approximate KL over an epoch's minibatches
            exceeds this value, :func:`ppo_update` stops taking
            further epochs early. ``None`` disables early stopping.
    """

    update_epochs: int = 4
    minibatch_size: int = 16
    clip_ratio: float = 0.20
    value_loss_weight: float = 0.0
    entropy_weight: float = 0.0
    maximum_gradient_norm: float = 0.50
    learning_rate: float = 5.0e-4
    target_kl: float | None = None

    def __post_init__(self) -> None:
        """
        Validate, coerce, and normalize every configuration field in place.

        ``update_epochs`` and ``minibatch_size`` must be positive
        integers; ``clip_ratio``, ``value_loss_weight``,
        ``entropy_weight``, ``maximum_gradient_norm``, and
        ``learning_rate`` must be nonnegative numeric values, with
        ``clip_ratio``, ``maximum_gradient_norm``, and
        ``learning_rate`` further required to be strictly positive.
        ``target_kl`` must be ``None`` or a positive numeric value.
        Fields are rewritten with their coerced ``int``/``float``
        values via ``object.__setattr__`` because the dataclass is
        frozen.

        Raises:
            TypeError: If a field has the wrong type (including
                ``bool`` where an integer or float is expected).
            ValueError: If a field fails its range check.
        """
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
    """
    Aggregated diagnostics from one PPO update.

    Each scalar field is the mean of the corresponding per-minibatch
    quantity across every minibatch processed by :func:`ppo_update`,
    over however many epochs actually ran.

    Attributes:
        policy_loss (float): Mean clipped surrogate policy loss
            (negated, so lower is better) across processed
            minibatches.
        value_loss (float): Mean squared-error value loss across
            processed minibatches.
        entropy (float): Mean policy entropy across processed
            minibatches.
        approximate_kl (float): Mean nonnegative approximate
            KL-divergence estimate ``ratio - 1 - log(ratio)`` between
            the old and updated policy, averaged across processed
            minibatches.
        clipped_fraction (float): Mean fraction of samples per
            minibatch whose probability ratio fell outside
            ``[1 - clip_ratio, 1 + clip_ratio]``.
        gradient_norm (float): Mean global gradient norm observed
            before clipping, across processed minibatches.
        minibatch_updates (int): Total number of minibatch gradient
            steps performed across all completed epochs.
        epochs_completed (int): Number of epochs actually run, which
            may be less than ``config.update_epochs`` when early
            stopping triggers.
        early_stopped (bool): Whether the update loop stopped before
            ``config.update_epochs`` epochs because the mean
            approximate KL for an epoch exceeded ``config.target_kl``.
    """

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
        """
        Return metrics in a logging-friendly dictionary.

        Returns:
            dict[str, float | int | bool]: One entry per field of
            this dataclass, keyed by field name.
        """

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
    """
    Create the standard Adam optimizer used by PPO.

    Args:
        network (RPairPolicyValueNetwork): Policy/value network whose
            parameters will be optimized.
        config (RPPOConfig): PPO configuration supplying
            ``learning_rate``.

    Returns:
        torch.optim.Adam: Optimizer constructed over
            ``network.parameters()`` with learning rate
            ``config.learning_rate``.
    """

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
    this update are copied to the requested training device. For
    ``config.update_epochs`` epochs, the rollout's steps are shuffled
    into a new random permutation and split into minibatches of
    ``config.minibatch_size`` steps. For each minibatch, the network
    is re-evaluated on the stored ``pair_inputs``/``available_masks``
    to obtain new action log-probabilities, a new value prediction,
    and the action distribution's entropy, and the following clipped
    PPO loss is minimized::

        ratio = exp(new_log_prob - old_log_prob)
        unclipped = ratio * advantage
        clipped = clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantage
        policy_loss = -mean(min(unclipped, clipped))
        value_loss = mean((predicted_value - return) ** 2)
        total_loss = policy_loss
            + value_loss_weight * value_loss
            - entropy_weight * entropy

    ``total_loss`` is backpropagated, gradients are clipped to
    ``config.maximum_gradient_norm`` by global norm, and one Adam
    step is taken per minibatch. After each epoch, if
    ``config.target_kl`` is set and the epoch's mean approximate KL
    exceeds it, training stops early before starting another epoch.
    The network is set to training mode (``network.train()``) for the
    duration of the update.

    Args:
        network (RPairPolicyValueNetwork): Policy/value network being
            trained. Must already reside on ``device``.
        optimizer (torch.optim.Optimizer): Optimizer stepping
            ``network``'s parameters, typically created by
            :func:`create_optimizer`.
        rollout (RRolloutBatch): CPU-resident rollout supplying
            per-step inputs, masks, actions, old log-probabilities,
            advantages, and returns.
        device (torch.device | str): Device the rollout tensors are
            copied to for the update; must match the device
            ``network`` already resides on.
        config (RPPOConfig): PPO hyperparameters controlling epochs,
            minibatch size, clipping, loss weights, gradient clipping,
            and early stopping.

    Returns:
        RPPOMetrics: Diagnostics averaged over every minibatch update
        actually performed.

    Raises:
        ValueError: If ``rollout`` contains zero steps.
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
