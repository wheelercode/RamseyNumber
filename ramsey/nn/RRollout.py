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
from .RRuntime import resolve_torch_device


class RRolloutReward(str, Enum):
    """
    State-transition quantity used as the training reward.

    Selects which per-step reward reported by
    :meth:`~ramsey.REnvironment.REnvironment.step` is used, after
    scaling by ``RRolloutConfig.reward_scale``, as the PPO training
    reward for that step.

    Attributes:
        EXACT_SCORE: Use the environment's ``immediate_reward``, the
            change in exact score (count of forbidden monochromatic
            K5 cliques) caused by the action.
        OBJECTIVE: Use the environment's ``objective_reward``, the
            change in the (possibly smoother) search objective caused
            by the action.
    """

    EXACT_SCORE = "exact-score"
    OBJECTIVE = "objective"


@dataclass(frozen=True, slots=True)
class RRolloutConfig:
    """
    Immutable settings for one policy rollout.

    Attributes:
        rollout_steps (int): Maximum number of environment steps to
            collect before stopping, unless the episode terminates or
            truncates first.
        discount (float): Discount factor ``gamma`` used by
            :func:`calculate_advantages` for both the temporal
            difference error and the advantage's exponential decay.
        gae_lambda (float): Generalized Advantage Estimation decay
            parameter ``lambda`` controlling the bias/variance
            trade-off in :func:`calculate_advantages`.
        reward_scale (float): Positive divisor applied to every raw
            per-step reward before it is stored, keeping reward
            magnitudes in a range suitable for PPO training.
        reward_source (RRolloutReward): Which environment reward
            quantity is used as the (unscaled) per-step reward.
        normalize_advantages (bool): When ``True``, advantages
            computed for the rollout are standardized to zero mean
            and unit variance before being stored.
    """

    rollout_steps: int = 256

    discount: float = 0.995
    gae_lambda: float = 0.95

    reward_scale: float = 10.0

    reward_source: RRolloutReward = RRolloutReward.EXACT_SCORE

    normalize_advantages: bool = True

    def __post_init__(self) -> None:
        """
        Validate, coerce, and normalize every configuration field in place.

        ``rollout_steps`` must be a positive integer; ``discount`` and
        ``gae_lambda`` must be numeric values in ``[0, 1]``;
        ``reward_scale`` must be a positive numeric value;
        ``reward_source`` must be convertible to :class:`RRolloutReward`;
        ``normalize_advantages`` must be a ``bool``. Fields are
        rewritten with their coerced values via ``object.__setattr__``
        because the dataclass is frozen.

        Raises:
            TypeError: If a field has the wrong type (including
                ``bool`` where an integer or float is expected).
            ValueError: If a field fails its range check, or if
                ``reward_source`` is not a valid :class:`RRolloutReward`
                value.
        """
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

    Produced by :func:`collect_rollout` and consumed by
    :func:`ramsey.nn.RPPO.ppo_update`, which copies the required
    tensors to the training device per minibatch. Every tensor field
    below has a leading dimension equal to
    :attr:`number_of_steps`, one entry per collected environment
    step, in the order the steps were taken.

    Attributes:
        pair_inputs (torch.Tensor): Float32 tensor of shape
            ``(number_of_steps, n_vertices, n_vertices, input_size)``
            holding the per-step, per-vertex-pair network input
            features used by the policy/value network.
        available_masks (torch.Tensor): Boolean tensor of shape
            ``(number_of_steps, number_of_edges)`` marking which
            edges/actions were available (unblocked by tabu memory or
            already-colored edges) at each step.
        actions (torch.Tensor): Long tensor of shape
            ``(number_of_steps,)`` holding the sampled edge/action
            index at each step.
        old_log_probabilities (torch.Tensor): Float32 tensor of shape
            ``(number_of_steps,)`` holding the log-probability the
            behavior policy assigned to the sampled action, used as
            the PPO reference policy for the probability ratio.
        rewards (torch.Tensor): Float32 tensor of shape
            ``(number_of_steps,)`` holding the scaled per-step reward
            (see ``RRolloutConfig.reward_scale`` and
            ``RRolloutConfig.reward_source``).
        old_values (torch.Tensor): Float32 tensor of shape
            ``(number_of_steps,)`` holding the value network's
            state-value prediction at each step, before the update.
        advantages (torch.Tensor): Float32 tensor of shape
            ``(number_of_steps,)`` holding the generalized advantage
            estimate for each step, as computed by
            :func:`calculate_advantages` and optionally normalized.
        returns (torch.Tensor): Float32 tensor of shape
            ``(number_of_steps,)`` holding the value-function
            regression target (``advantages + old_values``) for each
            step.
        initial_score (int): Exact score of the coloring the rollout
            started from.
        final_score (int): Exact score of the coloring at the end of
            the rollout (after the last collected step).
        best_score (int): Lowest exact score observed by the
            environment during the rollout.
        final_coloring (RColoring): Coloring snapshot at the end of
            the rollout.
        best_coloring (RColoring): Best (lowest-score) coloring
            snapshot observed during the rollout.
        terminated (bool): Whether the environment reached a true
            terminal search state (for example, a score-zero
            coloring) during the rollout.
        truncated (bool): Whether the environment stopped the episode
            due to a step or other time limit rather than a true
            terminal state.
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
        """
        Verify that every per-step tensor field has equal length.

        Raises:
            ValueError: If ``pair_inputs``, ``available_masks``,
                ``old_log_probabilities``, ``rewards``, ``old_values``,
                ``advantages``, or ``returns`` does not have the same
                length as ``actions``.
        """
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
        """int: Number of environment steps collected in this rollout."""
        return len(self.actions)

    @property
    def total_scaled_reward(self) -> float:
        """float: Sum of the scaled per-step rewards across the rollout."""
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

    Implements Generalized Advantage Estimation (GAE), computing the
    advantage at each step by a backward recursion over the
    one-step temporal difference errors::

        delta[t] = reward[t] + discount * next_value[t] * can_continue[t]
            - value[t]
        advantage[t] = delta[t]
            + discount * gae_lambda * can_continue[t] * advantage[t + 1]

    where ``next_value[t]`` is ``values[t + 1]`` for all but the last
    step, and ``last_value`` for the last step, and
    ``can_continue[t]`` is ``0`` when ``terminated[t]`` is ``True``
    and ``1`` otherwise. A true terminal transition therefore prevents
    value bootstrapping past that step (``can_continue`` zeroes out
    both the next-value term and the recursive advantage carry-over),
    while a time-limit truncation -- represented by the corresponding
    ``terminated`` entry being ``False`` -- does not, so bootstrapping
    continues using ``last_value``. Value-regression targets
    (``returns``) are then ``advantages + values``.

    Args:
        rewards (numpy.ndarray): One-dimensional array of per-step
            rewards, coerced to ``float32``.
        values (numpy.ndarray): One-dimensional array of per-step
            value predictions, coerced to ``float32``, the same
            length as ``rewards``.
        terminated (numpy.ndarray): One-dimensional boolean array,
            the same length as ``rewards``, marking which steps ended
            in a true terminal state (as opposed to a truncation).
        last_value (float): Bootstrap value used for the step after
            the final collected step, i.e. the value network's
            prediction for the state the rollout stopped at (``0.0``
            when that state is a true terminal state).
        discount (float): Discount factor ``gamma`` in ``[0, 1]``.
        gae_lambda (float): GAE decay parameter ``lambda`` in
            ``[0, 1]``.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: A ``(advantages, returns)``
        pair, each a ``float32`` array of the same length as
        ``rewards``.

    Raises:
        ValueError: If ``rewards``, ``values``, or ``terminated`` is
            not one-dimensional, if their lengths differ, or if
            ``discount`` or ``gae_lambda`` is outside ``[0, 1]``.
        TypeError: If ``terminated`` does not have boolean dtype.
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

    Resets ``environment`` to ``coloring`` and then repeatedly: builds
    the network input and available-action mask for the current
    search state, samples an action from the policy's categorical
    action distribution (network in evaluation mode, no gradient
    tracking), records the value prediction and action
    log-probability, and applies the action to the environment. This
    continues for up to ``config.rollout_steps`` steps, stopping early
    if the environment terminates or truncates. The per-step reward
    is taken from either the environment's exact-score reward or its
    objective reward (selected by ``config.reward_source``), divided
    by ``config.reward_scale``. After the loop, a bootstrap value is
    obtained: ``0.0`` if the environment reached a true terminal
    state, otherwise the value network's prediction for the final
    (non-terminal) state reached. Generalized advantages and returns
    are then computed with :func:`calculate_advantages`, and
    advantages are optionally standardized to zero mean and unit
    variance when ``config.normalize_advantages`` is set. All
    resulting tensors are assembled, CPU-resident, into an
    :class:`RRolloutBatch` alongside score and coloring summaries
    from ``environment``.

    Args:
        network (RPairPolicyValueNetwork): Policy/value network used
            to act in the environment and to estimate values. Must
            already reside on ``device`` and must have vertex/edge
            dimensions matching ``environment.graph``.
        environment (REnvironment): Search environment being driven;
            reset to ``coloring`` at the start of the call and mutated
            by repeated calls to ``environment.step``.
        coloring (RColoring): Seed coloring the rollout starts from.
        device (torch.device | str): Device the network is expected
            to reside on and that network inputs are built for.
            Resolved via :func:`~ramsey.nn.RRuntime.resolve_torch_device`.
        config (RRolloutConfig): Rollout settings controlling step
            budget, discounting, reward scaling/source, and advantage
            normalization.

    Returns:
        RRolloutBatch: CPU-resident experiences and summary data from
        this rollout.

    Raises:
        RuntimeError: If ``network``'s actual device does not match
            the resolved ``device``.
        ValueError: If ``environment.graph``'s vertex/edge dimensions
            do not match ``network``'s.
    """
    device = resolve_torch_device(device)

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
    """
    Verify that ``environment``'s host graph matches ``network``'s dimensions.

    Args:
        network (RPairPolicyValueNetwork): Policy/value network whose
            expected vertex and edge counts are checked against.
        environment (REnvironment): Environment supplying the host
            graph to validate.

    Raises:
        ValueError: If the environment's number of vertices or number
            of edges does not match the network's.
    """
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
    """
    Verify that ``network``'s parameters already reside on ``device``.

    Falls back to inspecting the ``edge_vertices`` buffer when
    ``network`` has no parameters (``next(network.parameters())``
    raises ``StopIteration``).

    Args:
        network (RPairPolicyValueNetwork): Network whose device
            placement is checked.
        device (torch.device): Expected device.

    Raises:
        RuntimeError: If ``network``'s actual device differs from
            ``device``.
    """
    try:
        actual_device = next(network.parameters()).device
    except StopIteration:
        actual_device = network.edge_vertices.device

    if actual_device != device:
        raise RuntimeError(
            f"Network is on {actual_device}, " "but rollout device is " f"{device}."
        )