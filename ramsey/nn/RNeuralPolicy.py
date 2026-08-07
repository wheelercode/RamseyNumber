"""Adapter connecting neural model output to the search policy interface.

Wraps a trained :class:`~ramsey.nn.RModel.RPairPolicyValueNetwork` in
an :class:`~ramsey.RPolicy.RPolicy` implementation so it can be used
by :class:`~ramsey.RSearch.RSearch` like any other search policy,
either sampling actions stochastically (for rollout collection during
PPO training) or selecting the highest-probability action greedily
(for evaluation).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributions import Categorical

from ..REnvironment import REnvironment
from ..RPolicy import RPolicy
from .REncoding import build_network_input
from .RModel import RPairPolicyValueNetwork
from .RRuntime import resolve_torch_device


@dataclass(frozen=True, slots=True)
class RNeuralDecision:
    """Action and diagnostic predictions from one policy evaluation.

    Attributes:
        edge (int): Encoded index of the edge action selected (either
            sampled or, in greedy mode, the highest-logit available
            edge).
        log_probability (float): Log-probability the policy's action
            distribution assigned to ``edge``, under the categorical
            distribution built from the network's logits.
        value (float): The network's scalar value estimate for the
            current search state, from the value head.
        entropy (float): Entropy of the full action distribution over
            available edges, a measure of how spread out (as opposed
            to peaked) the policy's action probabilities are.
    """

    edge: int
    log_probability: float
    value: float
    entropy: float


class RNeuralPolicy(RPolicy):
    """Select actions using a trained policy/value network.

    Builds the pair-input and action-mask tensors for the environment's
    current search state, runs the network in evaluation mode
    (gradient-free), and either samples from or takes the argmax of the
    resulting categorical action distribution over available edges.
    """

    def __init__(
        self,
        network: RPairPolicyValueNetwork,
        device: torch.device | str,
        *,
        greedy: bool = False,
    ) -> None:
        """Wrap a trained network as a search policy.

        Args:
            network (RPairPolicyValueNetwork): Policy/value network to
                query for action logits and value estimates. Must
                already reside on ``device``.
            device (torch.device | str): Device the network's
                parameters live on and the device the input tensors are
                built on; resolved via
                :func:`~ramsey.nn.RRuntime.resolve_torch_device`.
            greedy (bool): If ``True``, always select the
                highest-logit available action. If ``False`` (the
                default), sample from the action distribution.

        Raises:
            TypeError: If ``greedy`` is not a boolean.
        """
        if not isinstance(greedy, bool):
            raise TypeError("greedy must be boolean.")

        self._network = network
        self._device = resolve_torch_device(device)
        self._greedy = greedy

    @property
    def name(self) -> str:
        """str: ``"neural-greedy"`` or ``"neural-sampling"``, per ``greedy``."""
        if self._greedy:
            return "neural-greedy"

        return "neural-sampling"

    @property
    def network(
        self,
    ) -> RPairPolicyValueNetwork:
        """RPairPolicyValueNetwork: The wrapped policy/value network."""
        return self._network

    @property
    def device(self) -> torch.device:
        """torch.device: Device the network and its inputs reside on."""
        return self._device

    @property
    def greedy(self) -> bool:
        """bool: Whether actions are selected greedily rather than sampled."""
        return self._greedy

    @property
    def requires_full_analysis(self) -> bool:
        """bool: Always ``False``; the neural policy does not need cached exact action analysis."""
        return False

    def select_action(
        self,
        environment: REnvironment,
    ) -> int:
        """Return the selected edge index for the current search state.

        Args:
            environment (REnvironment): Environment providing the
                current search state and action availability.

        Returns:
            int: Encoded index of the edge action chosen by
            :meth:`evaluate`.
        """
        return self.evaluate(environment).edge

    @torch.no_grad()
    def evaluate(
        self,
        environment: REnvironment,
    ) -> RNeuralDecision:
        """Evaluate the current state and return one decision.

        Runs under ``torch.no_grad()``: no gradients are tracked, since
        this is inference-only evaluation (used both for rollout action
        selection and for policy diagnostics), not a training step.

        Args:
            environment (REnvironment): Environment providing the
                current search state and action availability.

        Returns:
            RNeuralDecision: The selected edge, its log-probability
            under the action distribution, the network's value
            estimate for the state, and the distribution's entropy.

        Raises:
            ValueError: If the environment's host graph dimensions do
                not match the wrapped network.
            RuntimeError: If the network's parameters do not reside on
                this policy's device.
        """
        self._require_compatible_environment(environment)

        self._require_network_device()

        available = environment.available_action_mask_fast()

        (
            pair_input,
            available_mask,
        ) = build_network_input(
            environment.state,
            available,
            device=self._device,
        )

        self._network.eval()

        logits, values = self._network(
            pair_input,
            available_mask,
        )

        distribution = Categorical(logits=logits)

        if self._greedy:
            action = torch.argmax(
                logits,
                dim=-1,
            )
        else:
            action = distribution.sample()

        log_probability = distribution.log_prob(action)

        entropy = distribution.entropy()

        return RNeuralDecision(
            edge=int(action.item()),
            log_probability=float(log_probability.item()),
            value=float(values.item()),
            entropy=float(entropy.item()),
        )

    def _require_compatible_environment(
        self,
        environment: REnvironment,
    ) -> None:
        """Require the environment's host graph to match the network.

        Args:
            environment (REnvironment): Environment whose host graph
                dimensions are checked against the network.

        Raises:
            ValueError: If the environment graph's vertex or edge
                count does not match the network.
        """
        graph = environment.graph

        if (
            graph.problem.n_vertices != self._network.n_vertices
            or graph.number_of_edges != self._network.number_of_edges
        ):
            raise ValueError(
                "Environment graph dimensions " "do not match the network."
            )

    def _require_network_device(
        self,
    ) -> None:
        """Require the network's parameters to reside on this policy's device.

        Falls back to inspecting the ``edge_vertices`` buffer's device
        if the network happens to expose no parameters.

        Raises:
            RuntimeError: If the network's parameters (or, absent any
                parameters, its ``edge_vertices`` buffer) are on a
                different device than this policy.
        """
        try:
            actual_device = next(self._network.parameters()).device
        except StopIteration:
            actual_device = self._network.edge_vertices.device

        if actual_device != self._device:
            raise RuntimeError(
                f"Network is on {actual_device}, "
                "but policy device is "
                f"{self._device}. Move the network "
                "with network.to(device)."
            )