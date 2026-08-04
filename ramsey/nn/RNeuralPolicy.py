"""Adapter connecting neural model output to the search policy interface."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributions import Categorical

from ..REnvironment import REnvironment
from ..RPolicy import RPolicy
from .REncoding import build_network_input
from .RModel import RPairPolicyValueNetwork


@dataclass(frozen=True, slots=True)
class RNeuralDecision:
    """
    Action and diagnostic predictions from one policy evaluation.
    """

    edge: int
    log_probability: float
    value: float
    entropy: float


class RNeuralPolicy(RPolicy):
    """
    Select actions using a trained policy/value network.
    """

    def __init__(
        self,
        network: RPairPolicyValueNetwork,
        device: torch.device | str,
        *,
        greedy: bool = False,
    ) -> None:
        if not isinstance(greedy, bool):
            raise TypeError("greedy must be boolean.")

        self._network = network
        self._device = self._resolve_device(device)
        self._greedy = greedy

    @property
    def name(self) -> str:
        if self._greedy:
            return "neural-greedy"

        return "neural-sampling"

    @property
    def network(
        self,
    ) -> RPairPolicyValueNetwork:
        return self._network

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def greedy(self) -> bool:
        return self._greedy

    @property
    def requires_full_analysis(self) -> bool:
        return False

    def select_action(
        self,
        environment: REnvironment,
    ) -> int:
        return self.evaluate(environment).edge

    @torch.no_grad()
    def evaluate(
        self,
        environment: REnvironment,
    ) -> RNeuralDecision:
        """
        Evaluate the current state and return one decision.
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

    def _resolve_device(
        device: torch.device | str,
    ) -> torch.device:
        """Resolve an unindexed CUDA device to the current device."""
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
