"""Permutation-equivariant pairwise policy and value models."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import torch
from torch import nn

from ..RGraph import RGraph
from .REncoding import PAIR_INPUT_CHANNELS


@dataclass(frozen=True, slots=True)
class RModelConfig:
    """
    Immutable architecture settings for the pairwise model.
    """

    input_size: int = PAIR_INPUT_CHANNELS
    hidden_size: int = 64
    number_of_layers: int = 4
    dropout: float = 0.0

    def __post_init__(self) -> None:
        integer_fields = (
            "input_size",
            "hidden_size",
            "number_of_layers",
        )

        for name in integer_fields:
            value = getattr(
                self,
                name,
            )

            if isinstance(value, bool) or not isinstance(
                value,
                Integral,
            ):
                raise TypeError(f"{name} must be an integer.")

            value = int(value)

            minimum = 0 if name == "number_of_layers" else 1

            if value < minimum:
                raise ValueError(f"{name} must be at least " f"{minimum}.")

            object.__setattr__(
                self,
                name,
                value,
            )

        if isinstance(self.dropout, bool) or not isinstance(
            self.dropout,
            Real,
        ):
            raise TypeError("dropout must be numeric.")

        dropout = float(self.dropout)

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be at least zero " "and less than one.")

        object.__setattr__(
            self,
            "dropout",
            dropout,
        )


class RPairMessageLayer(nn.Module):
    """
    Update every pair representation through all third vertices.

    For pair (i, j), information is aggregated through paths:

        i → k → j

    for every vertex k.
    """

    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.relation_projection = nn.Linear(
            hidden_size,
            hidden_size,
        )

        self.message_projection = nn.Sequential(
            nn.Linear(
                hidden_size,
                hidden_size,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_size,
                hidden_size,
            ),
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(
                hidden_size,
                hidden_size * 2,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_size * 2,
                hidden_size,
            ),
        )

        self.message_norm = nn.LayerNorm(hidden_size)

        self.feed_forward_norm = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        pair_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return updated pair features.

        pair_features has shape:

            (
                batch,
                n_vertices,
                n_vertices,
                hidden_size,
            )
        """
        if pair_features.ndim != 4:
            raise ValueError("pair_features must have " "four dimensions.")

        n_vertices = pair_features.shape[1]

        projected = self.relation_projection(pair_features)

        # message[b, i, j, d]
        #
        #     = sum_k projected[b, i, k, d]
        #             * projected[b, k, j, d]
        message = torch.einsum(
            "bikd,bkjd->bijd",
            projected,
            projected,
        )

        message = message / (n_vertices**0.5)

        pair_features = self.message_norm(
            pair_features + self.dropout(self.message_projection(message))
        )

        pair_features = self.feed_forward_norm(
            pair_features + self.dropout(self.feed_forward(pair_features))
        )

        # Edges are undirected, so explicitly preserve symmetry.
        return 0.5 * (
            pair_features
            + pair_features.transpose(
                1,
                2,
            )
        )


class RPairPolicyValueNetwork(nn.Module):
    """
    Permutation-equivariant edge-policy and graph-value network.

    Outputs
    -------
    logits:
        One action logit for every edge.

    values:
        One global value estimate for every graph.
    """

    unavailable_logit = -1.0e9

    def __init__(
        self,
        graph: RGraph,
        config: RModelConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = config if config is not None else RModelConfig()

        self.n_vertices = graph.problem.n_vertices

        self.number_of_edges = graph.number_of_edges

        edge_vertices = torch.as_tensor(
            graph.edges.astype(np.int64),
            dtype=torch.long,
        )

        # A registered buffer moves automatically between devices
        # and is included in the state dictionary.
        self.register_buffer(
            "edge_vertices",
            edge_vertices,
        )

        self.input_projection = nn.Sequential(
            nn.Linear(
                self.config.input_size,
                self.config.hidden_size,
            ),
            nn.GELU(),
            nn.Linear(
                self.config.hidden_size,
                self.config.hidden_size,
            ),
            nn.LayerNorm(self.config.hidden_size),
        )

        self.layers = nn.ModuleList(
            [
                RPairMessageLayer(
                    hidden_size=self.config.hidden_size,
                    dropout=self.config.dropout,
                )
                for _ in range(self.config.number_of_layers)
            ]
        )

        self.policy_head = nn.Sequential(
            nn.Linear(
                self.config.hidden_size,
                self.config.hidden_size,
            ),
            nn.GELU(),
            nn.Linear(
                self.config.hidden_size,
                1,
            ),
        )

        self.value_head = nn.Sequential(
            nn.Linear(
                self.config.hidden_size * 2,
                self.config.hidden_size,
            ),
            nn.GELU(),
            nn.Linear(
                self.config.hidden_size,
                1,
            ),
        )

    @property
    def trainable_parameter_count(self) -> int:
        """
        Return the number of trainable scalar parameters.
        """
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def forward(
        self,
        pair_input: torch.Tensor,
        available_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Return action logits and one value estimate per graph.

        Parameters
        ----------
        pair_input:
            Shape:

                (
                    batch,
                    n_vertices,
                    n_vertices,
                    input_size,
                )

        available_mask:
            Shape:

                (
                    batch,
                    number_of_edges,
                )

            True means the corresponding action is available.
        """
        self._validate_inputs(
            pair_input,
            available_mask,
        )

        pair_features = self.input_projection(pair_input)

        for layer in self.layers:
            pair_features = layer(pair_features)

        i = self.edge_vertices[:, 0]
        j = self.edge_vertices[:, 1]

        edge_features = pair_features[
            :,
            i,
            j,
            :,
        ]

        logits = self.policy_head(edge_features).squeeze(-1)

        logits = logits.masked_fill(
            ~available_mask,
            self.unavailable_logit,
        )

        # Pool only real edges, not diagonal matrix positions.
        mean_features = edge_features.mean(dim=1)

        maximum_features = edge_features.amax(dim=1)

        graph_features = torch.cat(
            [
                mean_features,
                maximum_features,
            ],
            dim=-1,
        )

        values = self.value_head(graph_features).squeeze(-1)

        return (
            logits,
            values,
        )

    def _validate_inputs(
        self,
        pair_input: torch.Tensor,
        available_mask: torch.Tensor,
    ) -> None:
        """
        Validate batch, graph, feature, and mask dimensions.
        """
        expected_pair_tail = (
            self.n_vertices,
            self.n_vertices,
            self.config.input_size,
        )

        if pair_input.ndim != 4 or pair_input.shape[1:] != expected_pair_tail:
            raise ValueError(
                "pair_input must have shape "
                f"(batch, {self.n_vertices}, "
                f"{self.n_vertices}, "
                f"{self.config.input_size})."
            )

        expected_mask_shape = (
            pair_input.shape[0],
            self.number_of_edges,
        )

        if available_mask.shape != expected_mask_shape:
            raise ValueError(
                "available_mask must have shape " f"{expected_mask_shape}."
            )

        if available_mask.dtype != torch.bool:
            raise TypeError("available_mask must have " "boolean dtype.")
