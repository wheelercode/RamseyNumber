"""Permutation-equivariant pairwise policy and value models.

Defines the learned policy/value network used by
:class:`~ramsey.nn.RNeuralPolicy.RNeuralPolicy` and trained by PPO.
The network operates on the vertex-pair feature tensors produced by
:mod:`ramsey.nn.REncoding` and is equivariant to vertex relabeling: it
has no notion of vertex identity beyond the pair features it is given,
so the same weights apply regardless of how the host graph's vertices
happen to be numbered. Its outputs — an edge-flip action distribution
and a scalar value estimate — are learned, heuristic approximations
guiding search; they are not exact computations of score reduction or
outcome, unlike the exact action-analysis machinery elsewhere in the
package.
"""

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
    """Immutable architecture settings for the pairwise model.

    Validated and normalized in :meth:`__post_init__`; values are
    coerced to their canonical Python type (``int``/``float``) and
    stored back onto the frozen instance via ``object.__setattr__``.

    Attributes:
        input_size (int): Number of input feature channels per vertex
            pair, i.e. the size of the last axis of the pair-input
            tensor. Defaults to ``PAIR_INPUT_CHANNELS`` from
            :mod:`ramsey.nn.REncoding`. Must be at least 1.
        hidden_size (int): Width of the hidden pair representation
            used throughout the message-passing layers and heads. Must
            be at least 1.
        number_of_layers (int): Number of stacked
            :class:`RPairMessageLayer` blocks. May be 0, in which case
            the network applies only the input projection before the
            policy and value heads.
        dropout (float): Dropout probability applied inside each
            :class:`RPairMessageLayer`. Must satisfy
            ``0.0 <= dropout < 1.0``.

    Raises:
        TypeError: If ``input_size``, ``hidden_size``, or
            ``number_of_layers`` is not an integer, or ``dropout`` is
            not numeric.
        ValueError: If ``input_size`` or ``hidden_size`` is less than
            1, ``number_of_layers`` is negative, or ``dropout`` is
            outside ``[0.0, 1.0)``.
    """

    input_size: int = PAIR_INPUT_CHANNELS
    hidden_size: int = 64
    number_of_layers: int = 4
    dropout: float = 0.0

    def __post_init__(self) -> None:
        """Validate and normalize field types and ranges in place.

        See the class docstring for the exact constraints enforced on
        each field.
        """
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
    """Update every pair representation through all third vertices.

    For pair ``(i, j)``, information is aggregated through paths::

        i -> k -> j

    for every vertex ``k``, via a bilinear (einsum) interaction
    between projected pair features, followed by a message
    feed-forward block and a position-wise feed-forward block, each
    wrapped in a residual connection and layer normalization. Because
    the aggregation is defined identically for every ``(i, j, k)``
    triple with no reference to fixed vertex identity, the layer's
    output is equivariant to a permutation of vertex labels. The
    output is explicitly symmetrized so that pair ``(i, j)`` and pair
    ``(j, i)`` always carry identical features, matching the
    undirected nature of host-graph edges.

    Attributes:
        relation_projection (torch.nn.Linear): Linear projection of
            pair features used to form both sides of the bilinear
            third-vertex interaction.
        message_projection (torch.nn.Sequential): Two-layer GELU
            feed-forward block applied to the aggregated message before
            it is added back into the pair features.
        feed_forward (torch.nn.Sequential): Position-wise two-layer
            GELU feed-forward block (expanding to twice ``hidden_size``
            and back) applied after the message update.
        message_norm (torch.nn.LayerNorm): Layer normalization applied
            after the residual message update.
        feed_forward_norm (torch.nn.LayerNorm): Layer normalization
            applied after the residual feed-forward update.
        dropout (torch.nn.Dropout): Dropout applied to the outputs of
            ``message_projection`` and ``feed_forward`` before they are
            added back as residuals.
    """

    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.0,
    ) -> None:
        """Construct the projections, feed-forward blocks, and norms.

        Args:
            hidden_size (int): Width of the pair feature representation
                consumed and produced by this layer.
            dropout (float): Dropout probability applied to the
                message and feed-forward residual branches.
        """
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
        """Return updated, symmetrized pair features.

        Projects ``pair_features`` and aggregates messages over every
        third vertex ``k`` via the bilinear interaction
        ``message[b, i, j, d] = sum_k projected[b, i, k, d] * projected[b, k, j, d]``,
        scaled by ``1 / sqrt(n_vertices)``. The message is passed
        through ``message_projection``, added back as a residual, and
        normalized; the result is then passed through ``feed_forward``,
        added back as a residual, and normalized again. The final
        output is symmetrized as
        ``0.5 * (pair_features + pair_features.transpose(1, 2))`` to
        preserve the fact that host-graph edges are undirected.

        Args:
            pair_features (torch.Tensor): Pair feature tensor of shape
                ``(batch, n_vertices, n_vertices, hidden_size)``.

        Returns:
            torch.Tensor: Updated pair features of the same shape as
            ``pair_features``, symmetric under transposing the two
            vertex axes.

        Raises:
            ValueError: If ``pair_features`` does not have four
                dimensions.
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
    """Permutation-equivariant edge-policy and graph-value network.

    Projects the per-vertex-pair input features into a hidden
    representation, refines that representation through a stack of
    :class:`RPairMessageLayer` blocks, then reads out per-edge action
    logits from a policy head and a single scalar value estimate per
    graph from a value head. The network has no learned parameters
    tied to specific vertex or edge identities, so its predictions are
    equivariant to vertex relabeling; this makes the same trained
    weights applicable to any host graph with the number of vertices
    and edges the network was constructed for. Both outputs are
    learned approximations used to guide search (a PPO policy
    distribution and a bootstrapped value estimate) rather than exact
    measurements of the effect of an action.

    Attributes:
        config (RModelConfig): Architecture settings this network was
            constructed with.
        n_vertices (int): Number of vertices in the host graph this
            network was built for.
        number_of_edges (int): Number of edges in the host graph this
            network was built for; also the size of the action space.
        edge_vertices (torch.Tensor): Registered buffer of shape
            ``(number_of_edges, 2)`` and dtype ``torch.long`` giving the
            two endpoint vertices of each edge, in the host graph's
            edge order. Moves automatically with the module between
            devices and is included in ``state_dict()``.
        input_projection (torch.nn.Sequential): Projects raw pair
            input features (of size ``config.input_size``) to the
            hidden representation (of size ``config.hidden_size``).
        layers (torch.nn.ModuleList): Stack of ``config.number_of_layers``
            :class:`RPairMessageLayer` blocks refining the pair
            representation.
        policy_head (torch.nn.Sequential): Maps each edge's hidden pair
            representation to a single scalar action logit.
        value_head (torch.nn.Sequential): Maps pooled graph-level
            features to a single scalar value estimate.
    """

    unavailable_logit = -1.0e9

    def __init__(
        self,
        graph: RGraph,
        config: RModelConfig | None = None,
    ) -> None:
        """Construct the network's layers for one fixed host graph.

        Args:
            graph (RGraph): Host graph whose vertex count, edge count,
                and edge ordering (``graph.edges``) fix the network's
                input/output dimensions and the ``edge_vertices``
                buffer used to gather per-edge features.
            config (RModelConfig | None): Architecture settings. Uses
                ``RModelConfig()`` defaults when ``None``.
        """
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
        """int: Total number of scalar parameters with ``requires_grad``."""
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
        """Return action logits and one value estimate per graph.

        Projects and refines the pair features through
        ``input_projection`` and each layer in ``layers``, gathers the
        refined feature vector at each edge's ``(i, j)`` position using
        ``edge_vertices``, and applies ``policy_head`` to obtain one
        logit per edge. Logits at unavailable actions are overwritten
        with ``unavailable_logit`` (effectively ``-inf`` for sampling
        or softmax purposes) so that an action distribution built from
        these logits never selects them. Graph-level features are
        formed by mean- and max-pooling the edge features over the
        edge axis only (diagonal, non-edge matrix positions are
        excluded from pooling because gathering is restricted to
        ``edge_vertices``), concatenated, and passed through
        ``value_head`` to obtain one scalar value per graph.

        Args:
            pair_input (torch.Tensor): Pair feature tensor of shape
                ``(batch, n_vertices, n_vertices, config.input_size)``,
                as produced by
                :func:`~ramsey.nn.REncoding.build_network_input`.
            available_mask (torch.Tensor): Boolean tensor of shape
                ``(batch, number_of_edges)`` where ``True`` marks the
                corresponding edge as a currently available action.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A pair
            ``(logits, values)``. ``logits`` has shape
            ``(batch, number_of_edges)``, with unavailable actions
            forced to ``unavailable_logit``. ``values`` has shape
            ``(batch,)``, one scalar value estimate per graph in the
            batch.

        Raises:
            ValueError: If ``pair_input`` does not have shape
                ``(batch, n_vertices, n_vertices, config.input_size)``
                or ``available_mask`` does not have shape
                ``(batch, number_of_edges)``.
            TypeError: If ``available_mask`` does not have boolean
                dtype.
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
        """Validate batch, graph, feature, and mask dimensions.

        Args:
            pair_input (torch.Tensor): Candidate pair-input tensor to
                validate against ``(batch, n_vertices, n_vertices,
                config.input_size)``.
            available_mask (torch.Tensor): Candidate action-mask
                tensor to validate against
                ``(pair_input.shape[0], number_of_edges)`` with boolean
                dtype.

        Raises:
            ValueError: If ``pair_input`` does not have four
                dimensions with the expected trailing shape, or if
                ``available_mask`` does not have the expected shape.
            TypeError: If ``available_mask`` does not have boolean
                dtype.
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
