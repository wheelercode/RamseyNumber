"""Validated, immutable edge-coloring values and conversions."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from numpy.typing import NDArray

from .RGraph import RGraph

ColorArray = NDArray[np.uint8]


@dataclass(
    frozen=True,
    slots=True,
    eq=False,
)
class RColoring:
    """
    One immutable assignment of colors to the edges of an RGraph.
    """

    graph: RGraph
    colors: ColorArray

    def __post_init__(self) -> None:
        supplied = np.asarray(self.colors)

        expected_shape = (self.graph.number_of_edges,)

        if supplied.shape != expected_shape:
            raise ValueError(
                f"Expected coloring shape "
                f"{expected_shape}, received "
                f"{supplied.shape}."
            )

        if not (
            np.issubdtype(
                supplied.dtype,
                np.integer,
            )
            or np.issubdtype(
                supplied.dtype,
                np.bool_,
            )
        ):
            raise TypeError("Coloring values must be integers.")

        if np.any(supplied < 0) or np.any(supplied >= self.graph.problem.n_colors):
            raise ValueError(
                "Coloring contains a color outside " "the problem's valid color range."
            )

        normalized = supplied.astype(
            np.uint8,
            copy=True,
        )

        normalized.flags.writeable = False

        object.__setattr__(
            self,
            "colors",
            normalized,
        )

    def mutable_copy(self) -> ColorArray:
        """
        Return a writable copy for use by mutable search state.
        """
        return self.colors.copy()

    def color_of_edge(
        self,
        edge: int,
    ) -> int:
        """
        Return the color assigned to one encoded edge.
        """
        if edge < 0 or edge >= len(self.colors):
            raise IndexError(f"Invalid edge index: {edge}")

        return int(self.colors[edge])

    def to_color_matrix(
        self,
    ) -> ColorArray:
        """
        Return the symmetric matrix of edge colors.

        Diagonal entries are zero and do not represent colored
        self-edges. They are unused.
        """
        n_vertices = self.graph.problem.n_vertices

        matrix = np.zeros(
            (
                n_vertices,
                n_vertices,
            ),
            dtype=np.uint8,
        )

        i = self.graph.edges[:, 0]
        j = self.graph.edges[:, 1]

        matrix[i, j] = self.colors
        matrix[j, i] = self.colors

        return matrix

    @classmethod
    def from_color_matrix(
        cls,
        graph: RGraph,
        matrix: NDArray[np.integer],
    ) -> "RColoring":
        """
        Construct a coloring from a symmetric edge-color matrix.
        """
        matrix = np.asarray(matrix)

        n_vertices = graph.problem.n_vertices

        if matrix.shape != (
            n_vertices,
            n_vertices,
        ):
            raise ValueError(
                f"Expected matrix shape "
                f"({n_vertices}, {n_vertices}), "
                f"received {matrix.shape}."
            )

        if not np.array_equal(
            matrix,
            matrix.T,
        ):
            raise ValueError("The matrix must be symmetric.")

        if np.any(matrix < 0) or np.any(matrix >= graph.problem.n_colors):
            raise ValueError(
                "The matrix contains a color "
                "outside the problem's valid "
                "color range."
            )

        colors = matrix[
            graph.edges[:, 0],
            graph.edges[:, 1],
        ]

        return cls(
            graph=graph,
            colors=colors,
        )

    def to_adjacency_matrix(
        self,
        edge_color: int = 1,
    ) -> NDArray[np.uint8]:
        """
        Project one selected color to ordinary graph edges.

        An entry is one exactly when its edge has edge_color.
        Every other edge color becomes a non-edge.
        """
        self._validate_color(edge_color)

        adjacency = np.zeros(
            (
                self.graph.problem.n_vertices,
                self.graph.problem.n_vertices,
            ),
            dtype=np.uint8,
        )

        selected_edges = (self.colors == edge_color).astype(np.uint8)

        i = self.graph.edges[:, 0]
        j = self.graph.edges[:, 1]

        adjacency[i, j] = selected_edges
        adjacency[j, i] = selected_edges

        return adjacency

    @classmethod
    def from_adjacency_matrix(
        cls,
        graph: RGraph,
        matrix: NDArray[np.integer],
        edge_color: int = 1,
    ) -> "RColoring":
        """
        Construct a binary coloring from an ordinary adjacency matrix.
        """
        if graph.problem.n_colors != 2:
            raise ValueError(
                "An adjacency matrix can reconstruct " "only a two-color coloring."
            )

        if edge_color not in (
            0,
            1,
        ):
            raise ValueError("edge_color must be zero or one.")

        matrix = np.asarray(matrix)

        n_vertices = graph.problem.n_vertices

        if matrix.shape != (
            n_vertices,
            n_vertices,
        ):
            raise ValueError(
                f"Expected matrix shape "
                f"({n_vertices}, {n_vertices}), "
                f"received {matrix.shape}."
            )

        if not np.array_equal(
            matrix,
            matrix.T,
        ):
            raise ValueError("The adjacency matrix must be symmetric.")

        if not np.all((matrix == 0) | (matrix == 1)):
            raise ValueError("Adjacency matrix values must " "be zero or one.")

        if np.any(np.diag(matrix) != 0):
            raise ValueError("Adjacency matrix diagonal entries " "must be zero.")

        present = matrix[
            graph.edges[:, 0],
            graph.edges[:, 1],
        ].astype(np.uint8)

        non_edge_color = 1 - edge_color

        colors = np.where(
            present == 1,
            edge_color,
            non_edge_color,
        ).astype(np.uint8)

        return cls(
            graph=graph,
            colors=colors,
        )

    def complement(self) -> "RColoring":
        """
        Swap the two colors in a binary Ramsey coloring.
        """
        if self.graph.problem.n_colors != 2:
            raise ValueError("Color complement is defined here " "only for two colors.")

        return RColoring(
            graph=self.graph,
            colors=np.uint8(1) - self.colors,
        )

    def vertex_degrees(
        self,
        edge_color: int = 1,
    ) -> NDArray[np.int64]:
        """
        Return ordinary-graph degrees for one selected edge color.
        """
        adjacency = self.to_adjacency_matrix(edge_color=edge_color)

        return adjacency.sum(
            axis=1,
            dtype=np.int64,
        )

    def isolated_vertices(
        self,
        edge_color: int = 1,
    ) -> NDArray[np.int32]:
        """
        Return vertices having degree zero in one color projection.
        """
        degrees = self.vertex_degrees(edge_color=edge_color)

        return np.flatnonzero(degrees == 0).astype(np.int32)

    def packed(self) -> bytes:
        """
        Return a compact deterministic byte representation.
        """
        if self.graph.problem.n_colors == 2:
            return np.packbits(
                self.colors,
                bitorder="little",
            ).tobytes()

        return self.colors.tobytes()

    def exact_hash(self) -> str:
        """
        Return an identifier containing problem and coloring data.
        """
        problem = self.graph.problem

        header = (
            f"{problem.n_vertices}:"
            f"{problem.forbidden_clique_sizes}:"
            f"{self.graph.number_of_edges}:"
        ).encode("ascii")

        return sha256(header + self.packed()).hexdigest()

    def exact_equals(
        self,
        other: object,
    ) -> bool:
        """
        Return whether another coloring has identical semantics.
        """
        if not isinstance(
            other,
            RColoring,
        ):
            return False

        return self.graph.problem == other.graph.problem and np.array_equal(
            self.colors,
            other.colors,
        )

    def _validate_color(
        self,
        color: int,
    ) -> None:
        """
        Validate a color index used for a graph projection.
        """
        if color < 0 or color >= self.graph.problem.n_colors:
            raise IndexError(f"Invalid color index: {color}")
