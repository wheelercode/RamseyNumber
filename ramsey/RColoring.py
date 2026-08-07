"""Validated, immutable edge-coloring values and conversions.

Defines :class:`RColoring`, the immutable value type for one complete
assignment of colors to every edge of an :class:`ramsey.RGraph.RGraph`.
It validates colors against the graph's problem definition and provides
conversions to and from color/adjacency matrix representations, along
with content-addressable hashing used by archives and duplicate
detection.
"""

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
    """One immutable assignment of colors to the edges of an RGraph.

    Attributes:
        graph (RGraph): Host graph whose edges are colored.
        colors (ColorArray): Read-only ``uint8`` array of shape
            ``(graph.number_of_edges,)``. Entry ``e`` is the color of
            host edge ``e``, aligned with ``graph.edges``.
    """

    graph: RGraph
    colors: ColorArray

    def __post_init__(self) -> None:
        """Validate the supplied colors and normalize/freeze them.

        Raises:
            ValueError: If ``colors`` does not have shape
                ``(graph.number_of_edges,)``, or contains a value
                outside the problem's valid color range.
            TypeError: If ``colors`` does not have an integer or
                boolean dtype.
        """
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
        """Return a writable copy for use by mutable search state.

        Returns:
            ColorArray: Independently owned, writable copy of
            ``colors``.
        """
        return self.colors.copy()

    def color_of_edge(
        self,
        edge: int,
    ) -> int:
        """Return the color assigned to one encoded edge.

        Args:
            edge (int): Index of the host edge to look up.

        Returns:
            int: Color currently assigned to ``edge``.

        Raises:
            IndexError: If ``edge`` is out of range.
        """
        if edge < 0 or edge >= len(self.colors):
            raise IndexError(f"Invalid edge index: {edge}")

        return int(self.colors[edge])

    def to_color_matrix(
        self,
    ) -> ColorArray:
        """Return the symmetric matrix of edge colors.

        Diagonal entries are zero and do not represent colored
        self-edges. They are unused.

        Returns:
            ColorArray: Owned ``uint8`` array of shape
            ``(n_vertices, n_vertices)``, symmetric, with entry
            ``[i, j]`` equal to the color of edge ``(i, j)``.
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
        """Construct a coloring from a symmetric edge-color matrix.

        Args:
            graph (RGraph): Host graph the coloring is defined over.
            matrix (NDArray[np.integer]): Symmetric ``(n_vertices,
                n_vertices)`` matrix whose entry ``[i, j]`` gives the
                color of edge ``(i, j)``. Diagonal entries are ignored.

        Returns:
            RColoring: New coloring extracted from ``matrix`` along
            ``graph.edges``.

        Raises:
            ValueError: If ``matrix`` does not have shape
                ``(n_vertices, n_vertices)``, is not symmetric, or
                contains a color outside the problem's valid range.
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
        """Project one selected color to ordinary graph edges.

        An entry is one exactly when its edge has ``edge_color``.
        Every other edge color becomes a non-edge.

        Args:
            edge_color (int): Color to project into the adjacency
                matrix.

        Returns:
            NDArray[np.uint8]: Owned, symmetric ``(n_vertices,
            n_vertices)`` binary adjacency matrix for ``edge_color``.

        Raises:
            IndexError: If ``edge_color`` is out of range.
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
        """Construct a binary coloring from an ordinary adjacency matrix.

        Edges present in ``matrix`` (value ``1``) receive ``edge_color``;
        absent edges (value ``0``) receive the other color.

        Args:
            graph (RGraph): Host graph the coloring is defined over;
                must be a two-color problem.
            matrix (NDArray[np.integer]): Symmetric, binary, zero-
                diagonal ``(n_vertices, n_vertices)`` adjacency matrix.
            edge_color (int): Color assigned to edges present in
                ``matrix``; must be ``0`` or ``1``.

        Returns:
            RColoring: New two-color coloring built from ``matrix``.

        Raises:
            ValueError: If ``graph.problem.n_colors`` is not ``2``, if
                ``edge_color`` is not ``0`` or ``1``, if ``matrix`` does
                not have shape ``(n_vertices, n_vertices)``, is not
                symmetric, contains values other than ``0``/``1``, or
                has a nonzero diagonal.
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
        """Swap the two colors in a binary Ramsey coloring.

        Returns:
            RColoring: New coloring with every edge's color flipped
            (``1 - color``).

        Raises:
            ValueError: If ``graph.problem.n_colors`` is not ``2``.
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
        """Return ordinary-graph degrees for one selected edge color.

        Args:
            edge_color (int): Color whose adjacency projection is used.

        Returns:
            NDArray[np.int64]: Degree of each vertex, shape
            ``(n_vertices,)``, in the ``edge_color`` adjacency
            projection.
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
        """Return vertices having degree zero in one color projection.

        Args:
            edge_color (int): Color whose adjacency projection is used.

        Returns:
            NDArray[np.int32]: Indices of vertices with no incident
            edge of ``edge_color``.
        """
        degrees = self.vertex_degrees(edge_color=edge_color)

        return np.flatnonzero(degrees == 0).astype(np.int32)

    def packed(self) -> bytes:
        """Return a compact deterministic byte representation.

        For a two-color problem, colors are bit-packed (one bit per
        edge); otherwise the raw color array bytes are used.

        Returns:
            bytes: Compact byte encoding of ``colors``, deterministic
            for a given coloring.
        """
        if self.graph.problem.n_colors == 2:
            return np.packbits(
                self.colors,
                bitorder="little",
            ).tobytes()

        return self.colors.tobytes()

    def exact_hash(self) -> str:
        """Return an identifier containing problem and coloring data.

        Combines the problem's vertex count, forbidden clique sizes,
        and edge count with :meth:`packed` into a SHA-256 digest,
        suitable as a content-addressable key for archives and
        duplicate detection.

        Returns:
            str: Hexadecimal SHA-256 digest identifying this coloring.
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
        """Return whether another coloring has identical semantics.

        Unlike default dataclass equality (disabled here via
        ``eq=False``), this compares the problem definition and color
        values rather than object identity.

        Args:
            other (object): Value to compare against.

        Returns:
            bool: ``True`` if ``other`` is an ``RColoring`` with an
            equal problem and identical colors.
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
        """Validate a color index used for a graph projection.

        Args:
            color (int): Color index to validate.

        Raises:
            IndexError: If ``color`` is negative or not less than the
                problem's number of colors.
        """
        if color < 0 or color >= self.graph.problem.n_colors:
            raise IndexError(f"Invalid color index: {color}")
