"""Mathematical specifications for finite Ramsey coloring problems."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from numbers import Integral


@dataclass(frozen=True, slots=True)
class RProblem:
    """
    Immutable definition of one classical Ramsey coloring problem.

    A problem fixes the order of the complete host graph and, for each
    color, the size of the monochromatic clique that a coloring must
    avoid in that color to be a valid construction.

    Attributes:
        n_vertices (int): Number of vertices in the complete host graph.
        forbidden_clique_sizes (tuple[int, ...]): Forbidden clique size
            for each color, indexed by color. Its length is the number
            of colors in the problem.
    """

    n_vertices: int
    forbidden_clique_sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate and normalize the problem definition.

        Raises:
            TypeError: If ``n_vertices`` or any forbidden clique size is
                not an integer.
            ValueError: If ``n_vertices`` is less than two, fewer than
                two colors/clique sizes are supplied, any forbidden
                clique size is less than two, or any forbidden clique
                size exceeds ``n_vertices``.
        """
        if isinstance(self.n_vertices, bool) or not isinstance(
            self.n_vertices,
            Integral,
        ):
            raise TypeError("n_vertices must be an integer.")

        n_vertices = int(self.n_vertices)

        if n_vertices < 2:
            raise ValueError("n_vertices must be at least two.")

        supplied_clique_sizes = tuple(self.forbidden_clique_sizes)

        if any(
            isinstance(size, bool) or not isinstance(size, Integral)
            for size in supplied_clique_sizes
        ):
            raise TypeError("Every forbidden clique size must be an integer.")

        clique_sizes = tuple(int(size) for size in supplied_clique_sizes)

        if len(clique_sizes) < 2:
            raise ValueError(
                "At least two colors and forbidden " "clique sizes are required."
            )

        for clique_size in clique_sizes:
            if clique_size < 2:
                raise ValueError("Every forbidden clique size " "must be at least two.")

            if clique_size > n_vertices:
                raise ValueError(
                    "A forbidden clique cannot contain "
                    "more vertices than the host graph."
                )

        object.__setattr__(
            self,
            "n_vertices",
            n_vertices,
        )

        object.__setattr__(
            self,
            "forbidden_clique_sizes",
            clique_sizes,
        )

    @property
    def n_colors(self) -> int:
        """
        Return the number of edge colors in the problem.

        Returns:
            int: Number of colors, equal to the length of
            ``forbidden_clique_sizes``.
        """
        return len(self.forbidden_clique_sizes)

    @property
    def edge_count(self) -> int:
        """
        Return the number of edges in the complete host graph.

        Returns:
            int: ``n_vertices`` choose 2.
        """
        return comb(
            self.n_vertices,
            2,
        )

    @property
    def required_clique_sizes(
        self,
    ) -> tuple[int, ...]:
        """
        Return the distinct clique sizes that must be indexed.

        Returns:
            tuple[int, ...]: The sorted, deduplicated set of forbidden
            clique sizes across all colors. A search-state or graph
            index only needs to enumerate cliques of these sizes.
        """
        return tuple(sorted(set(self.forbidden_clique_sizes)))

    @property
    def is_symmetric(self) -> bool:
        """
        Return whether every color forbids the same clique size.

        Returns:
            bool: True when ``required_clique_sizes`` contains exactly
            one value, i.e. the problem is a classical symmetric
            Ramsey number (such as R(5, 5)).
        """
        return len(self.required_clique_sizes) == 1

    def forbidden_clique_size(
        self,
        color: int,
    ) -> int:
        """
        Return the forbidden clique size for one color.

        Args:
            color (int): Zero-based color index.

        Returns:
            int: Size of the monochromatic clique forbidden in
            ``color``.

        Raises:
            IndexError: If ``color`` is not a valid color index.
        """
        if color < 0 or color >= self.n_colors:
            raise IndexError(f"Invalid color index: {color}")

        return self.forbidden_clique_sizes[color]

    def clique_count(
        self,
        clique_size: int,
    ) -> int:
        """
        Return the number of vertex subsets of the requested size.

        Args:
            clique_size (int): Clique size to count vertex subsets for.

        Returns:
            int: ``n_vertices`` choose ``clique_size``.

        Raises:
            ValueError: If ``clique_size`` is outside ``[2, n_vertices]``.
        """
        self._validate_clique_size(clique_size)

        return comb(
            self.n_vertices,
            clique_size,
        )

    def edges_per_clique(
        self,
        clique_size: int,
    ) -> int:
        """
        Return the number of edges in a clique.

        Args:
            clique_size (int): Clique size (number of vertices).

        Returns:
            int: ``clique_size`` choose 2.

        Raises:
            ValueError: If ``clique_size`` is outside ``[2, n_vertices]``.
        """
        self._validate_clique_size(clique_size)

        return comb(
            clique_size,
            2,
        )

    def cliques_per_edge(
        self,
        clique_size: int,
    ) -> int:
        """
        Return how many requested cliques contain one fixed edge.

        Args:
            clique_size (int): Clique size (number of vertices).

        Returns:
            int: ``n_vertices - 2`` choose ``clique_size - 2``, the
            number of cliques of the requested size that contain any
            given edge.

        Raises:
            ValueError: If ``clique_size`` is outside ``[2, n_vertices]``.
        """
        self._validate_clique_size(clique_size)

        return comb(
            self.n_vertices - 2,
            clique_size - 2,
        )

    def _validate_clique_size(
        self,
        clique_size: int,
    ) -> None:
        """
        Validate a clique size used in a derived calculation.

        Args:
            clique_size (int): Candidate clique size.

        Raises:
            ValueError: If ``clique_size`` is outside ``[2, n_vertices]``.
        """
        if clique_size < 2 or clique_size > self.n_vertices:
            raise ValueError("clique_size must be between " "two and n_vertices.")

    @classmethod
    def symmetric(
        cls,
        *,
        n_vertices: int,
        clique_size: int,
        n_colors: int = 2,
    ) -> "RProblem":
        """
        Construct a problem with the same forbidden size per color.

        Args:
            n_vertices (int): Number of vertices in the host graph.
            clique_size (int): Forbidden clique size shared by every
                color.
            n_colors (int): Number of colors. Defaults to 2.

        Returns:
            RProblem: A problem whose ``forbidden_clique_sizes`` repeats
            ``clique_size`` once per color.

        Raises:
            ValueError: If ``n_colors`` is less than two.
        """
        if n_colors < 2:
            raise ValueError("n_colors must be at least two.")

        return cls(
            n_vertices=n_vertices,
            forbidden_clique_sizes=(clique_size,) * n_colors,
        )

    @classmethod
    def r55(
        cls,
        n_vertices: int = 43,
    ) -> "RProblem":
        """
        Construct the two-color K5 problem at the requested order.

        This is the classical R(5, 5) lower-bound construction problem:
        color the edges of the complete graph on ``n_vertices`` vertices
        with two colors while avoiding any monochromatic K5.

        Args:
            n_vertices (int): Number of vertices in the host graph.
                Defaults to 43.

        Returns:
            RProblem: A symmetric two-color problem forbidding K5 in
            each color.
        """
        return cls.symmetric(
            n_vertices=n_vertices,
            clique_size=5,
            n_colors=2,
        )
