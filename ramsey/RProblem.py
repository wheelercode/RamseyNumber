"""Mathematical specifications for finite Ramsey coloring problems."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from numbers import Integral


@dataclass(frozen=True, slots=True)
class RProblem:
    """
    Immutable definition of one classical Ramsey coloring problem.
    """

    n_vertices: int
    forbidden_clique_sizes: tuple[int, ...]

    def __post_init__(self) -> None:
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
        """
        return len(self.forbidden_clique_sizes)

    @property
    def edge_count(self) -> int:
        """
        Return the number of edges in the complete host graph.
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
        """
        return tuple(sorted(set(self.forbidden_clique_sizes)))

    @property
    def is_symmetric(self) -> bool:
        """
        Return whether every color forbids the same clique size.
        """
        return len(self.required_clique_sizes) == 1

    def forbidden_clique_size(
        self,
        color: int,
    ) -> int:
        """
        Return the forbidden clique size for one color.
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
        """
        return cls.symmetric(
            n_vertices=n_vertices,
            clique_size=5,
            n_colors=2,
        )
