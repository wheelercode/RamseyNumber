"""Interchangeable construction of immutable seed colorings."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .RColoring import RColoring
from .RGraph import RGraph

EXOO_RED_DISTANCES = frozenset(
    {
        1,
        2,
        7,
        10,
        12,
        13,
        14,
        16,
        18,
        20,
        21,
    }
)

EXOO_BLUE_DISTANCES = frozenset(
    {
        3,
        4,
        5,
        6,
        8,
        9,
        11,
        15,
        17,
        19,
    }
)


class RConstruction(ABC):
    """
    Create one immutable seed coloring for a supplied graph.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return a stable construction name.
        """
        ...

    @abstractmethod
    def construct(
        self,
        graph: RGraph,
    ) -> RColoring:
        """
        Construct one coloring of the supplied graph.
        """
        ...


@dataclass(slots=True)
class RRandomConstruction(RConstruction):
    """
    Assign every edge color independently and uniformly.
    """

    rng: np.random.Generator

    @property
    def name(self) -> str:
        return "random"

    def construct(
        self,
        graph: RGraph,
    ) -> RColoring:
        colors = self.rng.integers(
            0,
            graph.problem.n_colors,
            size=graph.number_of_edges,
            dtype=np.uint8,
        )

        return RColoring(
            graph,
            colors,
        )


@dataclass(frozen=True, slots=True)
class RCyclicConstruction(RConstruction):
    """
    Color edges according to circular vertex distance.

    colors_by_distance[d - 1] specifies the color assigned to
    every edge having circular distance d.
    """

    colors_by_distance: tuple[int, ...]
    construction_name: str = "cyclic"

    def __post_init__(self) -> None:
        colors = tuple(int(color) for color in self.colors_by_distance)

        if not colors:
            raise ValueError("colors_by_distance cannot be empty.")

        if not self.construction_name:
            raise ValueError("construction_name cannot be empty.")

        object.__setattr__(
            self,
            "colors_by_distance",
            colors,
        )

    @property
    def name(self) -> str:
        return self.construction_name

    def construct(
        self,
        graph: RGraph,
    ) -> RColoring:
        n_vertices = graph.problem.n_vertices

        expected_distances = n_vertices // 2

        if len(self.colors_by_distance) != expected_distances:
            raise ValueError(
                f"K_{n_vertices} has "
                f"{expected_distances} circular edge "
                "distances, so colors_by_distance must "
                f"contain {expected_distances} entries."
            )

        colors_by_distance = np.asarray(
            self.colors_by_distance,
            dtype=np.int64,
        )

        if np.any(colors_by_distance < 0) or np.any(
            colors_by_distance >= graph.problem.n_colors
        ):
            raise ValueError(
                "colors_by_distance contains a color " "outside the problem."
            )

        differences = np.abs(
            graph.edges[:, 0].astype(np.int16) - graph.edges[:, 1].astype(np.int16)
        )

        circular_distances = np.minimum(
            differences,
            n_vertices - differences,
        )

        colors = colors_by_distance[circular_distances - 1].astype(np.uint8)

        return RColoring(
            graph,
            colors,
        )

    @classmethod
    def exoo(
        cls,
    ) -> "RCyclicConstruction":
        """
        Return Exoo's original cyclic K43 construction.
        """
        valid_distances = EXOO_RED_DISTANCES | EXOO_BLUE_DISTANCES

        if valid_distances != set(range(1, 22)):
            raise RuntimeError(
                "Exoo distance sets must partition " "distances 1 through 21."
            )

        colors = tuple(
            (1 if distance in EXOO_BLUE_DISTANCES else 0) for distance in range(1, 22)
        )

        return cls(
            colors,
            construction_name=("exoo-cyclic-k43"),
        )


@dataclass(frozen=True, slots=True)
class RFixedConstruction(RConstruction):
    """
    Return a fixed coloring.

    If an equivalent RGraph instance is supplied, the coloring is
    rebound to that graph without changing any colors.
    """

    coloring: RColoring
    construction_name: str = "fixed"

    def __post_init__(self) -> None:
        if not self.construction_name:
            raise ValueError("construction_name cannot be empty.")

    @property
    def name(self) -> str:
        return self.construction_name

    def construct(
        self,
        graph: RGraph,
    ) -> RColoring:
        if self.coloring.graph.problem != graph.problem:
            raise ValueError(
                "Fixed coloring problem does not " "match the supplied graph."
            )

        return RColoring(
            graph,
            self.coloring.colors,
        )
