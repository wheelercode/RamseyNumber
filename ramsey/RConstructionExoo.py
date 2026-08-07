"""Published Exoo zero-score construction for the R(5,5) K42 graph.

Reproduces the specific 42-vertex, two-color, K5-free coloring published
by Exoo as a lower-bound construction for the Ramsey number R(5,5), by
deriving it deterministically from the cyclic K43 construction plus a
fixed list of edge corrections.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .RColoring import RColoring
from .RConstruction import (
    EXOO_BLUE_DISTANCES,
    RConstruction,
)
from .RGraph import RGraph


# Exoo(42) starts from Cyclic(43), removes original vertex 0, and
# changes these sixteen red edges to blue.  The surviving original
# labels are 1..42; RGraph itself continues to use canonical labels
# 0..41 internally.
EXOO_42_BLUE_CORRECTIONS = (
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (11, 32),
    (13, 14),
    (14, 15),
    (15, 16),
    (16, 17),
    (23, 24),
    (24, 25),
    (30, 31),
    (33, 34),
    (39, 40),
    (40, 41),
    (41, 42),
)


@dataclass(
    frozen=True,
    slots=True,
)
class RConstructionExoo(RConstruction):
    """
    Construct Exoo's published zero-score red/blue coloring of K42.

    The construction follows Exoo(42) as restated and proved in:

        Ge, Jayasooriya, Qiu, Sun, Yuan (2023),
        "Study of Exoo's Lower Bound for Ramsey number R(5,5)",
        arXiv:2212.12630, Definition 1.2.

    It begins with the published Cyclic(43) distance coloring, deletes
    vertex 0, and changes sixteen specified red edges to blue.

    RGraph labels the resulting K42 vertices 0..41.  Internally this
    class maps them to Exoo's surviving labels 1..42 while constructing
    the coloring, then returns the ordinary canonical RColoring.
    """

    @property
    def name(self) -> str:
        """
        str: Always ``"exoo-k42"``.
        """
        return "exoo-k42"

    def construct(
        self,
        graph: RGraph,
    ) -> RColoring:
        """
        Build Exoo's published K42 coloring for the supplied graph.

        Computes the Cyclic(43) distance coloring under the label shift
        implied by deleting original vertex 0 (canonical vertex ``k``
        maps to original vertex ``k + 1``), then flips the sixteen edges
        listed in :data:`EXOO_42_BLUE_CORRECTIONS` from red to blue.

        Args:
            graph (RGraph): Host graph to color. Must have exactly 42
                vertices and a symmetric two-color problem.

        Returns:
            RColoring: The published Exoo zero-score coloring of K42.

        Raises:
            ValueError: If ``graph.problem`` does not have exactly 42
                vertices, or is not a symmetric two-color problem.
            RuntimeError: If a correction edge in
                :data:`EXOO_42_BLUE_CORRECTIONS` was not red in the
                Cyclic(43) precursor coloring, indicating the
                correction list is inconsistent with the base cyclic
                coloring.
        """
        problem = graph.problem

        if problem.n_vertices != 42:
            raise ValueError(
                "RConstructionExoo requires exactly 42 vertices."
            )

        if problem.n_colors != 2 or not problem.is_symmetric:
            raise ValueError(
                "RConstructionExoo requires a symmetric "
                "two-color problem."
            )

        colors = np.empty(
            graph.number_of_edges,
            dtype=np.uint8,
        )

        # Canonical K42 vertex k corresponds to original Cyclic(43)
        # vertex k+1 because published vertex 0 was deleted.
        first = (
            graph.edges[:, 0].astype(np.int16)
            + 1
        )
        second = (
            graph.edges[:, 1].astype(np.int16)
            + 1
        )

        differences = np.abs(
            first - second
        )

        circular_distances = np.minimum(
            differences,
            43 - differences,
        )

        blue_distances = np.asarray(
            sorted(EXOO_BLUE_DISTANCES),
            dtype=np.int16,
        )

        colors[:] = np.isin(
            circular_distances,
            blue_distances,
        ).astype(np.uint8)

        edge_lookup = {
            (
                int(u) + 1,
                int(v) + 1,
            ): edge_index
            for edge_index, (u, v) in enumerate(
                graph.edges
            )
        }

        for correction in EXOO_42_BLUE_CORRECTIONS:
            edge_index = edge_lookup[
                correction
            ]

            if colors[edge_index] != 0:
                raise RuntimeError(
                    "Published Exoo correction edge "
                    f"{correction} was not red in "
                    "the Cyclic(43) precursor."
                )

            colors[edge_index] = 1

        return RColoring(
            graph,
            colors,
        )