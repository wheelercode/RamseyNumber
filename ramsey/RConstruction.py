"""Interchangeable construction of immutable seed colorings."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from numbers import Real

import numpy as np

from .RArchive import (
    RArchive,
    RArchiveRecord,
)
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

    @property
    def last_source_name(self) -> str:
        """
        Return the construction that produced the most recent seed.

        Ordinary constructions produce their own stable name. Composite
        constructions override this property to identify the selected child.
        """
        return self.name


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


@dataclass(slots=True)
class RArchiveConstruction(RConstruction):
    """
    Sample seed colorings uniformly from an archive score range.
    """

    archive: RArchive
    rng: np.random.Generator
    minimum_score: int | None = None
    maximum_score: int | None = None
    limit: int | None = None

    _last_record: RArchiveRecord | None = field(
        init=False,
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.archive, RArchive):
            raise TypeError("archive must implement RArchive.")

        if not isinstance(
            self.rng,
            np.random.Generator,
        ):
            raise TypeError("rng must be a NumPy Generator.")

        self.minimum_score = _optional_nonnegative_integer(
            "minimum_score",
            self.minimum_score,
        )

        self.maximum_score = _optional_nonnegative_integer(
            "maximum_score",
            self.maximum_score,
        )

        if (
            self.minimum_score is not None
            and self.maximum_score is not None
            and self.minimum_score > self.maximum_score
        ):
            raise ValueError(
                "minimum_score cannot be greater than "
                "maximum_score."
            )

        self.limit = _optional_positive_integer(
            "limit",
            self.limit,
        )

    @property
    def name(self) -> str:
        lower = (
            "any"
            if self.minimum_score is None
            else str(self.minimum_score)
        )

        upper = (
            "any"
            if self.maximum_score is None
            else str(self.maximum_score)
        )

        return f"archive-score-{lower}-to-{upper}"

    @property
    def last_record(self) -> RArchiveRecord | None:
        """
        Return the record selected by the most recent construction.
        """
        return self._last_record

    def eligible_records(
        self,
        graph: RGraph,
    ) -> list[RArchiveRecord]:
        """
        Return the current archive pool eligible for sampling.
        """
        return self.archive.colorings_in_score_range(
            minimum_score=self.minimum_score,
            maximum_score=self.maximum_score,
            limit=self.limit,
            graph=graph,
        )

    def construct(
        self,
        graph: RGraph,
    ) -> RColoring:
        records = self.eligible_records(graph)

        if not records:
            raise RuntimeError(
                "Archive contains no colorings in the requested "
                "score range."
            )

        selected_index = int(
            self.rng.integers(
                0,
                len(records),
            )
        )

        self._last_record = records[selected_index]

        archived = self.archive.load_coloring(
            self._last_record.coloring_id,
            graph,
        )

        return archived.coloring


@dataclass(slots=True)
class RMixedConstruction(RConstruction):
    """
    Select among seed constructions using fixed probabilities.
    """

    constructions: tuple[RConstruction, ...]
    probabilities: tuple[float, ...]
    rng: np.random.Generator
    construction_name: str = "mixed"

    _last_source_name: str = field(
        init=False,
        default="mixed",
        repr=False,
    )

    def __post_init__(self) -> None:
        self.constructions = tuple(self.constructions)
        self.probabilities = tuple(self.probabilities)

        if not self.constructions:
            raise ValueError("constructions cannot be empty.")

        if not all(
            isinstance(construction, RConstruction)
            for construction in self.constructions
        ):
            raise TypeError(
                "Every construction must implement RConstruction."
            )

        if len(self.probabilities) != len(self.constructions):
            raise ValueError(
                "probabilities must contain one value per "
                "construction."
            )

        normalized_probabilities: list[float] = []

        for probability in self.probabilities:
            if isinstance(probability, bool) or not isinstance(
                probability,
                Real,
            ):
                raise TypeError("Every probability must be numeric.")

            probability = float(probability)

            if not np.isfinite(probability) or probability < 0.0:
                raise ValueError(
                    "Every probability must be finite and "
                    "nonnegative."
                )

            normalized_probabilities.append(probability)

        if not np.isclose(
            sum(normalized_probabilities),
            1.0,
        ):
            raise ValueError("probabilities must sum to one.")

        self.probabilities = tuple(normalized_probabilities)

        if not isinstance(
            self.rng,
            np.random.Generator,
        ):
            raise TypeError("rng must be a NumPy Generator.")

        if not isinstance(
            self.construction_name,
            str,
        ) or not self.construction_name.strip():
            raise ValueError(
                "construction_name must be a nonempty string."
            )

        self._last_source_name = self.construction_name

    @property
    def name(self) -> str:
        return self.construction_name

    @property
    def last_source_name(self) -> str:
        return self._last_source_name

    def construct(
        self,
        graph: RGraph,
    ) -> RColoring:
        selected_index = int(
            self.rng.choice(
                len(self.constructions),
                p=np.asarray(
                    self.probabilities,
                    dtype=np.float64,
                ),
            )
        )

        selected = self.constructions[selected_index]

        coloring = selected.construct(graph)

        self._last_source_name = selected.last_source_name

        return coloring


def _optional_nonnegative_integer(
    name: str,
    value: int | None,
) -> int | None:
    """
    Validate an optional nonnegative integer.
    """
    if value is None:
        return None

    if isinstance(value, bool) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(f"{name} must be an integer or None.")

    value = int(value)

    if value < 0:
        raise ValueError(f"{name} cannot be negative.")

    return value


def _optional_positive_integer(
    name: str,
    value: int | None,
) -> int | None:
    """
    Validate an optional positive integer.
    """
    value = _optional_nonnegative_integer(
        name,
        value,
    )

    if value == 0:
        raise ValueError(
            f"{name} must be positive when supplied."
        )

    return value