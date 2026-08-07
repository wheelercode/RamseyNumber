"""Interchangeable construction strategies for immutable seed colorings.

Defines the :class:`RConstruction` abstract interface together with several
concrete strategies (random, cyclic, fixed, and archive-backed) that each
build one immutable :class:`~ramsey.RColoring.RColoring` for a supplied
:class:`~ramsey.RGraph.RGraph`. Search code selects a construction and calls
``construct()`` without needing to know the mechanism used to produce the
seed coloring.
"""

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

# Circular distances (in the 43-vertex cycle) assigned red and blue,
# respectively, by Exoo's published Cyclic(43) construction. Together
# EXOO_RED_DISTANCES and EXOO_BLUE_DISTANCES partition every distance
# from 1 to 21.
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
    Abstract strategy that produces one immutable seed coloring.

    Concrete subclasses implement :meth:`construct` to build a single
    coloring for a supplied graph, together with a stable :attr:`name`
    identifying the strategy.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        str: Stable identifier for this construction strategy.
        """
        ...

    @abstractmethod
    def construct(
        self,
        graph: RGraph,
    ) -> RColoring:
        """
        Build one seed coloring for the supplied graph.

        Args:
            graph (RGraph): Host graph whose edges the returned coloring
                assigns colors to.

        Returns:
            RColoring: A freshly constructed coloring bound to ``graph``.
        """
        ...

    @property
    def last_source_name(self) -> str:
        """
        str: Name of the construction that produced the most recent seed.

        Ordinary constructions produce their own stable name. Composite
        constructions override this property to identify the selected child.
        """
        return self.name


@dataclass(slots=True)
class RRandomConstruction(RConstruction):
    """
    Assign every edge a color independently and uniformly.

    Attributes:
        rng (numpy.random.Generator): Source of randomness used to draw
            each edge's color.
    """

    rng: np.random.Generator

    @property
    def name(self) -> str:
        """
        str: Always ``"random"``.
        """
        return "random"

    def construct(
        self,
        graph: RGraph,
    ) -> RColoring:
        """
        Color every edge of ``graph`` with an independent uniform draw.

        Args:
            graph (RGraph): Host graph to color.

        Returns:
            RColoring: A coloring whose edge colors are drawn i.i.d.
            uniformly from the problem's ``n_colors`` color values.
        """
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

    ``colors_by_distance[d - 1]`` specifies the color assigned to every
    edge having circular distance ``d``, i.e. ``min(|u - v|, n_vertices -
    |u - v|)`` for endpoints ``u`` and ``v``. This reproduces circulant
    Ramsey colorings such as Exoo's cyclic K43 construction.

    Attributes:
        colors_by_distance (tuple[int, ...]): Color index assigned to each
            circular distance, indexed from distance 1 through
            ``n_vertices // 2``. Coerced to a tuple of ``int`` in
            ``__post_init__``.
        construction_name (str): Stable name reported by :attr:`name`.
    """

    colors_by_distance: tuple[int, ...]
    construction_name: str = "cyclic"

    def __post_init__(self) -> None:
        """
        Coerce ``colors_by_distance`` to ``int`` and validate arguments.

        Raises:
            ValueError: If ``colors_by_distance`` is empty or
                ``construction_name`` is empty.
        """
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
        """
        str: The configured ``construction_name``.
        """
        return self.construction_name

    def construct(
        self,
        graph: RGraph,
    ) -> RColoring:
        """
        Color ``graph`` by circular vertex distance.

        For every edge, computes the circular distance between its two
        endpoints and looks up the corresponding entry of
        ``colors_by_distance``.

        Args:
            graph (RGraph): Host graph to color. Its vertex count must be
                even, with exactly ``n_vertices // 2`` distinct circular
                distances.

        Returns:
            RColoring: A coloring determined entirely by circular
            distance.

        Raises:
            ValueError: If the number of circular distances implied by
                ``graph`` does not match ``len(colors_by_distance)``, or
                if ``colors_by_distance`` contains a color index outside
                the problem's color set.
        """
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
        Build Exoo's original cyclic K43 construction.

        Assigns blue (color 1) to every circular distance in
        :data:`EXOO_BLUE_DISTANCES` and red (color 0) to every distance in
        :data:`EXOO_RED_DISTANCES`; together these sets partition every
        circular distance from 1 to 21 in the 43-vertex cycle.

        Returns:
            RCyclicConstruction: A construction named
            ``"exoo-cyclic-k43"`` reproducing the published coloring.

        Raises:
            RuntimeError: If :data:`EXOO_RED_DISTANCES` and
                :data:`EXOO_BLUE_DISTANCES` do not together partition
                every distance from 1 to 21 (an internal consistency
                check on the module-level constants).
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
    Return a fixed coloring on every call.

    Useful for replaying a known coloring as a seed. If a graph describing
    the same Ramsey problem is supplied, the stored coloring's colors are
    rebound to that graph without changing any colors.

    Attributes:
        coloring (RColoring): The coloring whose colors are replayed.
        construction_name (str): Stable name reported by :attr:`name`.
    """

    coloring: RColoring
    construction_name: str = "fixed"

    def __post_init__(self) -> None:
        """
        Validate that ``construction_name`` is nonempty.

        Raises:
            ValueError: If ``construction_name`` is empty.
        """
        if not self.construction_name:
            raise ValueError("construction_name cannot be empty.")

    @property
    def name(self) -> str:
        """
        str: The configured ``construction_name``.
        """
        return self.construction_name

    def construct(
        self,
        graph: RGraph,
    ) -> RColoring:
        """
        Rebind the stored coloring's colors onto ``graph``.

        Args:
            graph (RGraph): Host graph to bind the fixed colors to. Must
                describe the same Ramsey problem as ``self.coloring``.

        Returns:
            RColoring: A new coloring over ``graph`` using
            ``self.coloring.colors`` unchanged.

        Raises:
            ValueError: If ``graph.problem`` does not equal the problem
                of the stored ``coloring``.
        """
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

    Queries an :class:`~ramsey.RArchive.RArchive` for every archived
    coloring whose exact score falls within ``[minimum_score,
    maximum_score]`` (either bound may be omitted), optionally capped to
    the best ``limit`` records, and selects uniformly at random among
    them on every call to :meth:`construct`.

    Attributes:
        archive (RArchive): Archive queried for eligible colorings.
        rng (numpy.random.Generator): Source of randomness used to select
            among eligible records.
        minimum_score (int | None): Inclusive lower score bound, or
            ``None`` for no lower bound.
        maximum_score (int | None): Inclusive upper score bound, or
            ``None`` for no upper bound.
        limit (int | None): Maximum number of best-scoring eligible
            records to consider, or ``None`` for no cap.
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
        """
        Validate ``archive``, ``rng``, and the score/limit bounds.

        Raises:
            TypeError: If ``archive`` does not implement
                :class:`RArchive`, or ``rng`` is not a NumPy
                ``Generator``.
            ValueError: If ``minimum_score`` and ``maximum_score`` are
                both supplied with ``minimum_score`` greater than
                ``maximum_score``, or if a score/limit value fails
                integer validation.
        """
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
        """
        str: Name encoding the configured score range, e.g.
        ``"archive-score-0-to-5"`` or ``"archive-score-any-to-any"``.
        """
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
        RArchiveRecord | None: The archive record selected by the most
        recent call to :meth:`construct`, or ``None`` beforehand.
        """
        return self._last_record

    def eligible_records(
        self,
        graph: RGraph,
    ) -> list[RArchiveRecord]:
        """
        Return the current archive pool eligible for sampling.

        Args:
            graph (RGraph): Host graph identifying which Ramsey
                problem's archive to query.

        Returns:
            list[RArchiveRecord]: Every archived record within the
            configured score range (and ``limit``) for ``graph``'s
            problem.
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
        """
        Sample one archived coloring uniformly at random.

        Args:
            graph (RGraph): Host graph to bind the sampled coloring to.

        Returns:
            RColoring: The coloring loaded from the selected archive
            record.

        Raises:
            RuntimeError: If no archived coloring falls within the
                configured score range for ``graph``'s problem.
        """
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

    On each call to :meth:`construct`, one child construction is drawn
    according to fixed ``probabilities`` (which must be nonnegative,
    finite, and sum to one) and delegated to for building the seed
    coloring.

    Attributes:
        constructions (tuple[RConstruction, ...]): Candidate child
            constructions.
        probabilities (tuple[float, ...]): Selection probability for each
            entry of ``constructions``, aligned by index.
        rng (numpy.random.Generator): Source of randomness used to
            select the child construction.
        construction_name (str): Stable name reported by :attr:`name`.
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
        """
        Normalize and validate the constructions, probabilities, and rng.

        Raises:
            TypeError: If any entry of ``constructions`` does not
                implement :class:`RConstruction`, any probability is
                non-numeric, or ``rng`` is not a NumPy ``Generator``.
            ValueError: If ``constructions`` is empty, ``probabilities``
                does not have one entry per construction, any
                probability is non-finite or negative, the probabilities
                do not sum to one, or ``construction_name`` is empty or
                whitespace.
        """
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
        """
        str: The configured ``construction_name``.
        """
        return self.construction_name

    @property
    def last_source_name(self) -> str:
        """
        str: The ``last_source_name`` of the child construction selected
        by the most recent call to :meth:`construct`, or
        ``construction_name`` beforehand.
        """
        return self._last_source_name

    def construct(
        self,
        graph: RGraph,
    ) -> RColoring:
        """
        Select one child construction by ``probabilities`` and delegate.

        Args:
            graph (RGraph): Host graph passed through to the selected
                child construction.

        Returns:
            RColoring: The coloring produced by the selected child
            construction.
        """
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

@dataclass(slots=True)
class RArchiveSnapshotConstruction(RConstruction):
    """
    Consume a fixed archive score-range snapshot without replacement.

    The eligible records are queried once, shuffled once, and then
    consumed exactly once each. Records subsequently added to the
    archive cannot enter this seed pool.

    Attributes:
        archive (RArchive): Archive queried for the snapshot.
        rng (numpy.random.Generator): Source of randomness used to
            shuffle the snapshot.
        minimum_score (int | None): Inclusive lower score bound, or
            ``None`` for no lower bound.
        maximum_score (int | None): Inclusive upper score bound, or
            ``None`` for no upper bound.
        limit (int | None): Maximum number of best-scoring eligible
            records to include in the snapshot, or ``None`` for no cap.
    """

    archive: RArchive
    rng: np.random.Generator
    minimum_score: int | None = None
    maximum_score: int | None = None
    limit: int | None = None

    _records: tuple[RArchiveRecord, ...] | None = field(
        init=False,
        default=None,
        repr=False,
    )

    _problem: object | None = field(
        init=False,
        default=None,
        repr=False,
    )

    _next_record: int = field(
        init=False,
        default=0,
        repr=False,
    )

    _last_record: RArchiveRecord | None = field(
        init=False,
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        """
        Validate ``archive``, ``rng``, and the score/limit bounds.

        Raises:
            TypeError: If ``archive`` does not implement
                :class:`RArchive`, or ``rng`` is not a NumPy
                ``Generator``.
            ValueError: If ``minimum_score`` and ``maximum_score`` are
                both supplied with ``minimum_score`` greater than
                ``maximum_score``, or if a score/limit value fails
                integer validation.
        """
        if not isinstance(self.archive, RArchive):
            raise TypeError(
                "archive must implement RArchive."
            )

        if not isinstance(
            self.rng,
            np.random.Generator,
        ):
            raise TypeError(
                "rng must be a NumPy Generator."
            )

        self.minimum_score = (
            _optional_nonnegative_integer(
                "minimum_score",
                self.minimum_score,
            )
        )

        self.maximum_score = (
            _optional_nonnegative_integer(
                "maximum_score",
                self.maximum_score,
            )
        )

        if (
            self.minimum_score is not None
            and self.maximum_score is not None
            and self.minimum_score
            > self.maximum_score
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
        """
        str: Name encoding the configured score range, e.g.
        ``"archive-snapshot-score-0-to-5"``.
        """
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

        return (
            f"archive-snapshot-score-"
            f"{lower}-to-{upper}"
        )

    @property
    def last_record(
        self,
    ) -> RArchiveRecord | None:
        """
        RArchiveRecord | None: The record selected by the most recent
        call to :meth:`construct`, or ``None`` beforehand.
        """
        return self._last_record

    @property
    def prepared(self) -> bool:
        """
        bool: Whether the snapshot has been queried and shuffled yet.
        """
        return self._records is not None

    @property
    def snapshot_size(self) -> int:
        """
        int: Total number of records in the frozen snapshot, or 0
        before preparation.
        """
        if self._records is None:
            return 0

        return len(self._records)

    @property
    def remaining_count(self) -> int:
        """
        int: Number of snapshot records not yet consumed, or 0 before
        preparation.
        """
        if self._records is None:
            return 0

        return (
            len(self._records)
            - self._next_record
        )

    def prepare(
        self,
        graph: RGraph,
    ) -> int:
        """
        Freeze and shuffle the currently eligible archive records.

        Idempotent once prepared for a given problem: a later call for
        the same problem simply returns the existing snapshot size
        without re-querying the archive.

        Args:
            graph (RGraph): Host graph identifying which Ramsey
                problem's archive to query.

        Returns:
            int: Number of records in the (possibly newly created)
            snapshot.

        Raises:
            ValueError: If the snapshot was already prepared for a
                different Ramsey problem than ``graph.problem``.
            RuntimeError: If no archived coloring falls within the
                configured score range for ``graph``'s problem.
        """
        if self._records is not None:
            if graph.problem != self._problem:
                raise ValueError(
                    "Prepared archive snapshot belongs "
                    "to a different Ramsey problem."
                )

            return len(self._records)

        records = (
            self.archive.colorings_in_score_range(
                minimum_score=self.minimum_score,
                maximum_score=self.maximum_score,
                limit=self.limit,
                graph=graph,
            )
        )

        if not records:
            raise RuntimeError(
                "Archive contains no colorings in the "
                "requested snapshot score range."
            )

        order = self.rng.permutation(
            len(records)
        )

        self._records = tuple(
            records[int(index)]
            for index in order
        )

        self._problem = graph.problem

        return len(self._records)

    def construct(
        self,
        graph: RGraph,
    ) -> RColoring:
        """
        Consume and return the next unconsumed snapshot record.

        Prepares the snapshot on first use (see :meth:`prepare`), then
        hands out the next record in the frozen shuffled order.

        Args:
            graph (RGraph): Host graph to bind the loaded coloring to.

        Returns:
            RColoring: The coloring loaded from the next snapshot
            record.

        Raises:
            RuntimeError: If no archived coloring falls within the
                configured score range for ``graph``'s problem, or if
                every snapshot record has already been consumed.
        """
        self.prepare(graph)

        if self._records is None:
            raise RuntimeError(
                "Archive snapshot was not prepared."
            )

        if (
            self._next_record
            >= len(self._records)
        ):
            raise RuntimeError(
                "Archive snapshot is exhausted; "
                "every seed has already been "
                "consumed once."
            )

        self._last_record = (
            self._records[
                self._next_record
            ]
        )

        self._next_record += 1

        archived = self.archive.load_coloring(
            self._last_record.coloring_id,
            graph,
        )

        return archived.coloring

def _optional_nonnegative_integer(
    name: str,
    value: int | None,
) -> int | None:
    """
    Validate and coerce an optional nonnegative integer parameter.

    Args:
        name (str): Parameter name used in raised error messages.
        value (int | None): Candidate value; ``None`` is passed through
            unchanged.

    Returns:
        int | None: ``None`` if ``value`` is ``None``, otherwise
        ``value`` coerced to ``int``.

    Raises:
        TypeError: If ``value`` is neither ``None`` nor an integer
            (``bool`` values are rejected).
        ValueError: If ``value`` is negative.
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
    Validate and coerce an optional positive integer parameter.

    Args:
        name (str): Parameter name used in raised error messages.
        value (int | None): Candidate value; ``None`` is passed through
            unchanged.

    Returns:
        int | None: ``None`` if ``value`` is ``None``, otherwise
        ``value`` coerced to ``int``.

    Raises:
        TypeError: If ``value`` is neither ``None`` nor an integer
            (``bool`` values are rejected).
        ValueError: If ``value`` is zero or negative.
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