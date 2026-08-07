"""Live cyclic queue construction backed by a Ramsey archive."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral

import numpy as np

from .RArchive import (
    RArchive,
    RArchiveRecord,
)
from .RColoring import RColoring
from .RConstruction import RConstruction
from .RGraph import RGraph


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
        Integral,
    ):
        raise TypeError(f"{name} must be an integer.")

    result = int(value)

    if result < 0:
        raise ValueError(f"{name} cannot be negative.")

    return result


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
    result = _optional_nonnegative_integer(
        name,
        value,
    )

    if result == 0:
        raise ValueError(f"{name} must be positive.")

    return result


@dataclass(slots=True)
class RArchiveQueueConstruction(RConstruction):
    """
    Consume repeated, refreshed archive queues without replacement.

    One queue generation is frozen and shuffled, then every seed in
    it is consumed exactly once.  After it is exhausted, the archive
    is queried again.  Results archived during one generation can
    therefore become seeds in the next generation without changing
    the population halfway through the current generation.

    If ``limit`` is supplied, the archive query first selects the best
    ``limit`` eligible records by its normal score ordering and this
    construction then shuffles that active pool.

    Attributes:
        archive (RArchive): Archive queried for each queue generation.
        rng (numpy.random.Generator): Source of randomness used to
            shuffle each queue generation.
        minimum_score (int | None): Inclusive lower score bound, or
            ``None`` for no lower bound.
        maximum_score (int | None): Inclusive upper score bound, or
            ``None`` for no upper bound.
        limit (int | None): Maximum number of best-scoring eligible
            records to include in each queue generation, or ``None``
            for no cap.
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

    _generation: int = field(
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
        ``"archive-queue-score-0-to-5"``.
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
            f"archive-queue-score-{lower}-to-{upper}"
        )

    @property
    def last_record(
        self,
    ) -> RArchiveRecord | None:
        """Return the most recently consumed archive record."""
        return self._last_record

    @property
    def generation(self) -> int:
        """Return the one-based queue generation, or zero before use."""
        return self._generation

    @property
    def current_queue_size(self) -> int:
        """Return the frozen size of the current queue generation."""
        if self._records is None:
            return 0

        return len(self._records)

    @property
    def remaining_count(self) -> int:
        """Return unconsumed seeds in the current queue generation."""
        if self._records is None:
            return 0

        return (
            len(self._records)
            - self._next_record
        )

    def refresh_queue(
        self,
        graph: RGraph,
    ) -> int:
        """
        Refresh and shuffle the next queue generation from SQLite.

        Normal construct() usage calls this automatically after the
        current queue is exhausted.  Calling it explicitly abandons
        any unconsumed records in the current queue.

        Args:
            graph (RGraph): Host graph identifying which Ramsey
                problem's archive to query.

        Returns:
            int: Number of records in the freshly queried and
            shuffled queue generation.

        Raises:
            ValueError: If this construction was already used for a
                different Ramsey problem than ``graph.problem``.
            RuntimeError: If no archived coloring falls within the
                configured queue score range for ``graph``'s problem.
        """
        if (
            self._problem is not None
            and graph.problem != self._problem
        ):
            raise ValueError(
                "Archive queue construction belongs to a "
                "different Ramsey problem."
            )

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
                "requested queue score range."
            )

        order = self.rng.permutation(
            len(records)
        )

        self._records = tuple(
            records[int(index)]
            for index in order
        )

        self._problem = graph.problem
        self._next_record = 0
        self._generation += 1

        return len(self._records)

    def construct(
        self,
        graph: RGraph,
    ) -> RColoring:
        """
        Consume and return the next queued archive record.

        Refreshes the queue (see :meth:`refresh_queue`) whenever the
        current generation is missing or fully consumed, then hands out
        the next record in that generation's shuffled order.

        Args:
            graph (RGraph): Host graph to bind the loaded coloring to.

        Returns:
            RColoring: The coloring loaded from the next queued
            archive record.

        Raises:
            ValueError: If this construction was already used for a
                different Ramsey problem than ``graph.problem``.
            RuntimeError: If no archived coloring falls within the
                configured queue score range for ``graph``'s problem.
        """
        if (
            self._records is None
            or self._next_record
            >= len(self._records)
        ):
            self.refresh_queue(graph)

        if self._records is None:
            raise RuntimeError(
                "Archive queue was not initialized."
            )

        self._last_record = (
            self._records[self._next_record]
        )

        self._next_record += 1

        archived = self.archive.load_coloring(
            self._last_record.coloring_id,
            graph,
        )

        return archived.coloring