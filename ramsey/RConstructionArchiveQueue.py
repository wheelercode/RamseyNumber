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
    """Validate an optional nonnegative integer."""
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
    """Validate an optional positive integer."""
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