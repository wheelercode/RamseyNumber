"""Batch production of searched colorings for a persistent archive.

Provides :class:`RArchiveBatch`, a sequential driver that repeatedly
constructs a seed coloring, runs a local search on it, and saves the
result to an :class:`ramsey.RArchive.RArchive`, until a target number of
unique colorings within a score band has been archived or an attempt
budget is exhausted.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Callable

from .RArchive import RArchive, RArchiveRecord
from .RConstruction import RConstruction
from .RGraph import RGraph
from .RSearch import RSearch, RSearchResult


@dataclass(frozen=True, slots=True)
class RArchiveBatchConfig:
    """Settings for filling one archive score band.

    Attributes:
        run_name (str): Nonempty name recorded as provenance for every
            coloring saved during this batch.
        target_count (int): Number of unique, in-range colorings the
            batch should try to accumulate in the archive before
            stopping early.
        maximum_attempts (int): Upper bound on the number of
            construct-and-search attempts to perform.
        minimum_score (int | None): Inclusive lower bound on score for a
            coloring to count toward ``target_count``, or ``None`` for
            no lower bound.
        maximum_score (int | None): Inclusive upper bound on score for a
            coloring to count toward ``target_count``, or ``None`` for
            no upper bound.
        start_iteration (int): Iteration number assigned to the first
            attempt; each subsequent attempt increments it by one.
        record_steps (bool): Whether the underlying search should record
            per-step history (passed through to
            :meth:`ramsey.RSearch.RSearch.run`).
        save_out_of_range (bool): Whether colorings whose score falls
            outside ``[minimum_score, maximum_score]`` should still be
            saved to the archive (without counting toward
            ``target_count``).
    """

    run_name: str
    target_count: int
    maximum_attempts: int
    minimum_score: int | None = None
    maximum_score: int | None = None
    start_iteration: int = 0
    record_steps: bool = False
    save_out_of_range: bool = True

    def __post_init__(self) -> None:
        """Validate and normalize configuration values.

        Raises:
            ValueError: If ``run_name`` is empty, ``target_count`` or
                ``maximum_attempts`` is not positive, ``start_iteration``
                is negative, either score bound is negative, or
                ``minimum_score`` exceeds ``maximum_score``.
            TypeError: If ``target_count``, ``maximum_attempts``,
                ``start_iteration``, ``minimum_score``, or
                ``maximum_score`` is not an integer (or ``None`` where
                allowed), or if ``record_steps``/``save_out_of_range`` is
                not boolean.
        """
        if not isinstance(self.run_name, str) or not self.run_name.strip():
            raise ValueError("run_name must be a nonempty string.")

        for name in ("target_count", "maximum_attempts"):
            value = _positive_integer(name, getattr(self, name))
            object.__setattr__(self, name, value)

        start_iteration = _nonnegative_integer(
            "start_iteration",
            self.start_iteration,
        )
        object.__setattr__(self, "start_iteration", start_iteration)

        minimum_score = _optional_nonnegative_integer(
            "minimum_score",
            self.minimum_score,
        )
        maximum_score = _optional_nonnegative_integer(
            "maximum_score",
            self.maximum_score,
        )
        if (
            minimum_score is not None
            and maximum_score is not None
            and minimum_score > maximum_score
        ):
            raise ValueError(
                "minimum_score cannot be greater than maximum_score."
            )
        object.__setattr__(self, "minimum_score", minimum_score)
        object.__setattr__(self, "maximum_score", maximum_score)

        if not isinstance(self.record_steps, bool):
            raise TypeError("record_steps must be boolean.")
        if not isinstance(self.save_out_of_range, bool):
            raise TypeError("save_out_of_range must be boolean.")


@dataclass(frozen=True, slots=True)
class RArchiveBatchAttempt:
    """Outcome of one construction, search, and optional save.

    Attributes:
        attempt (int): Zero-based index of this attempt within the
            batch.
        iteration (int): Iteration number assigned to this attempt
            (``config.start_iteration + attempt``), recorded as
            provenance if the coloring is saved.
        construction_name (str): Name of the construction source that
            produced the seed coloring for this attempt.
        search_result (RSearchResult): Full result of running the search
            on the seed coloring.
        in_score_range (bool): Whether the search's best score falls
            within the batch's configured score band.
        archive_record (RArchiveRecord | None): Archive record for the
            saved coloring, or ``None`` if the coloring was not saved
            (out of range and ``save_out_of_range`` was false).
        new_unique_coloring (bool): Whether this attempt's coloring was
            not already present in the archive (``times_seen == 1``
            after saving).
        eligible_count (int): Running count of unique in-range colorings
            in the archive as of this attempt.
    """

    attempt: int
    iteration: int
    construction_name: str
    search_result: RSearchResult
    in_score_range: bool
    archive_record: RArchiveRecord | None
    new_unique_coloring: bool
    eligible_count: int


@dataclass(frozen=True, slots=True)
class RArchiveBatchResult:
    """Outcome of one bounded archive-population batch.

    Attributes:
        run_name (str): Run name the batch was executed under.
        target_count (int): Target number of unique in-range colorings
            the batch was trying to reach.
        maximum_attempts (int): Attempt budget the batch was allowed to
            use.
        initial_eligible_count (int): Number of unique in-range
            colorings already in the archive before this batch ran.
        final_eligible_count (int): Number of unique in-range colorings
            in the archive after this batch finished, re-queried from
            the archive so it stays exact even if another process wrote
            to it concurrently.
        attempt_results (tuple[RArchiveBatchAttempt, ...]): Per-attempt
            outcomes, in the order they were performed.
    """

    run_name: str
    target_count: int
    maximum_attempts: int
    initial_eligible_count: int
    final_eligible_count: int
    attempt_results: tuple[RArchiveBatchAttempt, ...]

    @property
    def attempts_completed(self) -> int:
        """int: Number of construct-and-search attempts performed."""
        return len(self.attempt_results)

    @property
    def target_reached(self) -> bool:
        """bool: Whether the archive holds at least ``target_count`` eligible colorings."""
        return self.final_eligible_count >= self.target_count

    @property
    def new_eligible_colorings(self) -> int:
        """int: Net increase in unique in-range colorings caused by this batch."""
        return self.final_eligible_count - self.initial_eligible_count

    @property
    def best_score(self) -> int | None:
        """int | None: Lowest best-score seen across all attempts, or ``None`` if no attempts ran."""
        if not self.attempt_results:
            return None
        return min(
            result.search_result.best_score
            for result in self.attempt_results
        )


RArchiveBatchObserver = Callable[[RArchiveBatchAttempt], None]
"""Callback invoked with each :class:`RArchiveBatchAttempt` as it completes."""


class RArchiveBatch:
    """Populate an archive by repeatedly constructing and searching seeds."""

    def __init__(
        self,
        *,
        graph: RGraph,
        construction: RConstruction,
        search: RSearch,
        archive: RArchive,
    ) -> None:
        """Bind a construction, search, and archive to one host graph.

        Args:
            graph (RGraph): Host graph that seed colorings and archive
                queries are scoped to.
            construction (RConstruction): Construction used to produce a
                seed coloring for each attempt.
            search (RSearch): Search used to improve each seed coloring.
                Its environment must target the same problem as
                ``graph``.
            archive (RArchive): Archive that discovered colorings are
                saved to and queried against.

        Raises:
            ValueError: If ``search``'s environment problem does not
                match ``graph``'s problem.
            TypeError: If ``archive`` does not implement
                :class:`ramsey.RArchive.RArchive`.
        """
        if search.environment.graph.problem != graph.problem:
            raise ValueError(
                "Search environment problem does not match batch graph."
            )
        if not isinstance(archive, RArchive):
            raise TypeError("archive must implement RArchive.")

        self._graph = graph
        self._construction = construction
        self._search = search
        self._archive = archive

    @property
    def graph(self) -> RGraph:
        """RGraph: Host graph seed colorings and archive queries are scoped to."""
        return self._graph

    @property
    def construction(self) -> RConstruction:
        """RConstruction: Construction used to produce seed colorings."""
        return self._construction

    @property
    def search(self) -> RSearch:
        """RSearch: Search used to improve each seed coloring."""
        return self._search

    @property
    def archive(self) -> RArchive:
        """RArchive: Archive that discovered colorings are saved to."""
        return self._archive

    def populate(
        self,
        config: RArchiveBatchConfig,
        *,
        observer: RArchiveBatchObserver | None = None,
    ) -> RArchiveBatchResult:
        """Search until the score band is full or attempts are exhausted.

        Repeatedly constructs a seed coloring, runs the search on it,
        and saves the best coloring found to the archive (subject to
        ``config.save_out_of_range``), stopping as soon as the archive
        holds ``config.target_count`` unique colorings within the
        configured score band or ``config.maximum_attempts`` attempts
        have been made, whichever comes first.

        Args:
            config (RArchiveBatchConfig): Batch settings: run name,
                target count, attempt budget, score band, and starting
                iteration.
            observer (RArchiveBatchObserver | None): Optional callback
                invoked with each :class:`RArchiveBatchAttempt` as it
                completes, e.g. for progress reporting.

        Returns:
            RArchiveBatchResult: Summary of the batch, including the
            eligible-coloring counts before and after and every
            attempt's outcome.

        Raises:
            TypeError: If ``observer`` is supplied and is not callable.
        """
        if observer is not None and not callable(observer):
            raise TypeError("observer must be callable.")

        initial_count = self._eligible_count(config)
        eligible_count = initial_count
        attempt_results: list[RArchiveBatchAttempt] = []

        for attempt in range(config.maximum_attempts):
            if eligible_count >= config.target_count:
                break

            iteration = config.start_iteration + attempt
            seed_coloring = self._construction.construct(self._graph)
            construction_name = self._construction.last_source_name
            search_result = self._search.run(
                seed_coloring,
                record_steps=config.record_steps,
            )

            in_score_range = _score_is_in_range(
                search_result.best_score,
                minimum_score=config.minimum_score,
                maximum_score=config.maximum_score,
            )

            archive_record: RArchiveRecord | None = None
            if in_score_range or config.save_out_of_range:
                archive_record = self._archive.save_coloring(
                    search_result.best_coloring,
                    run_name=config.run_name,
                    iteration=iteration,
                )

            new_unique_coloring = (
                archive_record is not None
                and archive_record.times_seen == 1
            )
            if in_score_range and new_unique_coloring:
                eligible_count += 1

            attempt_result = RArchiveBatchAttempt(
                attempt=attempt,
                iteration=iteration,
                construction_name=construction_name,
                search_result=search_result,
                in_score_range=in_score_range,
                archive_record=archive_record,
                new_unique_coloring=new_unique_coloring,
                eligible_count=eligible_count,
            )
            attempt_results.append(attempt_result)

            if observer is not None:
                observer(attempt_result)

        # Re-query so the result remains exact even if another process wrote
        # to the archive while this batch was running.
        final_count = self._eligible_count(config)

        return RArchiveBatchResult(
            run_name=config.run_name,
            target_count=config.target_count,
            maximum_attempts=config.maximum_attempts,
            initial_eligible_count=initial_count,
            final_eligible_count=final_count,
            attempt_results=tuple(attempt_results),
        )

    def _eligible_count(self, config: RArchiveBatchConfig) -> int:
        """Query the archive for the current count of unique in-range colorings.

        Args:
            config (RArchiveBatchConfig): Batch settings supplying the
                score band to query.

        Returns:
            int: Number of unique colorings in the archive, scoped to
            :attr:`graph`, whose score falls within
            ``[config.minimum_score, config.maximum_score]``.
        """
        return self._archive.coloring_count_in_score_range(
            minimum_score=config.minimum_score,
            maximum_score=config.maximum_score,
            graph=self._graph,
        )


def _score_is_in_range(
    score: int,
    *,
    minimum_score: int | None,
    maximum_score: int | None,
) -> bool:
    """Check whether a score falls within an inclusive band.

    Args:
        score (int): Score to test.
        minimum_score (int | None): Inclusive lower bound, or ``None``
            for no lower bound.
        maximum_score (int | None): Inclusive upper bound, or ``None``
            for no upper bound.

    Returns:
        bool: ``True`` if ``score`` satisfies both bounds.
    """
    if minimum_score is not None and score < minimum_score:
        return False
    if maximum_score is not None and score > maximum_score:
        return False
    return True


def _nonnegative_integer(name: str, value: int) -> int:
    """Validate that a value is a nonnegative integer.

    Args:
        name (str): Parameter name, used in error messages.
        value (int): Candidate value.

    Returns:
        int: ``value`` coerced to ``int``.

    Raises:
        TypeError: If ``value`` is not an integer (booleans excluded).
        ValueError: If ``value`` is negative.
    """
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} cannot be negative.")
    return value


def _positive_integer(name: str, value: int) -> int:
    """Validate that a value is a positive integer.

    Args:
        name (str): Parameter name, used in error messages.
        value (int): Candidate value.

    Returns:
        int: ``value`` coerced to ``int``.

    Raises:
        TypeError: If ``value`` is not an integer (booleans excluded).
        ValueError: If ``value`` is not positive.
    """
    value = _nonnegative_integer(name, value)
    if value == 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _optional_nonnegative_integer(
    name: str,
    value: int | None,
) -> int | None:
    """Validate that a value is a nonnegative integer or ``None``.

    Args:
        name (str): Parameter name, used in error messages.
        value (int | None): Candidate value, or ``None``.

    Returns:
        int | None: ``None`` if ``value`` is ``None``, otherwise
        ``value`` coerced to ``int``.

    Raises:
        TypeError: If ``value`` is not ``None`` and not an integer.
        ValueError: If ``value`` is negative.
    """
    if value is None:
        return None
    return _nonnegative_integer(name, value)