"""Process-parallel archive population for independent Ramsey searches.

Provides :class:`RArchiveBatchParallel`, a driver analogous to
:class:`ramsey.RArchiveBatch.RArchiveBatch` that instead dispatches
batches of independent searches to a worker process pool
(:class:`ramsey.RSearchParallel.RExactGreedyProcessPool`). Seed
construction and all archive access stay in the parent process; workers
receive only edge-color arrays and deterministic policy seeds, run
complete searches, and return compact results that the parent then
saves and tallies at each synchronization barrier ("pass").
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from time import perf_counter
from typing import Callable

import numpy as np

from .RArchive import (
    RArchive,
    RArchiveRecord,
)
from .RColoring import RColoring
from .RConstruction import RConstruction
from .RGraph import RGraph
from .RSearchParallel import (
    RExactGreedyProcessPool,
    RParallelSearchResult,
    RParallelSearchTask,
)


@dataclass(frozen=True, slots=True)
class RArchiveBatchParallelConfig:
    """Settings for one process-parallel archive population run.

    Attributes:
        run_name (str): Nonempty name recorded as provenance for every
            coloring saved during this run.
        target_count (int): Number of unique, in-range colorings the run
            should try to accumulate in the archive before stopping
            early.
        maximum_attempts (int): Upper bound on the total number of
            independent searches to dispatch across all passes.
        batch_size (int): Number of independent search tasks dispatched
            to the worker pool at each pass (subject to the remaining
            attempt and target budgets).
        action_seed_base (int): Base value added to each attempt's index
            to derive that attempt's deterministic policy seed, so
            repeated runs with the same base reproduce the same
            per-attempt action sequences.
        minimum_score (int | None): Inclusive lower bound on score for a
            coloring to count toward ``target_count``, or ``None`` for
            no lower bound.
        maximum_score (int | None): Inclusive upper bound on score for a
            coloring to count toward ``target_count``, or ``None`` for
            no upper bound.
        start_iteration (int): Iteration number assigned to the first
            attempt; each subsequent attempt increments it by one.
        save_out_of_range (bool): Whether colorings whose score falls
            outside ``[minimum_score, maximum_score]`` should still be
            saved to the archive (without counting toward
            ``target_count``).
    """

    run_name: str
    target_count: int
    maximum_attempts: int
    batch_size: int
    action_seed_base: int
    minimum_score: int | None = None
    maximum_score: int | None = None
    start_iteration: int = 0
    save_out_of_range: bool = False

    def __post_init__(self) -> None:
        """Validate and normalize configuration values.

        Raises:
            ValueError: If ``run_name`` is empty, ``target_count``,
                ``maximum_attempts``, or ``batch_size`` is not positive,
                ``action_seed_base`` or ``start_iteration`` is negative,
                either score bound is negative, or ``minimum_score``
                exceeds ``maximum_score``.
            TypeError: If ``target_count``, ``maximum_attempts``,
                ``batch_size``, ``action_seed_base``,
                ``start_iteration``, ``minimum_score``, or
                ``maximum_score`` is not an integer (or ``None`` where
                allowed), or if ``save_out_of_range`` is not boolean.
        """
        if not isinstance(self.run_name, str) or not self.run_name.strip():
            raise ValueError("run_name must be a nonempty string.")

        for name in (
            "target_count",
            "maximum_attempts",
            "batch_size",
        ):
            object.__setattr__(
                self,
                name,
                _positive_integer(name, getattr(self, name)),
            )

        for name in (
            "action_seed_base",
            "start_iteration",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_integer(name, getattr(self, name)),
            )

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

        if not isinstance(self.save_out_of_range, bool):
            raise TypeError("save_out_of_range must be boolean.")


@dataclass(frozen=True, slots=True)
class RArchiveBatchParallelAttempt:
    """One completed worker search and its parent-side archive outcome.

    Attributes:
        attempt (int): Zero-based index of this attempt across the
            entire run (not just within its pass).
        iteration (int): Iteration number assigned to this attempt
            (``config.start_iteration + attempt``), recorded as
            provenance if the coloring is saved.
        construction_name (str): Name of the construction source that
            produced the seed coloring for this attempt.
        search_result (RParallelSearchResult): Compact result returned
            by the worker process for this attempt's search.
        in_score_range (bool): Whether the search's best score falls
            within the run's configured score band.
        archive_record (RArchiveRecord | None): Archive record for the
            saved coloring, or ``None`` if the coloring was not saved
            (out of range and ``save_out_of_range`` was false).
        new_unique_coloring (bool): Whether this attempt's coloring was
            not already present in the archive (``times_seen == 1``
            after saving).
    """

    attempt: int
    iteration: int
    construction_name: str
    search_result: RParallelSearchResult
    in_score_range: bool
    archive_record: RArchiveRecord | None
    new_unique_coloring: bool


@dataclass(frozen=True, slots=True)
class RArchiveBatchParallelPass:
    """Summary of one parallel barrier of independent searches.

    One "pass" is one round-trip to the worker pool: a batch of
    independent search tasks is dispatched, the pool blocks until all of
    them complete, and the parent process then saves and tallies the
    results before deciding whether another pass is needed.

    Attributes:
        pass_number (int): Zero-based index of this pass within the run.
        attempts (tuple[RArchiveBatchParallelAttempt, ...]): Outcome of
            every attempt dispatched in this pass.
        eligible_count (int): Number of unique in-range colorings in the
            archive, re-queried immediately after this pass completed.
        elapsed_seconds (float): Wall-clock time spent waiting on the
            worker pool for this pass (excludes seed construction and
            archive I/O).
    """

    pass_number: int
    attempts: tuple[RArchiveBatchParallelAttempt, ...]
    eligible_count: int
    elapsed_seconds: float

    @property
    def attempt_count(self) -> int:
        """int: Number of search attempts dispatched in this pass."""
        return len(self.attempts)

    @property
    def initial_scores(self) -> np.ndarray:
        """numpy.ndarray: ``int64`` array of each attempt's pre-search score, one entry per attempt."""
        return np.asarray(
            [
                attempt.search_result.initial_score
                for attempt in self.attempts
            ],
            dtype=np.int64,
        )

    @property
    def final_scores(self) -> np.ndarray:
        """numpy.ndarray: ``int64`` array of each attempt's score at search termination, one entry per attempt."""
        return np.asarray(
            [
                attempt.search_result.final_score
                for attempt in self.attempts
            ],
            dtype=np.int64,
        )

    @property
    def best_scores(self) -> np.ndarray:
        """numpy.ndarray: ``int64`` array of each attempt's best score seen during search, one entry per attempt."""
        return np.asarray(
            [
                attempt.search_result.best_score
                for attempt in self.attempts
            ],
            dtype=np.int64,
        )

    @property
    def improved_count(self) -> int:
        """int: Number of attempts whose best score improved on (was lower than) their initial score."""
        return sum(
            attempt.search_result.best_score
            < attempt.search_result.initial_score
            for attempt in self.attempts
        )

    @property
    def in_score_range_count(self) -> int:
        """int: Number of attempts whose best score fell within the configured score band."""
        return sum(
            attempt.in_score_range
            for attempt in self.attempts
        )

    @property
    def new_unique_count(self) -> int:
        """int: Number of attempts that were both in range and newly archived (not previously seen)."""
        return sum(
            attempt.in_score_range
            and attempt.new_unique_coloring
            for attempt in self.attempts
        )

    @property
    def attempts_per_second(self) -> float:
        """float: Throughput of this pass, or ``inf`` if it completed in effectively zero time."""
        if self.elapsed_seconds <= 0.0:
            return float("inf")
        return self.attempt_count / self.elapsed_seconds


@dataclass(frozen=True, slots=True)
class RArchiveBatchParallelResult:
    """Outcome of a bounded process-parallel population run.

    Attributes:
        run_name (str): Run name the population run was executed under.
        target_count (int): Target number of unique in-range colorings
            the run was trying to reach.
        maximum_attempts (int): Attempt budget the run was allowed to
            use.
        initial_eligible_count (int): Number of unique in-range
            colorings already in the archive before this run started.
        final_eligible_count (int): Number of unique in-range colorings
            in the archive after this run finished, re-queried from the
            archive so it stays exact even if another process wrote to
            it concurrently.
        passes (tuple[RArchiveBatchParallelPass, ...]): Summary of every
            parallel pass performed, in order.
        elapsed_seconds (float): Total wall-clock time spent across all
            passes.
    """

    run_name: str
    target_count: int
    maximum_attempts: int
    initial_eligible_count: int
    final_eligible_count: int
    passes: tuple[RArchiveBatchParallelPass, ...]
    elapsed_seconds: float

    @property
    def attempts_completed(self) -> int:
        """int: Total number of search attempts dispatched across all passes."""
        return sum(result.attempt_count for result in self.passes)

    @property
    def target_reached(self) -> bool:
        """bool: Whether the archive holds at least ``target_count`` eligible colorings."""
        return self.final_eligible_count >= self.target_count

    @property
    def new_eligible_colorings(self) -> int:
        """int: Net increase in unique in-range colorings caused by this run."""
        return self.final_eligible_count - self.initial_eligible_count

    @property
    def best_score(self) -> int | None:
        """int | None: Lowest best-score seen across every attempt in every pass, or ``None`` if no attempts ran."""
        scores = [
            attempt.search_result.best_score
            for parallel_pass in self.passes
            for attempt in parallel_pass.attempts
        ]
        return min(scores) if scores else None


RArchiveBatchParallelObserver = Callable[
    [RArchiveBatchParallelPass],
    None,
]
"""Callback invoked with each :class:`RArchiveBatchParallelPass` as it completes."""


class RArchiveBatchParallel:
    """
    Populate an archive in process-parallel search passes.

    Seed construction and all SQLite access remain in the parent
    process.  Workers receive only edge-color arrays and deterministic
    policy seeds, perform complete searches, and return compact results.
    """

    def __init__(
        self,
        *,
        graph: RGraph,
        construction: RConstruction,
        archive: RArchive,
        search_pool: RExactGreedyProcessPool,
    ) -> None:
        """Bind a construction, archive, and worker pool to one host graph.

        Args:
            graph (RGraph): Host graph that seed colorings and archive
                queries are scoped to.
            construction (RConstruction): Construction used to produce a
                seed coloring for each attempt.
            archive (RArchive): Archive that discovered colorings are
                saved to and queried against, from the parent process
                only.
            search_pool (RExactGreedyProcessPool): Worker process pool
                that runs the greedy searches dispatched at each pass.

        Raises:
            TypeError: If ``archive`` does not implement
                :class:`ramsey.RArchive.RArchive`.
        """
        if not isinstance(archive, RArchive):
            raise TypeError("archive must implement RArchive.")

        self._graph = graph
        self._construction = construction
        self._archive = archive
        self._search_pool = search_pool

    @property
    def graph(self) -> RGraph:
        """RGraph: Host graph seed colorings and archive queries are scoped to."""
        return self._graph

    @property
    def construction(self) -> RConstruction:
        """RConstruction: Construction used to produce seed colorings."""
        return self._construction

    @property
    def archive(self) -> RArchive:
        """RArchive: Archive that discovered colorings are saved to."""
        return self._archive

    @property
    def search_pool(self) -> RExactGreedyProcessPool:
        """RExactGreedyProcessPool: Worker process pool searches are dispatched to."""
        return self._search_pool

    def populate(
        self,
        config: RArchiveBatchParallelConfig,
        *,
        observer: RArchiveBatchParallelObserver | None = None,
    ) -> RArchiveBatchParallelResult:
        """Populate until the score band is full or attempts run out.

        Runs a sequence of passes. Each pass constructs up to
        ``config.batch_size`` seed colorings (capped by the remaining
        attempt budget and by how many more eligible colorings are
        still needed), dispatches them to the worker pool as one batch,
        waits for every task in the batch to complete, then saves
        qualifying results to the archive and re-queries the eligible
        count. Passes continue until the target count is reached or the
        attempt budget is exhausted.

        Args:
            config (RArchiveBatchParallelConfig): Run settings: run
                name, target count, attempt budget, pass size, action
                seed base, score band, and starting iteration.
            observer (RArchiveBatchParallelObserver | None): Optional
                callback invoked with each :class:`RArchiveBatchParallelPass`
                as it completes, e.g. for progress reporting.

        Returns:
            RArchiveBatchParallelResult: Summary of the run, including
            the eligible-coloring counts before and after, every pass's
            outcome, and total elapsed time.

        Raises:
            TypeError: If ``observer`` is supplied and is not callable.
        """
        if observer is not None and not callable(observer):
            raise TypeError("observer must be callable.")

        initial_count = self._eligible_count(config)
        eligible_count = initial_count
        attempts_completed = 0
        pass_number = 0
        pass_results: list[RArchiveBatchParallelPass] = []

        overall_start = perf_counter()

        while (
            eligible_count < config.target_count
            and attempts_completed < config.maximum_attempts
        ):
            remaining_attempts = (
                config.maximum_attempts - attempts_completed
            )
            remaining_target = config.target_count - eligible_count

            # Do not launch more work than can possibly be needed to
            # reach the target.  Duplicate/nonqualifying results may
            # require another pass.
            current_batch_size = min(
                config.batch_size,
                remaining_attempts,
                remaining_target,
            )

            tasks: list[RParallelSearchTask] = []
            construction_names: list[str] = []

            for offset in range(current_batch_size):
                attempt = attempts_completed + offset
                seed = self._construction.construct(self._graph)

                construction_names.append(
                    self._construction.last_source_name
                )
                tasks.append(
                    RParallelSearchTask(
                        task_id=attempt,
                        colors=seed.colors,
                        action_seed=(
                            config.action_seed_base + attempt
                        ),
                    )
                )

            pass_start = perf_counter()
            worker_results = self._search_pool.run(tasks)

            attempts: list[RArchiveBatchParallelAttempt] = []

            for offset, worker_result in enumerate(worker_results):
                attempt = attempts_completed + offset
                iteration = config.start_iteration + attempt

                in_score_range = _score_is_in_range(
                    worker_result.best_score,
                    minimum_score=config.minimum_score,
                    maximum_score=config.maximum_score,
                )

                archive_record: RArchiveRecord | None = None

                if in_score_range or config.save_out_of_range:
                    best_coloring = RColoring(
                        self._graph,
                        worker_result.best_colors,
                    )
                    archive_record = self._archive.save_coloring(
                        best_coloring,
                        run_name=config.run_name,
                        iteration=iteration,
                    )

                new_unique = (
                    archive_record is not None
                    and archive_record.times_seen == 1
                )

                if in_score_range and new_unique:
                    eligible_count += 1

                attempts.append(
                    RArchiveBatchParallelAttempt(
                        attempt=attempt,
                        iteration=iteration,
                        construction_name=construction_names[offset],
                        search_result=worker_result,
                        in_score_range=in_score_range,
                        archive_record=archive_record,
                        new_unique_coloring=new_unique,
                    )
                )

            pass_elapsed = perf_counter() - pass_start

            # Re-query at each barrier.  This is authoritative and also
            # accounts for any independent writer using the same archive.
            eligible_count = self._eligible_count(config)

            parallel_pass = RArchiveBatchParallelPass(
                pass_number=pass_number,
                attempts=tuple(attempts),
                eligible_count=eligible_count,
                elapsed_seconds=pass_elapsed,
            )
            pass_results.append(parallel_pass)

            attempts_completed += current_batch_size
            pass_number += 1

            if observer is not None:
                observer(parallel_pass)

        overall_elapsed = perf_counter() - overall_start
        final_count = self._eligible_count(config)

        return RArchiveBatchParallelResult(
            run_name=config.run_name,
            target_count=config.target_count,
            maximum_attempts=config.maximum_attempts,
            initial_eligible_count=initial_count,
            final_eligible_count=final_count,
            passes=tuple(pass_results),
            elapsed_seconds=overall_elapsed,
        )

    def _eligible_count(
        self,
        config: RArchiveBatchParallelConfig,
    ) -> int:
        """Query the archive for the current count of unique in-range colorings.

        Args:
            config (RArchiveBatchParallelConfig): Run settings supplying
                the score band to query.

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