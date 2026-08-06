"""Process-parallel archive population for independent Ramsey searches."""

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
    """Settings for one process-parallel archive population run."""

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
    """One completed worker search and its parent-side archive outcome."""

    attempt: int
    iteration: int
    construction_name: str
    search_result: RParallelSearchResult
    in_score_range: bool
    archive_record: RArchiveRecord | None
    new_unique_coloring: bool


@dataclass(frozen=True, slots=True)
class RArchiveBatchParallelPass:
    """Summary of one parallel barrier of independent searches."""

    pass_number: int
    attempts: tuple[RArchiveBatchParallelAttempt, ...]
    eligible_count: int
    elapsed_seconds: float

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def initial_scores(self) -> np.ndarray:
        return np.asarray(
            [
                attempt.search_result.initial_score
                for attempt in self.attempts
            ],
            dtype=np.int64,
        )

    @property
    def final_scores(self) -> np.ndarray:
        return np.asarray(
            [
                attempt.search_result.final_score
                for attempt in self.attempts
            ],
            dtype=np.int64,
        )

    @property
    def best_scores(self) -> np.ndarray:
        return np.asarray(
            [
                attempt.search_result.best_score
                for attempt in self.attempts
            ],
            dtype=np.int64,
        )

    @property
    def improved_count(self) -> int:
        return sum(
            attempt.search_result.best_score
            < attempt.search_result.initial_score
            for attempt in self.attempts
        )

    @property
    def in_score_range_count(self) -> int:
        return sum(
            attempt.in_score_range
            for attempt in self.attempts
        )

    @property
    def new_unique_count(self) -> int:
        return sum(
            attempt.in_score_range
            and attempt.new_unique_coloring
            for attempt in self.attempts
        )

    @property
    def attempts_per_second(self) -> float:
        if self.elapsed_seconds <= 0.0:
            return float("inf")
        return self.attempt_count / self.elapsed_seconds


@dataclass(frozen=True, slots=True)
class RArchiveBatchParallelResult:
    """Outcome of a bounded process-parallel population run."""

    run_name: str
    target_count: int
    maximum_attempts: int
    initial_eligible_count: int
    final_eligible_count: int
    passes: tuple[RArchiveBatchParallelPass, ...]
    elapsed_seconds: float

    @property
    def attempts_completed(self) -> int:
        return sum(result.attempt_count for result in self.passes)

    @property
    def target_reached(self) -> bool:
        return self.final_eligible_count >= self.target_count

    @property
    def new_eligible_colorings(self) -> int:
        return self.final_eligible_count - self.initial_eligible_count

    @property
    def best_score(self) -> int | None:
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
        if not isinstance(archive, RArchive):
            raise TypeError("archive must implement RArchive.")

        self._graph = graph
        self._construction = construction
        self._archive = archive
        self._search_pool = search_pool

    @property
    def graph(self) -> RGraph:
        return self._graph

    @property
    def construction(self) -> RConstruction:
        return self._construction

    @property
    def archive(self) -> RArchive:
        return self._archive

    @property
    def search_pool(self) -> RExactGreedyProcessPool:
        return self._search_pool

    def populate(
        self,
        config: RArchiveBatchParallelConfig,
        *,
        observer: RArchiveBatchParallelObserver | None = None,
    ) -> RArchiveBatchParallelResult:
        """Populate until the score band is full or attempts run out."""
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
    if minimum_score is not None and score < minimum_score:
        return False
    if maximum_score is not None and score > maximum_score:
        return False
    return True


def _nonnegative_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")

    value = int(value)

    if value < 0:
        raise ValueError(f"{name} cannot be negative.")

    return value


def _positive_integer(name: str, value: int) -> int:
    value = _nonnegative_integer(name, value)

    if value == 0:
        raise ValueError(f"{name} must be positive.")

    return value


def _optional_nonnegative_integer(
    name: str,
    value: int | None,
) -> int | None:
    if value is None:
        return None
    return _nonnegative_integer(name, value)