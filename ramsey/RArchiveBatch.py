"""Batch production of searched colorings for a persistent archive."""

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
    """Settings for filling one archive score band."""

    run_name: str
    target_count: int
    maximum_attempts: int
    minimum_score: int | None = None
    maximum_score: int | None = None
    start_iteration: int = 0
    record_steps: bool = False
    save_out_of_range: bool = True

    def __post_init__(self) -> None:
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
    """Outcome of one construction, search, and optional save."""

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
    """Outcome of one bounded archive-population batch."""

    run_name: str
    target_count: int
    maximum_attempts: int
    initial_eligible_count: int
    final_eligible_count: int
    attempt_results: tuple[RArchiveBatchAttempt, ...]

    @property
    def attempts_completed(self) -> int:
        return len(self.attempt_results)

    @property
    def target_reached(self) -> bool:
        return self.final_eligible_count >= self.target_count

    @property
    def new_eligible_colorings(self) -> int:
        return self.final_eligible_count - self.initial_eligible_count

    @property
    def best_score(self) -> int | None:
        if not self.attempt_results:
            return None
        return min(
            result.search_result.best_score
            for result in self.attempt_results
        )


RArchiveBatchObserver = Callable[[RArchiveBatchAttempt], None]


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
        return self._graph

    @property
    def construction(self) -> RConstruction:
        return self._construction

    @property
    def search(self) -> RSearch:
        return self._search

    @property
    def archive(self) -> RArchive:
        return self._archive

    def populate(
        self,
        config: RArchiveBatchConfig,
        *,
        observer: RArchiveBatchObserver | None = None,
    ) -> RArchiveBatchResult:
        """Search until the score band is full or attempts are exhausted."""
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