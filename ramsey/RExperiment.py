"""Reproducible assembly and execution of search experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .RArchive import (
    RArchive,
    RArchiveRecord,
)
from .RConstruction import RConstruction
from .RGraph import RGraph
from .RSearch import (
    RSearch,
    RSearchResult,
)


@dataclass(frozen=True, slots=True)
class RExperimentConfig:
    """
    Immutable settings for a sequence of search attempts.
    """

    run_name: str
    iterations: int

    start_iteration: int = 0
    record_steps: bool = False
    stop_on_solution: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.run_name, str) or not self.run_name.strip():
            raise ValueError("run_name must be a nonempty string.")

        if isinstance(self.iterations, bool) or not isinstance(
            self.iterations,
            int,
        ):
            raise TypeError("iterations must be an integer.")

        if self.iterations <= 0:
            raise ValueError("iterations must be positive.")

        if isinstance(self.start_iteration, bool) or not isinstance(
            self.start_iteration,
            int,
        ):
            raise TypeError("start_iteration must be an integer.")

        if self.start_iteration < 0:
            raise ValueError("start_iteration cannot be negative.")

        if not isinstance(
            self.record_steps,
            bool,
        ):
            raise TypeError("record_steps must be boolean.")

        if not isinstance(
            self.stop_on_solution,
            bool,
        ):
            raise TypeError("stop_on_solution must be boolean.")


@dataclass(frozen=True, slots=True)
class RExperimentIteration:
    """
    Outcome and archive metadata for one search attempt.
    """

    iteration: int
    construction_name: str

    search_result: RSearchResult
    archive_record: RArchiveRecord | None

    new_archive_best: bool


@dataclass(frozen=True, slots=True)
class RExperimentResult:
    """
    Immutable outcome of a complete experiment run.
    """

    run_name: str
    requested_iterations: int

    iteration_results: tuple[RExperimentIteration, ...]

    @property
    def completed_iterations(self) -> int:
        """
        Return the number of attempts actually completed.
        """
        return len(self.iteration_results)

    @property
    def best_iteration(
        self,
    ) -> RExperimentIteration:
        """
        Return the iteration having the lowest best score.
        """
        if not self.iteration_results:
            raise RuntimeError("Experiment contains no iteration results.")

        return min(
            self.iteration_results,
            key=lambda result: (result.search_result.best_score),
        )

    @property
    def best_score(self) -> int:
        """
        Return the lowest exact score found.
        """
        return self.best_iteration.search_result.best_score

    @property
    def solved(self) -> bool:
        """
        Return whether any iteration reached score zero.
        """
        return self.best_score == 0


RExperimentObserver = Callable[
    [RExperimentIteration],
    None,
]


class RExperiment:
    """
    Coordinate construction, search, archival, and reporting.

    RExperiment is deliberately thin. It does not implement any
    construction, policy, environment behavior, scoring, plotting,
    or database storage itself.
    """

    def __init__(
        self,
        graph: RGraph,
        construction: RConstruction,
        search: RSearch,
        archive: RArchive | None = None,
    ) -> None:
        if search.environment.graph.problem != graph.problem:
            raise ValueError(
                "Search environment problem does not " "match experiment graph."
            )

        self._graph = graph
        self._construction = construction
        self._search = search
        self._archive = archive

    @property
    def graph(self) -> RGraph:
        return self._graph

    @property
    def construction(
        self,
    ) -> RConstruction:
        return self._construction

    @property
    def search(self) -> RSearch:
        return self._search

    @property
    def archive(
        self,
    ) -> RArchive | None:
        return self._archive

    def run(
        self,
        config: RExperimentConfig,
        *,
        observer: RExperimentObserver | None = None,
    ) -> RExperimentResult:
        """
        Run the configured sequence of search attempts.

        The observer, when supplied, is called after each completed
        iteration. It can print progress, update a notebook, write
        additional logs, or perform other external reporting.
        """
        if observer is not None and not callable(observer):
            raise TypeError("observer must be callable.")

        iteration_results: list[RExperimentIteration] = []

        ending_iteration = config.start_iteration + config.iterations

        for iteration in range(
            config.start_iteration,
            ending_iteration,
        ):
            seed_coloring = self._construction.construct(self._graph)

            search_result = self._search.run(
                seed_coloring,
                record_steps=config.record_steps,
            )

            archive_record: RArchiveRecord | None = None

            new_archive_best = False

            if self._archive is not None:
                previous_best = self._archive.best_score(self._graph)

                archive_record = self._archive.save_coloring(
                    search_result.best_coloring,
                    run_name=config.run_name,
                    iteration=iteration,
                )

                new_archive_best = (
                    previous_best is None or archive_record.score < previous_best
                )

            iteration_result = RExperimentIteration(
                iteration=iteration,
                construction_name=self._construction.name,
                search_result=search_result,
                archive_record=archive_record,
                new_archive_best=new_archive_best,
            )

            iteration_results.append(iteration_result)

            if observer is not None:
                observer(iteration_result)

            if config.stop_on_solution and search_result.best_score == 0:
                break

        return RExperimentResult(
            run_name=config.run_name,
            requested_iterations=config.iterations,
            iteration_results=tuple(iteration_results),
        )
