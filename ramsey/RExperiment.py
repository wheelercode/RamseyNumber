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

    Attributes:
        run_name (str): Nonempty label identifying this experiment run,
            recorded with any archived colorings.
        iterations (int): Number of search attempts to run, starting
            at ``start_iteration``.
        start_iteration (int): Iteration number of the first attempt.
            Useful for resuming a run's numbering. Defaults to 0.
        record_steps (bool): Whether each attempt's :class:`RSearch`
            run keeps every intermediate ``RStepResult``. Defaults to
            False.
        stop_on_solution (bool): Whether to stop issuing further
            iterations as soon as one attempt reaches an exact score
            of zero. Defaults to True.
    """

    run_name: str
    iterations: int

    start_iteration: int = 0
    record_steps: bool = False
    stop_on_solution: bool = True

    def __post_init__(self) -> None:
        """Validate the configured field values.

        Raises:
            ValueError: If ``run_name`` is empty or whitespace-only,
                ``iterations`` is not positive, or ``start_iteration``
                is negative.
            TypeError: If ``iterations`` or ``start_iteration`` is not
                an integer, or if ``record_steps`` or
                ``stop_on_solution`` is not a ``bool``.
        """
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

    Attributes:
        iteration (int): Iteration number of this attempt.
        construction_name (str): Name of the construction that
            produced the seed coloring.
        search_result (RSearchResult): Complete outcome of the search
            attempt.
        archive_record (RArchiveRecord | None): Record describing
            where/how the best coloring was archived, or ``None`` if
            the experiment has no archive.
        new_archive_best (bool): True if ``archive_record`` improved
            on the archive's previous best score for this graph.
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

    Attributes:
        run_name (str): Label identifying the experiment run.
        requested_iterations (int): Number of iterations that were
            requested via :class:`RExperimentConfig`; may exceed
            ``completed_iterations`` if the run stopped early.
        iteration_results (tuple[RExperimentIteration, ...]): Outcome
            of each attempt actually completed, in order.
    """

    run_name: str
    requested_iterations: int

    iteration_results: tuple[RExperimentIteration, ...]

    @property
    def completed_iterations(self) -> int:
        """
        Return the number of attempts actually completed.

        Returns:
            int: Length of ``iteration_results``.
        """
        return len(self.iteration_results)

    @property
    def best_iteration(
        self,
    ) -> RExperimentIteration:
        """
        Return the iteration having the lowest best score.

        Returns:
            RExperimentIteration: The iteration result whose
            ``search_result.best_score`` is lowest.

        Raises:
            RuntimeError: If no iterations completed.
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

        Returns:
            int: ``best_iteration.search_result.best_score``.

        Raises:
            RuntimeError: If no iterations completed.
        """
        return self.best_iteration.search_result.best_score

    @property
    def solved(self) -> bool:
        """
        Return whether any iteration reached score zero.

        Returns:
            bool: True if ``best_score`` is zero.

        Raises:
            RuntimeError: If no iterations completed.
        """
        return self.best_score == 0


#: Callback invoked with each :class:`RExperimentIteration` as it completes.
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
        """Bind an experiment to a graph, construction, search, and archive.

        Args:
            graph (RGraph): Host graph attempts are run against; must
                describe the same problem as ``search.environment.graph``.
            construction (RConstruction): Seed-coloring generator used
                once per iteration.
            search (RSearch): Search coordinator run against each seed
                coloring.
            archive (RArchive | None): Optional store for best
                colorings found by each iteration. When omitted, no
                archiving occurs.

        Raises:
            ValueError: If ``search.environment.graph.problem`` does
                not match ``graph.problem``.
        """
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
        """
        Return the host graph attempts are run against.
        """
        return self._graph

    @property
    def construction(
        self,
    ) -> RConstruction:
        """
        Return the seed-coloring construction used each iteration.
        """
        return self._construction

    @property
    def search(self) -> RSearch:
        """
        Return the search coordinator run against each seed coloring.
        """
        return self._search

    @property
    def archive(
        self,
    ) -> RArchive | None:
        """
        Return the archive best colorings are saved to, if any.
        """
        return self._archive

    def run(
        self,
        config: RExperimentConfig,
        *,
        observer: RExperimentObserver | None = None,
    ) -> RExperimentResult:
        """
        Run the configured sequence of search attempts.

        For each iteration this constructs a seed coloring, runs it
        through the search coordinator to termination or truncation,
        optionally saves the best coloring to the archive, and reports
        the outcome to ``observer``. The observer, when supplied, is
        called after each completed iteration. It can print progress,
        update a notebook, write additional logs, or perform other
        external reporting. If ``config.stop_on_solution`` is true and
        an iteration reaches an exact score of zero, no further
        iterations are attempted.

        Args:
            config (RExperimentConfig): Settings governing how many
                iterations to run and how to run them.
            observer (RExperimentObserver | None): Optional callback
                invoked with each completed iteration's result.

        Returns:
            RExperimentResult: Immutable summary of every completed
            iteration.

        Raises:
            TypeError: If ``observer`` is supplied and is not callable.
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
