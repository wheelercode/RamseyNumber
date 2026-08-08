"""Repeated population search in explicit parent-to-child generations."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Callable

import numpy as np

from .RArchive import RArchive, RArchiveRecord
from .RColoring import RColoring
from .RConstruction import RConstruction
from .RGraph import RGraph
from .RSearch import RSearch, RSearchResult


def _positive_integer(name: str, value: int) -> int:
    """Return a validated positive built-in integer."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")

    result = int(value)

    if result <= 0:
        raise ValueError(f"{name} must be positive.")

    return result


@dataclass(frozen=True, slots=True)
class RGenerationalExperimentConfig:
    """Configure a fixed-size sequence of search generations.

    Args:
        run_name (str): Stable experiment name used for archive provenance.
        population_size (int): Number of independent ancestry lines.
        generations (int): Number of complete search passes to perform.
        record_steps (bool): Whether each search retains every step result.
        stop_on_solution (bool): Whether score zero ends the whole experiment.
    """

    run_name: str
    population_size: int
    generations: int
    record_steps: bool = False
    stop_on_solution: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.run_name, str) or not self.run_name.strip():
            raise ValueError("run_name must be a nonempty string.")

        object.__setattr__(
            self,
            "population_size",
            _positive_integer("population_size", self.population_size),
        )
        object.__setattr__(
            self,
            "generations",
            _positive_integer("generations", self.generations),
        )

        if not isinstance(self.record_steps, bool):
            raise TypeError("record_steps must be boolean.")

        if not isinstance(self.stop_on_solution, bool):
            raise TypeError("stop_on_solution must be boolean.")


@dataclass(frozen=True, slots=True)
class RGenerationalExperimentMember:
    """One parent-to-child search within a generation.

    Attributes:
        generation (int): One-based generation number.
        member (int): Zero-based ancestry-line index.
        parent_archive_record (RArchiveRecord | None): Archive record for the
            preceding generation. It is ``None`` for the random/constructed
            initial population.
        search_result (RSearchResult): Search performed on the parent.
        archive_record (RArchiveRecord): Archived child selected from the
            best coloring observed during the search.
        new_archive_best (bool): Whether this child established a new
            persistent archive score record at the time it was saved.
    """

    generation: int
    member: int
    parent_archive_record: RArchiveRecord | None
    search_result: RSearchResult
    archive_record: RArchiveRecord
    new_archive_best: bool

    @property
    def child_coloring(self) -> RColoring:
        """Return the coloring passed to this lineage's next generation."""
        return self.search_result.best_coloring


@dataclass(frozen=True, slots=True)
class RGenerationalExperimentGeneration:
    """Results for one complete population generation."""

    generation: int
    members: tuple[RGenerationalExperimentMember, ...]

    @property
    def best_score(self) -> int:
        """Return the lowest score observed in this generation."""
        return min(member.search_result.best_score for member in self.members)

    @property
    def mean_initial_score(self) -> float:
        """Return the population mean score entering the generation."""
        return float(
            np.mean(
                [member.search_result.initial_score for member in self.members]
            )
        )

    @property
    def mean_child_score(self) -> float:
        """Return the population mean score handed to the next generation."""
        return float(
            np.mean([member.search_result.best_score for member in self.members])
        )

    @property
    def mean_steps(self) -> float:
        """Return the mean search length within this generation."""
        return float(
            np.mean([member.search_result.steps_completed for member in self.members])
        )


@dataclass(frozen=True, slots=True)
class RGenerationalExperimentResult:
    """Complete ancestry-preserving result of a generational experiment."""

    run_name: str
    initial_population: tuple[RColoring, ...]
    generations: tuple[RGenerationalExperimentGeneration, ...]

    @property
    def generations_completed(self) -> int:
        """Return the number of generations that ran."""
        return len(self.generations)

    @property
    def best_score(self) -> int:
        """Return the lowest score observed in any generation."""
        if not self.generations:
            raise RuntimeError("No generations were completed.")

        return min(generation.best_score for generation in self.generations)

    @property
    def final_population(self) -> tuple[RColoring, ...]:
        """Return the final generation's child colorings."""
        if not self.generations:
            return self.initial_population

        return tuple(
            member.child_coloring for member in self.generations[-1].members
        )


RGenerationalExperimentObserver = Callable[
    [RGenerationalExperimentMember],
    None,
]


class RGenerationalExperiment:
    """Search a population repeatedly while preserving exact ancestry.

    The initial construction is invoked exactly ``population_size`` times.
    Generation one searches those constructed colorings. Every later
    generation searches exactly the best children produced by the preceding
    generation; it never re-queries the global archive to choose parents.

    Generation is therefore experiment provenance rather than mutable graph
    state. Archive run names include the generation number so descendants
    remain identifiable in SQLite without coupling ``RSearchState`` to the
    experiment that happened to produce a coloring.
    """

    def __init__(
        self,
        *,
        graph: RGraph,
        initial_construction: RConstruction,
        search: RSearch,
        archive: RArchive,
    ) -> None:
        if search.environment.graph.problem != graph.problem:
            raise ValueError(
                "Search environment problem does not match experiment graph."
            )

        if not isinstance(archive, RArchive):
            raise TypeError("archive must implement RArchive.")

        self._graph = graph
        self._initial_construction = initial_construction
        self._search = search
        self._archive = archive

    @property
    def graph(self) -> RGraph:
        """Return the graph shared by every population member."""
        return self._graph

    @property
    def initial_construction(self) -> RConstruction:
        """Return the construction used only for generation-zero parents."""
        return self._initial_construction

    @property
    def search(self) -> RSearch:
        """Return the search applied once to every parent each generation."""
        return self._search

    @property
    def archive(self) -> RArchive:
        """Return the persistent destination for generation children."""
        return self._archive

    def run(
        self,
        config: RGenerationalExperimentConfig,
        *,
        observer: RGenerationalExperimentObserver | None = None,
    ) -> RGenerationalExperimentResult:
        """Construct the initial population and run all generations.

        Args:
            config (RGenerationalExperimentConfig): Population and run limits.
            observer (RGenerationalExperimentObserver | None): Optional callback
                invoked after each child is searched and archived.

        Returns:
            RGenerationalExperimentResult: Immutable generation history and
            final population.
        """
        if observer is not None and not callable(observer):
            raise TypeError("observer must be callable.")

        initial_population = tuple(
            self._initial_construction.construct(self._graph)
            for _ in range(config.population_size)
        )

        parent_population = initial_population
        parent_records: tuple[RArchiveRecord | None, ...] = (
            (None,) * config.population_size
        )
        generation_results: list[RGenerationalExperimentGeneration] = []

        for generation in range(1, config.generations + 1):
            members: list[RGenerationalExperimentMember] = []
            generation_run_name = (
                f"{config.run_name}-generation-{generation:04d}"
            )

            for member, (parent, parent_record) in enumerate(
                zip(parent_population, parent_records, strict=True)
            ):
                previous_best = self._archive.best_score(self._graph)
                search_result = self._search.run(
                    parent,
                    record_steps=config.record_steps,
                )
                archive_record = self._archive.save_coloring(
                    search_result.best_coloring,
                    run_name=generation_run_name,
                    iteration=member,
                )
                new_archive_best = (
                    previous_best is None
                    or archive_record.score < previous_best
                )

                result = RGenerationalExperimentMember(
                    generation=generation,
                    member=member,
                    parent_archive_record=parent_record,
                    search_result=search_result,
                    archive_record=archive_record,
                    new_archive_best=new_archive_best,
                )
                members.append(result)

                if observer is not None:
                    observer(result)

                if config.stop_on_solution and search_result.best_score == 0:
                    break

            generation_result = RGenerationalExperimentGeneration(
                generation=generation,
                members=tuple(members),
            )
            generation_results.append(generation_result)

            if config.stop_on_solution and generation_result.best_score == 0:
                break

            parent_population = tuple(
                member.child_coloring for member in members
            )
            parent_records = tuple(
                member.archive_record for member in members
            )

        return RGenerationalExperimentResult(
            run_name=config.run_name,
            initial_population=initial_population,
            generations=tuple(generation_results),
        )
