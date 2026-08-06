"""Process-parallel execution of independent exact-greedy searches."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from numbers import Integral
from time import perf_counter
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from .RColoring import RColoring
from .REnvironment import REnvironment
from .REnvironmentConfig import (
    REnvironmentConfig,
    RTabuMemoryConfig,
)
from .REnvironmentMemory import RTabuMemory
from .RGraph import RGraph
from .RObjective import RMonochromaticObjective
from .RPolicy import RGreedyPolicy
from .RProblem import RProblem
from .RSearch import RSearch


@dataclass(
    frozen=True,
    slots=True,
)
class RExactGreedyProcessConfig:
    """Immutable configuration shared by exact-greedy workers."""

    problem: RProblem
    environment: REnvironmentConfig
    memory: RTabuMemoryConfig


@dataclass(
    frozen=True,
    slots=True,
    eq=False,
)
class RParallelSearchTask:
    """One coloring and deterministic policy seed sent to a worker."""

    task_id: int
    colors: NDArray[np.uint8]
    action_seed: int

    def __post_init__(self) -> None:
        if isinstance(self.task_id, bool) or not isinstance(
            self.task_id,
            Integral,
        ):
            raise TypeError("task_id must be an integer.")

        if isinstance(self.action_seed, bool) or not isinstance(
            self.action_seed,
            Integral,
        ):
            raise TypeError("action_seed must be an integer.")

        task_id = int(self.task_id)
        action_seed = int(self.action_seed)

        if task_id < 0:
            raise ValueError("task_id cannot be negative.")

        if action_seed < 0:
            raise ValueError("action_seed cannot be negative.")

        colors = np.asarray(
            self.colors,
            dtype=np.uint8,
        ).copy()

        if colors.ndim != 1:
            raise ValueError("colors must be one-dimensional.")

        colors.flags.writeable = False

        object.__setattr__(
            self,
            "task_id",
            task_id,
        )
        object.__setattr__(
            self,
            "action_seed",
            action_seed,
        )
        object.__setattr__(
            self,
            "colors",
            colors,
        )


@dataclass(
    frozen=True,
    slots=True,
    eq=False,
)
class RParallelSearchResult:
    """Compact worker result returned to the parent process."""

    task_id: int
    initial_score: int
    final_score: int
    best_score: int
    best_colors: NDArray[np.uint8]
    elapsed_seconds: float

    def __post_init__(self) -> None:
        best_colors = np.asarray(
            self.best_colors,
            dtype=np.uint8,
        ).copy()

        best_colors.flags.writeable = False

        object.__setattr__(
            self,
            "best_colors",
            best_colors,
        )


_WORKER_CONFIG: RExactGreedyProcessConfig | None = None
_WORKER_GRAPH: RGraph | None = None


def _initialize_exact_greedy_worker(
    config: RExactGreedyProcessConfig,
) -> None:
    """Build immutable graph indexing once inside a worker process."""
    global _WORKER_CONFIG
    global _WORKER_GRAPH

    _WORKER_CONFIG = config
    _WORKER_GRAPH = RGraph(
        config.problem
    )


def _run_exact_greedy_task(
    task: RParallelSearchTask,
) -> RParallelSearchResult:
    """Execute one independent exact-greedy search in a worker."""
    if _WORKER_CONFIG is None or _WORKER_GRAPH is None:
        raise RuntimeError(
            "Parallel search worker was not initialized."
        )

    graph = _WORKER_GRAPH
    config = _WORKER_CONFIG

    if task.colors.shape != (
        graph.number_of_edges,
    ):
        raise ValueError(
            "Task coloring edge count does not match "
            "the worker graph."
        )

    environment = REnvironment(
        graph=graph,
        objective=RMonochromaticObjective(),
        memory=RTabuMemory(
            graph.number_of_edges,
            config.memory,
        ),
        config=config.environment,
    )

    policy = RGreedyPolicy(
        rng=np.random.default_rng(
            task.action_seed
        ),
        use_objective_reward=False,
    )

    search = RSearch(
        environment=environment,
        policy=policy,
    )

    coloring = RColoring(
        graph,
        task.colors,
    )

    start = perf_counter()

    result = search.run(
        coloring,
        record_steps=False,
    )

    elapsed = perf_counter() - start

    return RParallelSearchResult(
        task_id=task.task_id,
        initial_score=result.initial_score,
        final_score=result.final_score,
        best_score=result.best_score,
        best_colors=result.best_coloring.colors,
        elapsed_seconds=elapsed,
    )


class RExactGreedyProcessPool:
    """
    Reusable process pool for independent exact-greedy searches.

    The expensive graph/K5 indexing is constructed once per worker.
    A pool may then execute multiple generations of tasks before it is
    closed.  The pool never reads from or writes to an archive.
    """

    def __init__(
        self,
        config: RExactGreedyProcessConfig,
        *,
        max_workers: int,
    ) -> None:
        if isinstance(max_workers, bool) or not isinstance(
            max_workers,
            Integral,
        ):
            raise TypeError("max_workers must be an integer.")

        max_workers = int(max_workers)

        if max_workers <= 0:
            raise ValueError("max_workers must be positive.")

        self._config = config
        self._max_workers = max_workers
        self._executor: ProcessPoolExecutor | None = None

    @property
    def max_workers(self) -> int:
        return self._max_workers

    @property
    def running(self) -> bool:
        return self._executor is not None

    def start(self) -> None:
        """Start worker processes if the pool is not already running."""
        if self._executor is not None:
            return

        self._executor = ProcessPoolExecutor(
            max_workers=self._max_workers,
            initializer=_initialize_exact_greedy_worker,
            initargs=(self._config,),
        )

    def run(
        self,
        tasks: Iterable[RParallelSearchTask],
    ) -> tuple[RParallelSearchResult, ...]:
        """Execute one batch of tasks and preserve input order."""
        self.start()

        if self._executor is None:
            raise RuntimeError("Process pool failed to start.")

        task_tuple = tuple(tasks)

        if not task_tuple:
            return ()

        return tuple(
            self._executor.map(
                _run_exact_greedy_task,
                task_tuple,
                chunksize=1,
            )
        )

    def close(self) -> None:
        """Shut down all worker processes."""
        if self._executor is None:
            return

        self._executor.shutdown(
            wait=True,
            cancel_futures=False,
        )

        self._executor = None

    def __enter__(
        self,
    ) -> "RExactGreedyProcessPool":
        self.start()
        return self

    def __exit__(
        self,
        exception_type,
        exception_value,
        traceback,
    ) -> None:
        self.close()