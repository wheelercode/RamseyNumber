"""Process-parallel execution of independent exact-greedy searches.

Runs many independent exact-score greedy searches (:class:`RGreedyPolicy`
with ``use_objective_reward=False`` over an :class:`RMonochromaticObjective`,
each with its own :class:`RTabuMemory`) concurrently across worker
processes via :class:`RExactGreedyProcessPool`. Each worker builds the
host graph and clique index once and reuses it for every task it
receives; the pool only parallelizes running independent searches to
completion and collecting their results, preserving input task order.
There is no cross-worker synchronization or shared state during a
search: every task's coloring, tabu memory, and RNG are independent, and
results are only aggregated back into a tuple after all tasks in a
batch complete.
"""

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
    """Immutable configuration shared by exact-greedy workers.

    Sent once to each worker process at pool startup (via
    ``ProcessPoolExecutor`` initializer arguments) and used to build
    that worker's graph and to construct the environment and tabu
    memory for every task it subsequently runs.

    Attributes:
        problem (RProblem): Ramsey coloring problem defining the host
            graph each worker builds.
        environment (REnvironmentConfig): Environment settings (step
            limit, aspiration) used for every worker task.
        memory (RTabuMemoryConfig): Tabu memory settings used for every
            worker task.
    """

    problem: RProblem
    environment: REnvironmentConfig
    memory: RTabuMemoryConfig


@dataclass(
    frozen=True,
    slots=True,
    eq=False,
)
class RParallelSearchTask:
    """One coloring and deterministic policy seed sent to a worker.

    Attributes:
        task_id (int): Nonnegative identifier used to correlate this
            task with its :class:`RParallelSearchResult`.
        colors (numpy.ndarray): Uint8 array, shape ``(number_of_edges,)``,
            owned and read-only. Seed edge coloring for the task's
            search attempt; must match the worker graph's edge count.
        action_seed (int): Nonnegative seed for the worker's random
            number generator, giving each task's greedy tie-breaking a
            reproducible, independent stream.
    """

    task_id: int
    colors: NDArray[np.uint8]
    action_seed: int

    def __post_init__(self) -> None:
        """Validate and normalize the task's fields.

        Raises:
            TypeError: If ``task_id`` or ``action_seed`` is not an
                integer.
            ValueError: If ``task_id`` or ``action_seed`` is negative,
                or ``colors`` is not one-dimensional.
        """
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
    """Compact worker result returned to the parent process.

    Attributes:
        task_id (int): Identifier matching the originating
            :class:`RParallelSearchTask`.
        initial_score (int): Exact score of the task's seed coloring.
        final_score (int): Exact score when the worker's search ended.
        best_score (int): Best (lowest) exact score found during the
            search.
        best_colors (numpy.ndarray): Uint8 array, shape
            ``(number_of_edges,)``, owned and read-only. Edge coloring
            achieving ``best_score``.
        elapsed_seconds (float): Wall-clock time spent running the
            search in the worker process.
    """

    task_id: int
    initial_score: int
    final_score: int
    best_score: int
    best_colors: NDArray[np.uint8]
    elapsed_seconds: float

    def __post_init__(self) -> None:
        """Copy ``best_colors`` and mark it read-only."""
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


#: Per-process configuration set once by ``_initialize_exact_greedy_worker``.
_WORKER_CONFIG: RExactGreedyProcessConfig | None = None
#: Per-process host graph built once by ``_initialize_exact_greedy_worker``.
_WORKER_GRAPH: RGraph | None = None


def _initialize_exact_greedy_worker(
    config: RExactGreedyProcessConfig,
) -> None:
    """Build immutable graph indexing once inside a worker process.

    Runs as the ``ProcessPoolExecutor`` initializer, so it executes
    exactly once per worker process before any task function runs. It
    stores ``config`` and the freshly built host graph in module-level
    globals so subsequent calls to ``_run_exact_greedy_task`` in the
    same process can reuse the (expensive to build) graph and clique
    index without rebuilding them per task.

    Args:
        config (RExactGreedyProcessConfig): Shared configuration for
            every task this worker process will run.
    """
    global _WORKER_CONFIG
    global _WORKER_GRAPH

    _WORKER_CONFIG = config
    _WORKER_GRAPH = RGraph(
        config.problem
    )


def _run_exact_greedy_task(
    task: RParallelSearchTask,
) -> RParallelSearchResult:
    """Execute one independent exact-greedy search in a worker.

    Builds a fresh :class:`REnvironment`, :class:`RTabuMemory`, and
    :class:`RGreedyPolicy` (seeded from ``task.action_seed``) for this
    task alone, runs an :class:`RSearch` to completion from
    ``task.colors``, and reports a compact summary. Each call is
    independent: nothing here is shared with other tasks or workers
    beyond the read-only graph and config built once in
    ``_initialize_exact_greedy_worker``.

    Args:
        task (RParallelSearchTask): Seed coloring, task id, and action
            RNG seed for this search attempt.

    Returns:
        RParallelSearchResult: Compact outcome of the search, along
        with the elapsed wall-clock time.

    Raises:
        RuntimeError: If this worker process was not initialized via
            ``_initialize_exact_greedy_worker`` (i.e. the pool was
            used incorrectly).
        ValueError: If ``task.colors`` does not have one entry per
            edge of the worker's host graph.
    """
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
        """Configure a process pool without starting any workers yet.

        Args:
            config (RExactGreedyProcessConfig): Shared configuration
                passed to every worker process on startup.
            max_workers (int): Number of worker processes to run
                concurrently; must be positive.

        Raises:
            TypeError: If ``max_workers`` is not an integer.
            ValueError: If ``max_workers`` is not positive.
        """
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
        """
        Return the number of worker processes the pool runs.
        """
        return self._max_workers

    @property
    def running(self) -> bool:
        """
        Return whether the pool currently has worker processes started.
        """
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
        """Execute one batch of tasks and preserve input order.

        Starts the pool if it is not already running, then distributes
        ``tasks`` across worker processes (one task per unit of work,
        via ``chunksize=1``) and blocks until every task in the batch
        completes. A running pool may be reused for multiple
        successive calls to ``run`` before it is closed.

        Args:
            tasks (Iterable[RParallelSearchTask]): Independent search
                tasks to execute. An empty iterable returns immediately.

        Returns:
            tuple[RParallelSearchResult, ...]: One result per task, in
            the same order as ``tasks``.

        Raises:
            RuntimeError: If the process pool failed to start.
        """
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
        """Start the pool and return it for use in a ``with`` block.

        Returns:
            RExactGreedyProcessPool: This pool, now running.
        """
        self.start()
        return self

    def __exit__(
        self,
        exception_type,
        exception_value,
        traceback,
    ) -> None:
        """Shut down the pool when the ``with`` block exits.

        Waits for any in-flight tasks to finish (``close`` uses
        ``wait=True``) regardless of whether the block exited normally
        or via an exception; exceptions are not suppressed.
        """
        self.close()