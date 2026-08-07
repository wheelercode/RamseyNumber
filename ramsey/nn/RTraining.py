"""End-to-end orchestration of PPO training experiments."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from ..RArchive import (
    RArchive,
    RArchiveRecord,
)
from ..RConstruction import RConstruction
from ..RColoring import RColoring
from ..REnvironment import REnvironment
from ..RGraph import RGraph
from .RCheckpoint import (
    save_training_checkpoint,
)
from .RModel import (
    RPairPolicyValueNetwork,
)
from .RPPO import (
    RPPOConfig,
    RPPOMetrics,
    ppo_update,
)
from .RRollout import (
    RRolloutBatch,
    RRolloutConfig,
    collect_rollout,
)


@dataclass(frozen=True, slots=True)
class RTrainingConfig:
    """
    Settings for a sequence of PPO training iterations.

    Attributes:
        run_name (str): Nonempty label identifying this training run,
            used when archiving colorings and saving checkpoints.
        iterations (int): Number of training iterations to run,
            starting at ``start_iteration``.
        start_iteration (int): Index of the first iteration, used to
            number iterations (and checkpoints) when resuming a run.
            Defaults to ``0``.
        stop_on_solution (bool): When ``True``, :meth:`RPPOTrainer.run`
            stops after the first iteration whose rollout reaches a
            score-zero coloring, instead of continuing through the
            full requested ``iterations``. Defaults to ``True``.
    """

    run_name: str
    iterations: int
    start_iteration: int = 0
    stop_on_solution: bool = True

    def __post_init__(self) -> None:
        """
        Validate, coerce, and normalize every configuration field in place.

        ``run_name`` must be a nonempty string; ``iterations`` must be
        a positive integer; ``start_iteration`` must be a nonnegative
        integer; ``stop_on_solution`` must be a ``bool``. Integer
        fields are rewritten with their coerced ``int`` values via
        ``object.__setattr__`` because the dataclass is frozen.

        Raises:
            TypeError: If a field has the wrong type (including
                ``bool`` where an integer is expected).
            ValueError: If ``run_name`` is empty/whitespace, if
                ``iterations`` is less than ``1``, or if
                ``start_iteration`` is negative.
        """
        if (
            not isinstance(
                self.run_name,
                str,
            )
            or not self.run_name.strip()
        ):
            raise ValueError("run_name must be a nonempty string.")

        for name in (
            "iterations",
            "start_iteration",
        ):
            value = getattr(
                self,
                name,
            )

            if isinstance(value, bool) or not isinstance(
                value,
                Integral,
            ):
                raise TypeError(f"{name} must be an integer.")

            value = int(value)

            minimum = 1 if name == "iterations" else 0

            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}.")

            object.__setattr__(
                self,
                name,
                value,
            )

        if not isinstance(
            self.stop_on_solution,
            bool,
        ):
            raise TypeError("stop_on_solution must be boolean.")


@dataclass(frozen=True, slots=True)
class RCheckpointSchedule:
    """
    Optional periodic and final checkpoint schedule.

    Attributes:
        directory (Path): Directory checkpoint files are written
            into.
        interval (int): Save a checkpoint every ``interval``
            completed iterations. Defaults to ``100``.
        save_final (bool): When ``True``, also save a checkpoint at
            the final requested iteration, or when training stops
            early on a solution (subject to
            ``RTrainingConfig.stop_on_solution``). Defaults to
            ``True``.
    """

    directory: Path
    interval: int = 100
    save_final: bool = True

    def __post_init__(self) -> None:
        """
        Coerce ``directory`` to a ``Path`` and validate the remaining fields.

        ``interval`` must be a positive integer and ``save_final``
        must be a ``bool``. Fields are rewritten via
        ``object.__setattr__`` because the dataclass is frozen.

        Raises:
            TypeError: If ``interval`` is not an integer (including
                ``bool``), or if ``save_final`` is not a ``bool``.
            ValueError: If ``interval`` is not positive.
        """
        directory = Path(self.directory)

        object.__setattr__(
            self,
            "directory",
            directory,
        )

        if isinstance(
            self.interval,
            bool,
        ) or not isinstance(
            self.interval,
            Integral,
        ):
            raise TypeError("interval must be an integer.")

        interval = int(self.interval)

        if interval <= 0:
            raise ValueError("interval must be positive.")

        object.__setattr__(
            self,
            "interval",
            interval,
        )

        if not isinstance(
            self.save_final,
            bool,
        ):
            raise TypeError("save_final must be boolean.")

    def path_for(
        self,
        iteration: int,
    ) -> Path:
        """
        Return the checkpoint path for one iteration.

        Args:
            iteration (int): Training iteration index the checkpoint
                corresponds to.

        Returns:
            Path: ``directory`` joined with a zero-padded,
            iteration-numbered file name.
        """

        return self.directory / ("ramsey_policy_iteration_" f"{iteration:06d}.pt")


@dataclass(frozen=True, slots=True)
class RTrainingIteration:
    """
    Compact outcome and persistence data for one iteration.

    Attributes:
        iteration (int): Index of this training iteration.
        construction_name (str): Name of the construction used to
            build this iteration's seed coloring.
        construction_source (str): Identifier of the specific source
            (for example, an archived coloring or a random
            construction) the seed coloring was built from.
        initial_score (int): Exact score of the seed coloring before
            the rollout.
        final_score (int): Exact score of the coloring at the end of
            the rollout.
        best_score (int): Lowest exact score observed during the
            rollout.
        steps_completed (int): Number of environment steps collected
            in the rollout.
        total_scaled_reward (float): Sum of the rollout's scaled
            per-step rewards.
        final_coloring (RColoring): Coloring snapshot at the end of
            the rollout.
        best_coloring (RColoring): Best (lowest-score) coloring
            snapshot observed during the rollout.
        terminated (bool): Whether the rollout ended in a true
            terminal search state.
        truncated (bool): Whether the rollout ended due to a step or
            other time limit.
        metrics (RPPOMetrics | None): PPO update diagnostics, or
            ``None`` if the rollout was empty and no update was
            performed.
        archive_record (RArchiveRecord | None): Archive record for
            the rollout's best coloring, or ``None`` if no archive was
            configured.
        new_archive_best (bool): Whether this iteration's archived
            coloring improved on the archive's previous best score for
            this graph.
        checkpoint_path (Path | None): Path a checkpoint was saved to
            for this iteration, or ``None`` if no checkpoint was due.
    """

    iteration: int
    construction_name: str
    construction_source: str
    initial_score: int
    final_score: int
    best_score: int
    steps_completed: int
    total_scaled_reward: float
    final_coloring: RColoring
    best_coloring: RColoring
    terminated: bool
    truncated: bool
    metrics: RPPOMetrics | None
    archive_record: RArchiveRecord | None
    new_archive_best: bool
    checkpoint_path: Path | None

    @property
    def parameter_update_performed(
        self,
    ) -> bool:
        """bool: Whether this iteration's rollout was nonempty and a PPO update ran."""

        return self.metrics is not None


@dataclass(frozen=True, slots=True)
class RTrainingResult:
    """
    Outcome of a complete PPO training run.

    Attributes:
        run_name (str): Label identifying the training run.
        requested_iterations (int): Number of iterations that were
            requested via :class:`RTrainingConfig`, which may exceed
            :attr:`completed_iterations` if training stopped early.
        iteration_results (tuple[RTrainingIteration, ...]): Per-iteration
            outcomes, in the order iterations were run.
    """

    run_name: str
    requested_iterations: int
    iteration_results: tuple[RTrainingIteration, ...]

    @property
    def completed_iterations(
        self,
    ) -> int:
        """int: Number of iterations actually completed."""

        return len(self.iteration_results)

    @property
    def best_iteration(
        self,
    ) -> RTrainingIteration:
        """
        Return the iteration with the lowest score.

        Returns:
            RTrainingIteration: The iteration result whose
            ``best_score`` is lowest across the run.

        Raises:
            RuntimeError: If no iterations completed.
        """

        if not self.iteration_results:
            raise RuntimeError("Training contains no iteration results.")

        return min(
            self.iteration_results,
            key=lambda result: (result.best_score),
        )

    @property
    def best_score(
        self,
    ) -> int:
        """int: Lowest exact score found during training."""

        return self.best_iteration.best_score

    @property
    def solved(
        self,
    ) -> bool:
        """bool: Whether training found a score-zero (fully valid) coloring."""

        return self.best_score == 0


RTrainingObserver = Callable[
    [RTrainingIteration],
    None,
]
"""Callback type invoked with each :class:`RTrainingIteration` as it completes."""


class RPPOTrainer:
    """
    Coordinate seed construction, rollouts, PPO updates, and persistence.

    The trainer does not implement graph construction, environment
    behavior, scoring, neural encoding, PPO mathematics, archiving,
    or checkpoint serialization. It only coordinates those objects.
    """

    def __init__(
        self,
        *,
        graph: RGraph,
        construction: RConstruction,
        environment: REnvironment,
        network: RPairPolicyValueNetwork,
        optimizer: torch.optim.Optimizer,
        device: torch.device | str,
        rng: np.random.Generator,
        rollout_config: RRolloutConfig,
        ppo_config: RPPOConfig,
        archive: RArchive | None = None,
    ) -> None:
        """
        Construct a trainer coordinating one graph/network/environment set.

        Args:
            graph (RGraph): Host graph training runs against.
            construction (RConstruction): Construction strategy used
                to build a new seed coloring at the start of each
                iteration.
            environment (REnvironment): Search environment driven
                during rollouts. Must operate on the same problem as
                ``graph``.
            network (RPairPolicyValueNetwork): Policy/value network
                being trained. Must have vertex/edge dimensions
                matching ``graph``.
            optimizer (torch.optim.Optimizer): Optimizer stepping
                ``network``'s parameters during PPO updates.
            device (torch.device | str): Device ``network`` and
                rollout/update tensors reside on.
            rng (numpy.random.Generator): NumPy random generator
                available to collaborating components (for example,
                stochastic constructions).
            rollout_config (RRolloutConfig): Settings used to collect
                each iteration's rollout.
            ppo_config (RPPOConfig): Settings used for each PPO
                update.
            archive (RArchive | None): Optional archive that each
                iteration's best coloring is saved to. ``None``
                disables archiving. Defaults to ``None``.

        Raises:
            ValueError: If ``environment.graph.problem`` does not
                match ``graph.problem``, or if ``network``'s
                vertex/edge dimensions do not match ``graph``'s.
            TypeError: If ``rng`` is not a ``numpy.random.Generator``.
        """
        if environment.graph.problem != graph.problem:
            raise ValueError("Environment problem does not match " "training graph.")

        if (
            network.n_vertices != graph.problem.n_vertices
            or network.number_of_edges != graph.number_of_edges
        ):
            raise ValueError("Network dimensions do not match " "training graph.")

        if not isinstance(
            rng,
            np.random.Generator,
        ):
            raise TypeError("rng must be a NumPy Generator.")

        self._graph = graph
        self._construction = construction
        self._environment = environment
        self._network = network
        self._optimizer = optimizer
        self._device = torch.device(device)
        self._rng = rng
        self._rollout_config = rollout_config
        self._ppo_config = ppo_config
        self._archive = archive

    @property
    def graph(
        self,
    ) -> RGraph:
        """RGraph: Host graph training runs against."""
        return self._graph

    @property
    def construction(
        self,
    ) -> RConstruction:
        """RConstruction: Strategy used to build each iteration's seed coloring."""
        return self._construction

    @property
    def environment(
        self,
    ) -> REnvironment:
        """REnvironment: Search environment driven during rollouts."""
        return self._environment

    @property
    def network(
        self,
    ) -> RPairPolicyValueNetwork:
        """RPairPolicyValueNetwork: Policy/value network being trained."""
        return self._network

    @property
    def optimizer(
        self,
    ) -> torch.optim.Optimizer:
        """torch.optim.Optimizer: Optimizer stepping the network's parameters."""
        return self._optimizer

    @property
    def device(
        self,
    ) -> torch.device:
        """torch.device: Device the network and training tensors reside on."""
        return self._device

    @property
    def rng(
        self,
    ) -> np.random.Generator:
        """numpy.random.Generator: Random generator shared with collaborating components."""
        return self._rng

    @property
    def rollout_config(
        self,
    ) -> RRolloutConfig:
        """RRolloutConfig: Settings used to collect each iteration's rollout."""
        return self._rollout_config

    @property
    def ppo_config(
        self,
    ) -> RPPOConfig:
        """RPPOConfig: Settings used for each PPO update."""
        return self._ppo_config

    @property
    def archive(
        self,
    ) -> RArchive | None:
        """RArchive | None: Archive each iteration's best coloring is saved to, if any."""
        return self._archive

    def run(
        self,
        config: RTrainingConfig,
        *,
        checkpoint_schedule: RCheckpointSchedule | None = None,
        observer: RTrainingObserver | None = None,
    ) -> RTrainingResult:
        """
        Run sequential PPO iterations.

        For each iteration from ``config.start_iteration`` through
        ``config.start_iteration + config.iterations - 1``: a new seed
        coloring is built with ``self.construction``; a rollout is
        collected from that seed with :func:`~ramsey.nn.RRollout.collect_rollout`;
        if the rollout is nonempty, one PPO update
        (:func:`~ramsey.nn.RPPO.ppo_update`) is immediately performed on
        it (a terminal seed coloring yields an empty rollout with
        nothing to train from, but its result is still archived and
        reported); the rollout's best coloring is archived, if an
        archive is configured; a checkpoint is saved if one is due per
        ``checkpoint_schedule``; and an :class:`RTrainingIteration` is
        appended to the results and passed to ``observer``, if given.
        If ``config.stop_on_solution`` is ``True`` and the iteration's
        rollout reached a score-zero coloring, the loop stops after
        that iteration even if further iterations were requested.

        Args:
            config (RTrainingConfig): Settings controlling the
                iteration range and stop-on-solution behavior for this
                run.
            checkpoint_schedule (RCheckpointSchedule | None): Optional
                schedule controlling when checkpoints are written.
                ``None`` disables checkpointing. Defaults to ``None``.
            observer (RTrainingObserver | None): Optional callback
                invoked with each :class:`RTrainingIteration` as it
                completes. Defaults to ``None``.

        Returns:
            RTrainingResult: Aggregated results for every iteration
            that completed before the loop ended.

        Raises:
            TypeError: If ``observer`` is given and is not callable.
        """

        if observer is not None and not callable(observer):
            raise TypeError("observer must be callable.")

        results: list[RTrainingIteration] = []

        ending_iteration = config.start_iteration + config.iterations

        for iteration in range(
            config.start_iteration,
            ending_iteration,
        ):
            seed_coloring = self._construction.construct(self._graph)

            construction_name = self._construction.name
            construction_source = self._construction.last_source_name

            rollout = collect_rollout(
                network=self._network,
                environment=self._environment,
                coloring=seed_coloring,
                device=self._device,
                config=self._rollout_config,
            )

            metrics: RPPOMetrics | None = None

            # A terminal seed produces an empty rollout. There is
            # nothing to train from, but it can still be archived
            # and reported as a solution.
            if rollout.number_of_steps > 0:
                metrics = ppo_update(
                    network=self._network,
                    optimizer=self._optimizer,
                    rollout=rollout,
                    device=self._device,
                    config=self._ppo_config,
                )

            (
                archive_record,
                new_archive_best,
            ) = self._archive_rollout(
                rollout,
                run_name=config.run_name,
                iteration=iteration,
            )

            solved = rollout.best_score == 0

            final_requested_iteration = iteration == ending_iteration - 1

            checkpoint_path = self._checkpoint_if_due(
                checkpoint_schedule,
                config=config,
                iteration=iteration,
                final_requested_iteration=final_requested_iteration,
                solved=solved,
                archive_record=archive_record,
                construction_source=construction_source,
            )

            iteration_result = RTrainingIteration(
                iteration=iteration,
                construction_name=construction_name,
                construction_source=construction_source,
                initial_score=rollout.initial_score,
                final_score=rollout.final_score,
                best_score=rollout.best_score,
                steps_completed=rollout.number_of_steps,
                total_scaled_reward=rollout.total_scaled_reward,
                final_coloring=rollout.final_coloring,
                best_coloring=rollout.best_coloring,
                terminated=rollout.terminated,
                truncated=rollout.truncated,
                metrics=metrics,
                archive_record=archive_record,
                new_archive_best=new_archive_best,
                checkpoint_path=checkpoint_path,
            )

            results.append(iteration_result)

            if observer is not None:
                observer(iteration_result)

            if config.stop_on_solution and solved:
                break

        return RTrainingResult(
            run_name=config.run_name,
            requested_iterations=config.iterations,
            iteration_results=tuple(results),
        )

    def _archive_rollout(
        self,
        rollout: RRolloutBatch,
        *,
        run_name: str,
        iteration: int,
    ) -> tuple[
        RArchiveRecord | None,
        bool,
    ]:
        """
        Archive the best coloring found during one rollout.

        Args:
            rollout (RRolloutBatch): Rollout whose ``best_coloring``
                is saved.
            run_name (str): Run label recorded with the archive
                entry.
            iteration (int): Iteration index recorded with the
                archive entry.

        Returns:
            tuple[RArchiveRecord | None, bool]: The saved archive
            record (``None`` if no archive is configured) and whether
            it improved on the archive's previous best score for
            ``self.graph``.
        """

        if self._archive is None:
            return None, False

        previous_best = self._archive.best_score(self._graph)

        record = self._archive.save_coloring(
            rollout.best_coloring,
            run_name=run_name,
            iteration=iteration,
        )

        new_archive_best = previous_best is None or record.score < previous_best

        return (
            record,
            new_archive_best,
        )

    def _checkpoint_if_due(
        self,
        schedule: RCheckpointSchedule | None,
        *,
        config: RTrainingConfig,
        iteration: int,
        final_requested_iteration: bool,
        solved: bool,
        archive_record: RArchiveRecord | None,
        construction_source: str,
    ) -> Path | None:
        """
        Save and return a checkpoint path when scheduled.

        A checkpoint is saved when the number of iterations completed
        since ``config.start_iteration`` is a multiple of
        ``schedule.interval`` (periodic), or when
        ``schedule.save_final`` is set and either this is the final
        requested iteration or ``config.stop_on_solution`` is set and
        ``solved`` is ``True`` (final). Checkpoint metadata records
        the run name, construction name/source, and, when available,
        the archived coloring's id and score.

        Args:
            schedule (RCheckpointSchedule | None): Checkpoint schedule
                to consult; ``None`` disables checkpointing for this
                call.
            config (RTrainingConfig): Training configuration supplying
                ``run_name``, ``start_iteration``, and
                ``stop_on_solution``.
            iteration (int): Index of the current iteration.
            final_requested_iteration (bool): Whether ``iteration`` is
                the last one requested by ``config``.
            solved (bool): Whether this iteration's rollout reached a
                score-zero coloring.
            archive_record (RArchiveRecord | None): This iteration's
                archive record, if any, used to enrich checkpoint
                metadata.
            construction_source (str): Identifier of the seed
                coloring's construction source, recorded in checkpoint
                metadata.

        Returns:
            Path | None: The path the checkpoint was written to, or
            ``None`` if no checkpoint was due.
        """

        if schedule is None:
            return None

        iterations_completed = iteration - config.start_iteration + 1

        periodic = iterations_completed % schedule.interval == 0

        final = schedule.save_final and (
            final_requested_iteration or (config.stop_on_solution and solved)
        )

        if not periodic and not final:
            return None

        metadata: dict[
            str,
            object,
        ] = {
            "run_name": (config.run_name),
            "construction_name": (self._construction.name),
            "construction_source": construction_source,
        }

        if archive_record is not None:
            metadata["coloring_id"] = archive_record.coloring_id

            metadata["best_score"] = archive_record.score

        return save_training_checkpoint(
            schedule.path_for(iteration),
            network=self._network,
            optimizer=self._optimizer,
            graph=self._graph,
            rollout_config=self._rollout_config,
            ppo_config=self._ppo_config,
            completed_iteration=iteration,
            rng=self._rng,
            metadata=metadata,
        )