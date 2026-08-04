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
    """Settings for a sequence of PPO training iterations."""

    run_name: str
    iterations: int
    start_iteration: int = 0
    stop_on_solution: bool = True

    def __post_init__(self) -> None:
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
    """Optional periodic and final checkpoint schedule."""

    directory: Path
    interval: int = 100
    save_final: bool = True

    def __post_init__(self) -> None:
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
        """Return the checkpoint path for one iteration."""

        return self.directory / ("ramsey_policy_iteration_" f"{iteration:06d}.pt")


@dataclass(frozen=True, slots=True)
class RTrainingIteration:
    """Compact outcome and persistence data for one iteration."""

    iteration: int
    construction_name: str
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
        """Return whether this iteration trained the network."""

        return self.metrics is not None


@dataclass(frozen=True, slots=True)
class RTrainingResult:
    """Outcome of a complete PPO training run."""

    run_name: str
    requested_iterations: int
    iteration_results: tuple[RTrainingIteration, ...]

    @property
    def completed_iterations(
        self,
    ) -> int:
        """Return the number of completed training iterations."""

        return len(self.iteration_results)

    @property
    def best_iteration(
        self,
    ) -> RTrainingIteration:
        """Return the iteration with the lowest score."""

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
        """Return the lowest score found during training."""

        return self.best_iteration.best_score

    @property
    def solved(
        self,
    ) -> bool:
        """Return whether training found a score-zero coloring."""

        return self.best_score == 0


RTrainingObserver = Callable[
    [RTrainingIteration],
    None,
]


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
        return self._graph

    @property
    def construction(
        self,
    ) -> RConstruction:
        return self._construction

    @property
    def environment(
        self,
    ) -> REnvironment:
        return self._environment

    @property
    def network(
        self,
    ) -> RPairPolicyValueNetwork:
        return self._network

    @property
    def optimizer(
        self,
    ) -> torch.optim.Optimizer:
        return self._optimizer

    @property
    def device(
        self,
    ) -> torch.device:
        return self._device

    @property
    def rng(
        self,
    ) -> np.random.Generator:
        return self._rng

    @property
    def rollout_config(
        self,
    ) -> RRolloutConfig:
        return self._rollout_config

    @property
    def ppo_config(
        self,
    ) -> RPPOConfig:
        return self._ppo_config

    @property
    def archive(
        self,
    ) -> RArchive | None:
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

        A new seed coloring is constructed for every iteration.
        Each nonempty rollout is immediately used for one PPO update.
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
            )

            iteration_result = RTrainingIteration(
                iteration=iteration,
                construction_name=self._construction.name,
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
        """Archive the best coloring found during one rollout."""

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
    ) -> Path | None:
        """Save and return a checkpoint path when scheduled."""

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
