"""Reproducible, no-training evaluation of neural checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from statistics import mean, median
from typing import Iterable

import numpy as np
import torch

from ..RColoring import RColoring
from ..RConstruction import RConstruction
from ..REnvironment import REnvironment
from ..RGraph import RGraph
from ..RSearch import RSearch
from .RCheckpoint import load_training_checkpoint
from .RNeuralPolicy import RNeuralPolicy
from .RRuntime import resolve_torch_device, seed_torch


@dataclass(frozen=True, slots=True)
class REvaluationSeed:
    """One named coloring shared by every evaluated checkpoint."""

    name: str
    source_name: str
    coloring: RColoring

    def __post_init__(self) -> None:
        for field_name in (
            "name",
            "source_name",
        ):
            value = getattr(
                self,
                field_name,
            )

            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{field_name} must be a nonempty string."
                )

        if not isinstance(self.coloring, RColoring):
            raise TypeError("coloring must be an RColoring.")


@dataclass(frozen=True, slots=True)
class RCheckpointEvaluationConfig:
    """Settings shared by every checkpoint comparison."""

    repetitions_per_seed: int = 1
    action_seed: int = 1_000_000
    greedy: bool = False
    score_thresholds: tuple[int, ...] = (
        600,
        550,
        500,
    )

    def __post_init__(self) -> None:
        repetitions = _nonnegative_integer(
            "repetitions_per_seed",
            self.repetitions_per_seed,
        )

        if repetitions == 0:
            raise ValueError(
                "repetitions_per_seed must be positive."
            )

        object.__setattr__(
            self,
            "repetitions_per_seed",
            repetitions,
        )

        object.__setattr__(
            self,
            "action_seed",
            _nonnegative_integer(
                "action_seed",
                self.action_seed,
            ),
        )

        if not isinstance(self.greedy, bool):
            raise TypeError("greedy must be boolean.")

        thresholds = tuple(
            _nonnegative_integer(
                "score threshold",
                threshold,
            )
            for threshold in self.score_thresholds
        )

        if len(set(thresholds)) != len(thresholds):
            raise ValueError("score_thresholds must be unique.")

        object.__setattr__(
            self,
            "score_thresholds",
            tuple(
                sorted(
                    thresholds,
                    reverse=True,
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class RCheckpointEvaluationRun:
    """Outcome of one checkpoint, seed, and repetition."""

    seed_name: str
    source_name: str
    repetition: int
    action_seed: int

    initial_score: int
    final_score: int
    best_score: int
    steps_completed: int
    terminated: bool
    truncated: bool

    @property
    def score_reduction(self) -> int:
        """Return improvement from initial to final score."""
        return self.initial_score - self.final_score

    @property
    def best_score_reduction(self) -> int:
        """Return improvement from initial to best score."""
        return self.initial_score - self.best_score


@dataclass(frozen=True, slots=True)
class RCheckpointEvaluation:
    """Aggregate evaluation of one checkpoint."""

    checkpoint_path: Path
    completed_iteration: int
    metadata: dict
    greedy: bool
    score_thresholds: tuple[int, ...]
    runs: tuple[RCheckpointEvaluationRun, ...]

    @property
    def number_of_runs(self) -> int:
        return len(self.runs)

    @property
    def mean_final_score(self) -> float:
        return float(
            mean(run.final_score for run in self.runs)
        )

    @property
    def median_final_score(self) -> float:
        return float(
            median(run.final_score for run in self.runs)
        )

    @property
    def mean_best_score(self) -> float:
        return float(
            mean(run.best_score for run in self.runs)
        )

    @property
    def median_best_score(self) -> float:
        return float(
            median(run.best_score for run in self.runs)
        )

    @property
    def minimum_best_score(self) -> int:
        return min(run.best_score for run in self.runs)

    @property
    def mean_score_reduction(self) -> float:
        return float(
            mean(run.score_reduction for run in self.runs)
        )

    @property
    def mean_best_score_reduction(self) -> float:
        return float(
            mean(
                run.best_score_reduction
                for run in self.runs
            )
        )

    @property
    def improved_final_count(self) -> int:
        return sum(
            run.score_reduction > 0
            for run in self.runs
        )

    @property
    def improved_best_count(self) -> int:
        return sum(
            run.best_score_reduction > 0
            for run in self.runs
        )

    @property
    def threshold_counts(self) -> dict[int, int]:
        return {
            threshold: sum(
                run.best_score <= threshold
                for run in self.runs
            )
            for threshold in self.score_thresholds
        }

    @property
    def source_names(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    run.source_name
                    for run in self.runs
                }
            )
        )

    def runs_for_source(
        self,
        source_name: str,
    ) -> tuple[RCheckpointEvaluationRun, ...]:
        """Return only runs originating from one seed source."""
        return tuple(
            run
            for run in self.runs
            if run.source_name == source_name
        )


@dataclass(frozen=True, slots=True)
class RCheckpointEvaluationResult:
    """Comparable evaluations for a collection of checkpoints."""

    seeds: tuple[REvaluationSeed, ...]
    evaluations: tuple[RCheckpointEvaluation, ...]

    @property
    def strongest_checkpoint(
        self,
    ) -> RCheckpointEvaluation:
        """Rank checkpoints by typical performance, then minimum."""
        if not self.evaluations:
            raise RuntimeError("No checkpoints were evaluated.")

        return min(
            self.evaluations,
            key=lambda evaluation: (
                evaluation.mean_best_score,
                evaluation.median_best_score,
                evaluation.minimum_best_score,
            ),
        )


class RCheckpointEvaluator:
    """Evaluate trained checkpoints without optimization or writes."""

    def __init__(
        self,
        graph: RGraph,
        environment: REnvironment,
        device: torch.device | str = "auto",
    ) -> None:
        if environment.graph.problem != graph.problem:
            raise ValueError(
                "Environment problem does not match graph."
            )

        self._graph = graph
        self._environment = environment
        self._device = resolve_torch_device(device)

    def evaluate(
        self,
        checkpoint_paths: Iterable[str | Path],
        seeds: Iterable[REvaluationSeed],
        config: RCheckpointEvaluationConfig | None = None,
    ) -> RCheckpointEvaluationResult:
        """Run every checkpoint under identical evaluation cases."""
        config = (
            config
            if config is not None
            else RCheckpointEvaluationConfig()
        )

        paths = tuple(
            Path(path)
            for path in checkpoint_paths
        )

        evaluation_seeds = tuple(seeds)

        if not paths:
            raise ValueError("checkpoint_paths cannot be empty.")

        if len(set(paths)) != len(paths):
            raise ValueError("checkpoint_paths must be unique.")

        if not evaluation_seeds:
            raise ValueError("seeds cannot be empty.")

        for seed in evaluation_seeds:
            if not isinstance(seed, REvaluationSeed):
                raise TypeError(
                    "Every seed must be an REvaluationSeed."
                )

            if seed.coloring.graph.problem != self._graph.problem:
                raise ValueError(
                    "Evaluation seed problem does not match graph."
                )

        cpu_rng_state = torch.get_rng_state()
        cuda_rng_states = (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        )

        try:
            evaluations = tuple(
                self._evaluate_checkpoint(
                    path,
                    evaluation_seeds,
                    config,
                )
                for path in paths
            )
        finally:
            torch.set_rng_state(cpu_rng_state)

            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(
                    cuda_rng_states
                )

        return RCheckpointEvaluationResult(
            seeds=evaluation_seeds,
            evaluations=evaluations,
        )

    def _evaluate_checkpoint(
        self,
        checkpoint_path: Path,
        seeds: tuple[REvaluationSeed, ...],
        config: RCheckpointEvaluationConfig,
    ) -> RCheckpointEvaluation:
        restored = load_training_checkpoint(
            checkpoint_path,
            graph=self._graph,
            device=self._device,
            rng=np.random.default_rng(0),
        )

        policy = RNeuralPolicy(
            restored.network,
            self._device,
            greedy=config.greedy,
        )

        search = RSearch(
            self._environment,
            policy,
        )

        runs: list[RCheckpointEvaluationRun] = []

        for seed_index, evaluation_seed in enumerate(seeds):
            for repetition in range(
                config.repetitions_per_seed
            ):
                action_seed = (
                    config.action_seed
                    + seed_index
                    * config.repetitions_per_seed
                    + repetition
                )

                seed_torch(action_seed)

                search_result = search.run(
                    evaluation_seed.coloring
                )

                runs.append(
                    RCheckpointEvaluationRun(
                        seed_name=evaluation_seed.name,
                        source_name=(
                            evaluation_seed.source_name
                        ),
                        repetition=repetition,
                        action_seed=action_seed,
                        initial_score=(
                            search_result.initial_score
                        ),
                        final_score=(
                            search_result.final_score
                        ),
                        best_score=(
                            search_result.best_score
                        ),
                        steps_completed=(
                            search_result.steps_completed
                        ),
                        terminated=search_result.terminated,
                        truncated=search_result.truncated,
                    )
                )

        return RCheckpointEvaluation(
            checkpoint_path=checkpoint_path,
            completed_iteration=(
                restored.completed_iteration
            ),
            metadata=dict(restored.metadata),
            greedy=config.greedy,
            score_thresholds=config.score_thresholds,
            runs=tuple(runs),
        )


def build_evaluation_seeds(
    graph: RGraph,
    construction: RConstruction,
    number_of_seeds: int,
    *,
    name_prefix: str | None = None,
) -> tuple[REvaluationSeed, ...]:
    """Construct and freeze a reusable checkpoint seed collection."""
    number_of_seeds = _nonnegative_integer(
        "number_of_seeds",
        number_of_seeds,
    )

    if number_of_seeds == 0:
        raise ValueError("number_of_seeds must be positive.")

    if name_prefix is not None and (
        not isinstance(name_prefix, str)
        or not name_prefix.strip()
    ):
        raise ValueError(
            "name_prefix must be a nonempty string or None."
        )

    seeds: list[REvaluationSeed] = []

    for index in range(number_of_seeds):
        coloring = construction.construct(graph)

        source_name = construction.last_source_name

        prefix = (
            name_prefix
            if name_prefix is not None
            else source_name
        )

        seeds.append(
            REvaluationSeed(
                name=f"{prefix}-{index:04d}",
                source_name=source_name,
                coloring=coloring,
            )
        )

    return tuple(seeds)


def _nonnegative_integer(
    name: str,
    value: int,
) -> int:
    """Validate and normalize one nonnegative integer."""
    if isinstance(value, bool) or not isinstance(
        value,
        Integral,
    ):
        raise TypeError(f"{name} must be an integer.")

    value = int(value)

    if value < 0:
        raise ValueError(f"{name} cannot be negative.")

    return value