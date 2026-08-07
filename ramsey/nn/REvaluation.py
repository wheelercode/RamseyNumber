"""Reproducible, no-training evaluation of neural checkpoints.

Loads one or more saved training checkpoints, runs each restored
policy (greedy or sampling) through :class:`~ramsey.RSearch.RSearch`
over a shared, fixed collection of starting colorings, and reports
comparable score statistics across checkpoints without performing any
optimization or writing any state back out. Global PyTorch RNG state
is saved and restored around evaluation so evaluating checkpoints does
not disturb random-number sequences used elsewhere (e.g. by an
in-progress training run).
"""

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
    """One named coloring shared by every evaluated checkpoint.

    Attributes:
        name (str): Unique, human-readable identifier for this seed
            (e.g. ``"<source_name>-0000"``), used to label individual
            evaluation runs.
        source_name (str): Name of the construction that produced
            ``coloring``, used to group runs by originating
            construction.
        coloring (RColoring): The starting coloring every evaluated
            checkpoint's search is initialized from.
    """

    name: str
    source_name: str
    coloring: RColoring

    def __post_init__(self) -> None:
        """Validate that the name fields are nonempty strings and the coloring is an RColoring.

        Raises:
            ValueError: If ``name`` or ``source_name`` is not a
                nonempty (after stripping) string.
            TypeError: If ``coloring`` is not an :class:`RColoring`.
        """
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
    """Settings shared by every checkpoint comparison.

    Attributes:
        repetitions_per_seed (int): Number of independent search runs
            performed per evaluation seed, each with a distinct action
            seed. Must be positive.
        action_seed (int): Base seed used to derive a distinct,
            reproducible per-run action seed (via
            :func:`~ramsey.nn.RRuntime.seed_torch`) for every
            combination of seed and repetition.
        greedy (bool): Whether each restored checkpoint's policy
            selects actions greedily rather than by sampling.
        score_thresholds (tuple[int, ...]): Exact-score thresholds used
            to report, for each checkpoint, how many runs reached a
            best score at or below each threshold. Normalized to a
            unique, descending tuple in ``__post_init__``.
    """

    repetitions_per_seed: int = 1
    action_seed: int = 1_000_000
    greedy: bool = False
    score_thresholds: tuple[int, ...] = (
        600,
        550,
        500,
    )

    def __post_init__(self) -> None:
        """Validate and normalize all fields in place.

        Raises:
            TypeError: If ``repetitions_per_seed`` or ``action_seed``
                is not an integer, if ``greedy`` is not a boolean, or
                if any entry of ``score_thresholds`` is not an integer.
            ValueError: If ``repetitions_per_seed`` is not positive, if
                ``action_seed`` or any threshold is negative, or if
                ``score_thresholds`` contains duplicate values.
        """
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
    """Outcome of one checkpoint, seed, and repetition.

    Mirrors the fields of :class:`~ramsey.RSearch.RSearchResult`,
    together with identifying information about which seed, repetition,
    and action seed produced this run.

    Attributes:
        seed_name (str): Name of the :class:`REvaluationSeed` this run
            started from.
        source_name (str): Name of the construction that produced the
            starting coloring.
        repetition (int): Index of this run among the repetitions
            performed for its seed.
        action_seed (int): Seed used to make this run's action
            selection reproducible.
        initial_score (int): Exact monochromatic score of the starting
            coloring, before any actions were taken.
        final_score (int): Exact monochromatic score of the coloring
            when the search run ended.
        best_score (int): Best (lowest) exact monochromatic score
            observed at any point during the run.
        steps_completed (int): Number of actions applied during the
            run.
        terminated (bool): Whether the search ended because the
            environment reached a terminal state.
        truncated (bool): Whether the search ended because it was cut
            off (e.g. a step limit), rather than terminating naturally.
    """

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
        """int: Improvement from the initial score to the final score."""
        return self.initial_score - self.final_score

    @property
    def best_score_reduction(self) -> int:
        """int: Improvement from the initial score to the best score observed."""
        return self.initial_score - self.best_score


@dataclass(frozen=True, slots=True)
class RCheckpointEvaluation:
    """Aggregate evaluation of one checkpoint.

    Bundles every :class:`RCheckpointEvaluationRun` produced for one
    checkpoint across all evaluation seeds and repetitions, together
    with summary statistics computed over those runs.

    Attributes:
        checkpoint_path (Path): Path the checkpoint was loaded from.
        completed_iteration (int): Training iteration count recorded in
            the checkpoint.
        metadata (dict): Free-form metadata recorded in the checkpoint.
        greedy (bool): Whether this checkpoint's policy selected
            actions greedily rather than by sampling.
        score_thresholds (tuple[int, ...]): Exact-score thresholds used
            by :attr:`threshold_counts`.
        runs (tuple[RCheckpointEvaluationRun, ...]): Every run performed
            for this checkpoint.
    """

    checkpoint_path: Path
    completed_iteration: int
    metadata: dict
    greedy: bool
    score_thresholds: tuple[int, ...]
    runs: tuple[RCheckpointEvaluationRun, ...]

    @property
    def number_of_runs(self) -> int:
        """int: Total number of runs performed for this checkpoint."""
        return len(self.runs)

    @property
    def mean_final_score(self) -> float:
        """float: Mean final exact score across all runs."""
        return float(
            mean(run.final_score for run in self.runs)
        )

    @property
    def median_final_score(self) -> float:
        """float: Median final exact score across all runs."""
        return float(
            median(run.final_score for run in self.runs)
        )

    @property
    def mean_best_score(self) -> float:
        """float: Mean best (lowest) exact score observed across all runs."""
        return float(
            mean(run.best_score for run in self.runs)
        )

    @property
    def median_best_score(self) -> float:
        """float: Median best (lowest) exact score observed across all runs."""
        return float(
            median(run.best_score for run in self.runs)
        )

    @property
    def minimum_best_score(self) -> int:
        """int: Lowest best exact score observed across all runs."""
        return min(run.best_score for run in self.runs)

    @property
    def mean_score_reduction(self) -> float:
        """float: Mean improvement from initial to final score across all runs."""
        return float(
            mean(run.score_reduction for run in self.runs)
        )

    @property
    def mean_best_score_reduction(self) -> float:
        """float: Mean improvement from initial to best observed score across all runs."""
        return float(
            mean(
                run.best_score_reduction
                for run in self.runs
            )
        )

    @property
    def improved_final_count(self) -> int:
        """int: Number of runs whose final score improved on the initial score."""
        return sum(
            run.score_reduction > 0
            for run in self.runs
        )

    @property
    def improved_best_count(self) -> int:
        """int: Number of runs whose best observed score improved on the initial score."""
        return sum(
            run.best_score_reduction > 0
            for run in self.runs
        )

    @property
    def threshold_counts(self) -> dict[int, int]:
        """dict[int, int]: Number of runs whose best score was at or below each threshold in ``score_thresholds``."""
        return {
            threshold: sum(
                run.best_score <= threshold
                for run in self.runs
            )
            for threshold in self.score_thresholds
        }

    @property
    def source_names(self) -> tuple[str, ...]:
        """tuple[str, ...]: Sorted, unique construction source names represented among ``runs``."""
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
        """Return only runs originating from one seed source.

        Args:
            source_name (str): Construction source name to filter
                ``runs`` by.

        Returns:
            tuple[RCheckpointEvaluationRun, ...]: The subset of ``runs``
            whose ``source_name`` equals ``source_name``, in original
            order.
        """
        return tuple(
            run
            for run in self.runs
            if run.source_name == source_name
        )


@dataclass(frozen=True, slots=True)
class RCheckpointEvaluationResult:
    """Comparable evaluations for a collection of checkpoints.

    Attributes:
        seeds (tuple[REvaluationSeed, ...]): The evaluation seeds every
            evaluated checkpoint was run against.
        evaluations (tuple[RCheckpointEvaluation, ...]): One aggregate
            evaluation per checkpoint, in the order the checkpoints
            were evaluated.
    """

    seeds: tuple[REvaluationSeed, ...]
    evaluations: tuple[RCheckpointEvaluation, ...]

    @property
    def strongest_checkpoint(
        self,
    ) -> RCheckpointEvaluation:
        """Rank checkpoints by typical performance, then minimum.

        Checkpoints are ordered by ``mean_best_score``, breaking ties by
        ``median_best_score`` and then ``minimum_best_score`` — all
        ascending, since a lower exact score is better.

        Returns:
            RCheckpointEvaluation: The evaluation with the best-ranked
            combination of mean, median, and minimum best score.

        Raises:
            RuntimeError: If ``evaluations`` is empty.
        """
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
    """Evaluate trained checkpoints without optimization or writes.

    For each checkpoint, restores the policy/value network, wraps it in
    an :class:`~ramsey.nn.RNeuralPolicy.RNeuralPolicy`, and runs it
    through :class:`~ramsey.RSearch.RSearch` once per evaluation seed
    and repetition, recording the resulting scores. No gradients are
    computed and no files are written; the network is loaded purely for
    inference.
    """

    def __init__(
        self,
        graph: RGraph,
        environment: REnvironment,
        device: torch.device | str = "auto",
    ) -> None:
        """Bind the evaluator to a fixed host graph and search environment.

        Args:
            graph (RGraph): Host graph every restored checkpoint's
                network must match.
            environment (REnvironment): Search environment every
                restored policy is run through. Its problem must match
                ``graph.problem``.
            device (torch.device | str): Device checkpoints are
                restored onto; resolved via
                :func:`~ramsey.nn.RRuntime.resolve_torch_device`.
                Defaults to ``"auto"``.

        Raises:
            ValueError: If ``environment.graph.problem`` does not match
                ``graph.problem``.
        """
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
        """Run every checkpoint under identical evaluation cases.

        Every checkpoint is evaluated against the same ``seeds`` using
        the same per-run action seeds (derived from
        ``config.action_seed``), so resulting scores are directly
        comparable across checkpoints. PyTorch's global RNG state is
        saved before evaluation and restored afterward (even if
        evaluation raises), so this call does not disturb random-number
        sequences used elsewhere.

        Args:
            checkpoint_paths (Iterable[str | Path]): Paths to the
                checkpoints to evaluate. Must be nonempty and contain no
                duplicates.
            seeds (Iterable[REvaluationSeed]): Starting colorings every
                checkpoint is evaluated against. Must be nonempty, and
                each seed's coloring must belong to a graph with the
                same problem as ``graph``.
            config (RCheckpointEvaluationConfig | None): Evaluation
                settings. Uses ``RCheckpointEvaluationConfig()`` defaults
                when ``None``.

        Returns:
            RCheckpointEvaluationResult: The evaluation seeds used and
            one :class:`RCheckpointEvaluation` per checkpoint, in the
            order of ``checkpoint_paths``.

        Raises:
            ValueError: If ``checkpoint_paths`` or ``seeds`` is empty,
                if ``checkpoint_paths`` contains duplicate paths, or if
                any seed's coloring problem does not match ``graph``.
            TypeError: If any element of ``seeds`` is not an
                :class:`REvaluationSeed`.
        """
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
        """Restore one checkpoint and run it against every seed and repetition.

        The checkpoint is restored using a throwaway NumPy generator
        (seeded with 0), since evaluation only needs the network and
        optimizer state, not reproducible NumPy sampling. Each
        ``(seed, repetition)`` combination gets its own deterministic
        action seed, derived from ``config.action_seed`` so that runs
        for the same seed/repetition pair are reproducible and directly
        comparable across checkpoints.

        Args:
            checkpoint_path (Path): Path to the checkpoint to restore
                and evaluate.
            seeds (tuple[REvaluationSeed, ...]): Starting colorings to
                run the restored policy against.
            config (RCheckpointEvaluationConfig): Evaluation settings
                controlling repetitions, action seeding, greediness, and
                score thresholds.

        Returns:
            RCheckpointEvaluation: The aggregate evaluation for this
            checkpoint, containing one :class:`RCheckpointEvaluationRun`
            per seed and repetition.
        """
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
    """Construct and freeze a reusable checkpoint seed collection.

    Repeatedly invokes ``construction.construct(graph)`` to build
    ``number_of_seeds`` starting colorings, wrapping each in an
    :class:`REvaluationSeed` so the same fixed collection of seeds can
    be reused across multiple calls to
    :meth:`RCheckpointEvaluator.evaluate`.

    Args:
        graph (RGraph): Host graph to construct starting colorings for.
        construction (RConstruction): Construction strategy used to
            build each starting coloring.
        number_of_seeds (int): Number of seeds to construct. Must be
            positive.
        name_prefix (str | None): Prefix used to build each seed's
            ``name`` as ``f"{prefix}-{index:04d}"``. When ``None``, each
            seed uses its own construction's
            ``construction.last_source_name`` as the prefix.

    Returns:
        tuple[REvaluationSeed, ...]: ``number_of_seeds`` newly
        constructed evaluation seeds.

    Raises:
        TypeError: If ``number_of_seeds`` is not an integer.
        ValueError: If ``number_of_seeds`` is not positive, or if
            ``name_prefix`` is neither ``None`` nor a nonempty string.
    """
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
    """Validate and normalize one nonnegative integer.

    Args:
        name (str): Name of the value, used only in error messages.
        value (int): Candidate value to validate.

    Returns:
        int: ``value`` converted to a built-in ``int``.

    Raises:
        TypeError: If ``value`` is a boolean or not an integral type.
        ValueError: If ``value`` is negative.
    """
    if isinstance(value, bool) or not isinstance(
        value,
        Integral,
    ):
        raise TypeError(f"{name} must be an integer.")

    value = int(value)

    if value < 0:
        raise ValueError(f"{name} cannot be negative.")

    return value