"""Test reproducible, no-training checkpoint evaluation."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ramsey.RColoring import RColoring
from ramsey.RConstruction import (
    RFixedConstruction,
    RMixedConstruction,
)
from ramsey.REnvironment import REnvironment
from ramsey.REnvironmentConfig import REnvironmentConfig
from ramsey.REnvironmentMemory import RNullMemory
from ramsey.RGraph import RGraph
from ramsey.RObjective import RMonochromaticObjective
from ramsey.RProblem import RProblem
from ramsey.nn.RCheckpoint import save_training_checkpoint
from ramsey.nn.REvaluation import (
    RCheckpointEvaluationConfig,
    RCheckpointEvaluator,
    REvaluationSeed,
    build_evaluation_seeds,
)
from ramsey.nn.RModel import (
    RModelConfig,
    RPairPolicyValueNetwork,
)
from ramsey.nn.RPPO import RPPOConfig, create_optimizer
from ramsey.nn.RRollout import RRolloutConfig


def make_evaluation_components(tmp_path):
    """Create two identical checkpoints and a small environment."""
    graph = RGraph(
        RProblem.r55(
            n_vertices=7,
        )
    )

    environment = REnvironment(
        graph=graph,
        objective=RMonochromaticObjective(),
        memory=RNullMemory(),
        config=REnvironmentConfig(
            max_steps=3,
        ),
    )

    network = RPairPolicyValueNetwork(
        graph,
        RModelConfig(
            hidden_size=8,
            number_of_layers=1,
        ),
    )

    ppo_config = RPPOConfig(
        update_epochs=1,
        minibatch_size=2,
    )

    optimizer = create_optimizer(
        network,
        ppo_config,
    )

    paths = (
        tmp_path / "checkpoint-a.pt",
        tmp_path / "checkpoint-b.pt",
    )

    for iteration, path in enumerate(paths):
        save_training_checkpoint(
            path,
            network=network,
            optimizer=optimizer,
            graph=graph,
            rollout_config=RRolloutConfig(
                rollout_steps=3,
            ),
            ppo_config=ppo_config,
            completed_iteration=iteration,
            rng=np.random.default_rng(
                500 + iteration
            ),
            metadata={
                "label": path.stem,
            },
        )

    return (
        graph,
        environment,
        paths,
    )


def test_checkpoint_evaluator_uses_identical_cases_and_preserves_rng(
    tmp_path,
) -> None:
    torch.manual_seed(510)

    (
        graph,
        environment,
        paths,
    ) = make_evaluation_components(tmp_path)

    seeds = (
        REvaluationSeed(
            name="red",
            source_name="fixed",
            coloring=RColoring(
                graph,
                np.zeros(
                    graph.number_of_edges,
                    dtype=np.uint8,
                ),
            ),
        ),
        REvaluationSeed(
            name="blue",
            source_name="fixed",
            coloring=RColoring(
                graph,
                np.ones(
                    graph.number_of_edges,
                    dtype=np.uint8,
                ),
            ),
        ),
    )

    torch_rng_state = torch.get_rng_state().clone()

    result = RCheckpointEvaluator(
        graph,
        environment,
        "cpu",
    ).evaluate(
        paths,
        seeds,
        RCheckpointEvaluationConfig(
            repetitions_per_seed=2,
            action_seed=600,
            score_thresholds=(
                20,
                10,
                0,
            ),
        ),
    )

    assert torch.equal(
        torch.get_rng_state(),
        torch_rng_state,
    )

    assert len(result.evaluations) == 2

    first, second = result.evaluations

    assert first.number_of_runs == 4
    assert second.number_of_runs == 4

    first_scores = tuple(
        (
            run.initial_score,
            run.final_score,
            run.best_score,
            run.action_seed,
        )
        for run in first.runs
    )

    second_scores = tuple(
        (
            run.initial_score,
            run.final_score,
            run.best_score,
            run.action_seed,
        )
        for run in second.runs
    )

    assert first_scores == second_scores

    assert first.completed_iteration == 0
    assert second.completed_iteration == 1

    assert first.metadata == {
        "label": "checkpoint-a",
    }

    assert first.source_names == ("fixed",)
    assert len(first.runs_for_source("fixed")) == 4

    assert set(first.threshold_counts) == {
        20,
        10,
        0,
    }

    assert result.strongest_checkpoint is first


def test_build_evaluation_seeds_records_selected_source() -> None:
    graph = RGraph(
        RProblem.r55(
            n_vertices=7,
        )
    )

    fixed = RFixedConstruction(
        RColoring(
            graph,
            np.zeros(
                graph.number_of_edges,
                dtype=np.uint8,
            ),
        ),
        construction_name="archive-source",
    )

    mixed = RMixedConstruction(
        constructions=(fixed,),
        probabilities=(1.0,),
        rng=np.random.default_rng(520),
        construction_name="mixed",
    )

    seeds = build_evaluation_seeds(
        graph,
        mixed,
        2,
        name_prefix="benchmark",
    )

    assert tuple(
        seed.name
        for seed in seeds
    ) == (
        "benchmark-0000",
        "benchmark-0001",
    )

    assert all(
        seed.source_name == "archive-source"
        for seed in seeds
    )


@pytest.mark.parametrize(
    (
        "values",
        "exception_type",
    ),
    [
        (
            {
                "repetitions_per_seed": 0,
            },
            ValueError,
        ),
        (
            {
                "action_seed": -1,
            },
            ValueError,
        ),
        (
            {
                "greedy": 1,
            },
            TypeError,
        ),
        (
            {
                "score_thresholds": (
                    500,
                    500,
                ),
            },
            ValueError,
        ),
    ],
)
def test_checkpoint_evaluation_config_validation(
    values,
    exception_type,
) -> None:
    with pytest.raises(exception_type):
        RCheckpointEvaluationConfig(**values)