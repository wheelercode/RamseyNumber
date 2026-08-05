"""Test complete PPO training orchestration."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ramsey.RArchive import RSQLiteArchive
from ramsey.RColoring import RColoring
from ramsey.RConstruction import (
    RFixedConstruction,
    RMixedConstruction,
)
from ramsey.REnvironment import REnvironment
from ramsey.REnvironmentConfig import (
    REnvironmentConfig,
)
from ramsey.REnvironmentMemory import (
    RNullMemory,
)
from ramsey.RGraph import RGraph
from ramsey.RObjective import (
    RMonochromaticObjective,
)
from ramsey.RProblem import RProblem
from ramsey.nn.RModel import (
    RModelConfig,
    RPairPolicyValueNetwork,
)
from ramsey.nn.RPPO import (
    RPPOConfig,
    create_optimizer,
)
from ramsey.nn.RRollout import (
    RRolloutConfig,
)
from ramsey.nn.RTraining import (
    RCheckpointSchedule,
    RPPOTrainer,
    RTrainingConfig,
)


def make_trainer(
    *,
    n_vertices: int = 7,
    max_steps: int = 2,
    colors: np.ndarray | None = None,
    archive=None,
    construction=None,
) -> RPPOTrainer:
    """Create a small complete training assembly."""

    graph = RGraph(
        RProblem.r55(
            n_vertices=n_vertices,
        )
    )

    if colors is None:
        colors = np.zeros(
            graph.number_of_edges,
            dtype=np.uint8,
        )

    if construction is None:
        construction = RFixedConstruction(
            RColoring(
                graph,
                colors,
            )
        )

    environment = REnvironment(
        graph=graph,
        objective=RMonochromaticObjective(),
        memory=RNullMemory(),
        config=REnvironmentConfig(
            max_steps=max_steps,
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
        value_loss_weight=0.5,
        learning_rate=1.0e-3,
    )

    return RPPOTrainer(
        graph=graph,
        construction=construction,
        environment=environment,
        network=network,
        optimizer=create_optimizer(
            network,
            ppo_config,
        ),
        device="cpu",
        rng=np.random.default_rng(901),
        rollout_config=(
            RRolloutConfig(
                rollout_steps=max_steps,
                normalize_advantages=True,
            )
        ),
        ppo_config=ppo_config,
        archive=archive,
    )


@pytest.mark.parametrize(
    (
        "values",
        "exception_type",
    ),
    [
        (
            {
                "run_name": "",
                "iterations": 1,
            },
            ValueError,
        ),
        (
            {
                "run_name": "run",
                "iterations": 0,
            },
            ValueError,
        ),
        (
            {
                "run_name": "run",
                "iterations": 1.5,
            },
            TypeError,
        ),
        (
            {
                "run_name": "run",
                "iterations": 1,
                "start_iteration": -1,
            },
            ValueError,
        ),
        (
            {
                "run_name": "run",
                "iterations": 1,
                "stop_on_solution": 1,
            },
            TypeError,
        ),
    ],
)
def test_training_config_validation(
    values,
    exception_type,
) -> None:
    with pytest.raises(exception_type):
        RTrainingConfig(**values)


@pytest.mark.parametrize(
    (
        "values",
        "exception_type",
    ),
    [
        (
            {
                "directory": "checkpoints",
                "interval": 0,
            },
            ValueError,
        ),
        (
            {
                "directory": "checkpoints",
                "interval": 1.5,
            },
            TypeError,
        ),
        (
            {
                "directory": "checkpoints",
                "save_final": 1,
            },
            TypeError,
        ),
    ],
)
def test_checkpoint_schedule_validation(
    values,
    exception_type,
) -> None:
    with pytest.raises(exception_type):
        RCheckpointSchedule(**values)


def test_checkpoint_schedule_builds_deterministic_path(
    tmp_path,
) -> None:
    schedule = RCheckpointSchedule(
        tmp_path,
        interval=5,
    )

    assert schedule.path_for(17) == (tmp_path / "ramsey_policy_iteration_000017.pt")


def test_trainer_updates_network_archives_and_notifies_observer(
    tmp_path,
) -> None:
    torch.manual_seed(902)

    observed = []

    with RSQLiteArchive(tmp_path / "training.sqlite3") as archive:
        trainer = make_trainer(
            archive=archive,
        )

        before = [
            parameter.detach().clone() for parameter in trainer.network.parameters()
        ]

        result = trainer.run(
            RTrainingConfig(
                run_name="ppo-test",
                iterations=2,
                start_iteration=10,
            ),
            observer=observed.append,
        )

        assert result.completed_iterations == 2

        assert [item.iteration for item in result.iteration_results] == [
            10,
            11,
        ]

        assert observed == list(result.iteration_results)

        assert all(item.parameter_update_performed for item in result.iteration_results)

        assert all(item.archive_record is not None for item in result.iteration_results)

        assert all(
            item.construction_name == "fixed"
            for item in result.iteration_results
        )

        assert all(
            item.construction_source == "fixed"
            for item in result.iteration_results
        )

        assert archive.coloring_count(trainer.graph) >= 1

        assert result.best_score == result.best_iteration.best_score

        assert any(
            not torch.equal(
                old_parameter,
                new_parameter,
            )
            for (
                old_parameter,
                new_parameter,
            ) in zip(
                before,
                trainer.network.parameters(),
            )
        )

        assert not hasattr(
            result.iteration_results[0],
            "rollout",
        )


def test_trainer_reports_selected_mixed_construction_source() -> None:
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
        rng=np.random.default_rng(904),
        construction_name="mixed-curriculum",
    )

    trainer = make_trainer(
        construction=mixed,
    )

    result = trainer.run(
        RTrainingConfig(
            run_name="mixed-source-test",
            iterations=1,
        )
    )

    iteration = result.iteration_results[0]

    assert iteration.construction_name == "mixed-curriculum"
    assert iteration.construction_source == "archive-source"


def test_trainer_saves_periodic_and_final_checkpoints(
    tmp_path,
) -> None:
    torch.manual_seed(903)

    trainer = make_trainer()

    result = trainer.run(
        RTrainingConfig(
            run_name="checkpoint-test",
            iterations=3,
        ),
        checkpoint_schedule=(
            RCheckpointSchedule(
                tmp_path / "checkpoints",
                interval=2,
                save_final=True,
            )
        ),
    )

    paths = [item.checkpoint_path for item in result.iteration_results]

    assert paths[0] is None

    assert paths[1] is not None and paths[1].exists()

    assert paths[2] is not None and paths[2].exists()


def test_trainer_skips_update_and_stops_for_terminal_seed() -> None:
    graph = RGraph(
        RProblem.r55(
            n_vertices=5,
        )
    )

    colors = np.zeros(
        graph.number_of_edges,
        dtype=np.uint8,
    )

    # This destroys the single monochromatic K5 in K5,
    # making the initial coloring a score-zero solution.
    colors[0] = 1

    trainer = make_trainer(
        n_vertices=5,
        max_steps=4,
        colors=colors,
    )

    result = trainer.run(
        RTrainingConfig(
            run_name="solved",
            iterations=10,
        )
    )

    assert result.completed_iterations == 1

    assert result.solved

    assert result.iteration_results[0].metrics is None

    assert not (result.iteration_results[0].parameter_update_performed)


def test_trainer_rejects_mismatched_environment_problem() -> None:
    trainer = make_trainer(
        n_vertices=7,
    )

    other_graph = RGraph(
        RProblem.r55(
            n_vertices=8,
        )
    )

    environment = REnvironment(
        graph=other_graph,
        objective=RMonochromaticObjective(),
        memory=RNullMemory(),
    )

    fixed_coloring = RColoring(
        trainer.graph,
        np.zeros(
            trainer.graph.number_of_edges,
            dtype=np.uint8,
        ),
    )

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        RPPOTrainer(
            graph=trainer.graph,
            construction=RFixedConstruction(fixed_coloring),
            environment=environment,
            network=trainer.network,
            optimizer=trainer.optimizer,
            device="cpu",
            rng=np.random.default_rng(904),
            rollout_config=trainer.rollout_config,
            ppo_config=trainer.ppo_config,
        )