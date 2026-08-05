"""Optional neural encoding, models, rollouts, and training."""

# The core ramsey package must never import this package.

from .REncoding import (
    AVAILABILITY_CHANNEL,
    DIAGONAL_CHANNEL,
    EDGE_COLOR_CHANNEL,
    PAIR_INPUT_CHANNELS,
    build_network_input,
    edge_permutation_indices,
    encode_pair_features,
)
from .RModel import (
    RModelConfig,
    RPairMessageLayer,
    RPairPolicyValueNetwork,
)
from .RNeuralPolicy import (
    RNeuralDecision,
    RNeuralPolicy,
)
from .RRuntime import (
    create_numpy_generator,
    resolve_torch_device,
    seed_torch,
)
from .RRollout import (
    RRolloutBatch,
    RRolloutConfig,
    RRolloutReward,
    calculate_advantages,
    collect_rollout,
)
from .RCheckpoint import (
    CHECKPOINT_VERSION,
    RTrainingCheckpoint,
    load_training_checkpoint,
    save_training_checkpoint,
)
from .RPPO import (
    RPPOConfig,
    RPPOMetrics,
    create_optimizer,
    ppo_update,
)
from .RTraining import (
    RCheckpointSchedule,
    RPPOTrainer,
    RTrainingConfig,
    RTrainingIteration,
    RTrainingObserver,
    RTrainingResult,
)
from .REvaluation import (
    RCheckpointEvaluation,
    RCheckpointEvaluationConfig,
    RCheckpointEvaluationResult,
    RCheckpointEvaluationRun,
    RCheckpointEvaluator,
    REvaluationSeed,
    build_evaluation_seeds,
)

__all__ = [
    "AVAILABILITY_CHANNEL",
    "DIAGONAL_CHANNEL",
    "EDGE_COLOR_CHANNEL",
    "PAIR_INPUT_CHANNELS",
    "RModelConfig",
    "RPairMessageLayer",
    "RPairPolicyValueNetwork",
    "build_network_input",
    "edge_permutation_indices",
    "encode_pair_features",
    "RNeuralDecision",
    "RNeuralPolicy",
    "create_numpy_generator",
    "resolve_torch_device",
    "seed_torch",
    "RRolloutBatch",
    "RRolloutConfig",
    "RRolloutReward",
    "calculate_advantages",
    "collect_rollout",
    "CHECKPOINT_VERSION",
    "RPPOConfig",
    "RPPOMetrics",
    "RTrainingCheckpoint",
    "create_optimizer",
    "load_training_checkpoint",
    "ppo_update",
    "save_training_checkpoint",
    "RCheckpointSchedule",
    "RPPOTrainer",
    "RTrainingConfig",
    "RTrainingIteration",
    "RTrainingObserver",
    "RTrainingResult",
    "RCheckpointEvaluation",
    "RCheckpointEvaluationConfig",
    "RCheckpointEvaluationResult",
    "RCheckpointEvaluationRun",
    "RCheckpointEvaluator",
    "REvaluationSeed",
    "build_evaluation_seeds",
]