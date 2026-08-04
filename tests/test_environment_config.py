"""Test immutable environment and memory configuration."""

from dataclasses import FrozenInstanceError

import pytest

from ramsey.REnvironmentConfig import (
    REnvironmentConfig,
    RTabuMemoryConfig,
)


def test_environment_config_defaults() -> None:
    config = REnvironmentConfig()

    assert config.max_steps == 1_000
    assert config.use_aspiration


def test_tabu_memory_config_defaults() -> None:
    config = RTabuMemoryConfig()

    assert config.edge_tenure == 20
    assert config.visited_state_window == 2_000


@pytest.mark.parametrize(
    "max_steps",
    [0, -1],
)
def test_environment_rejects_invalid_max_steps(
    max_steps: int,
) -> None:
    with pytest.raises(ValueError):
        REnvironmentConfig(max_steps=max_steps)


@pytest.mark.parametrize(
    ("name", "values"),
    [
        (
            "edge_tenure",
            {"edge_tenure": -1},
        ),
        (
            "visited_state_window",
            {"visited_state_window": -1},
        ),
    ],
)
def test_memory_rejects_negative_values(
    name: str,
    values: dict[str, int],
) -> None:
    with pytest.raises(
        ValueError,
        match=name,
    ):
        RTabuMemoryConfig(**values)


def test_configurations_are_frozen() -> None:
    config = REnvironmentConfig()

    with pytest.raises(FrozenInstanceError):
        config.max_steps = 12
