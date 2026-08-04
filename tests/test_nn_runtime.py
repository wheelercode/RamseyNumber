"""Test reproducible NumPy and PyTorch runtime initialization."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ramsey.nn.RRuntime import (
    create_numpy_generator,
    resolve_torch_device,
    seed_torch,
)


def test_numpy_generators_are_reproducible_and_independent() -> None:
    first = create_numpy_generator(1_001)

    second = create_numpy_generator(1_001)

    expected = first.integers(
        0,
        10_000,
        size=20,
    )

    actual = second.integers(
        0,
        10_000,
        size=20,
    )

    assert np.array_equal(
        actual,
        expected,
    )

    first.integers(
        0,
        10_000,
        size=10,
    )

    assert not np.array_equal(
        first.integers(
            0,
            10_000,
            size=20,
        ),
        second.integers(
            0,
            10_000,
            size=20,
        ),
    )


def test_torch_seeding_is_reproducible() -> None:
    seed_torch(1_002)

    expected = torch.rand(20)

    seed_torch(1_002)

    actual = torch.rand(20)

    assert torch.equal(
        actual,
        expected,
    )


@pytest.mark.parametrize(
    (
        "function",
        "seed",
        "exception_type",
    ),
    [
        (
            create_numpy_generator,
            True,
            TypeError,
        ),
        (
            create_numpy_generator,
            1.5,
            TypeError,
        ),
        (
            seed_torch,
            -1,
            ValueError,
        ),
    ],
)
def test_runtime_seed_validation(
    function,
    seed,
    exception_type,
) -> None:
    with pytest.raises(exception_type):
        function(seed)


def test_automatic_device_uses_cpu_without_cuda(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: False,
    )

    assert resolve_torch_device() == torch.device("cpu")

    assert resolve_torch_device(None) == torch.device("cpu")


def test_automatic_device_uses_current_cuda_device(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: True,
    )

    monkeypatch.setattr(
        torch.cuda,
        "current_device",
        lambda: 2,
    )

    assert resolve_torch_device() == torch.device("cuda:2")

    assert resolve_torch_device("cuda") == torch.device("cuda:2")


def test_explicit_device_is_preserved() -> None:
    assert resolve_torch_device("cpu") == torch.device("cpu")

    assert resolve_torch_device("cuda:3") == torch.device("cuda:3")

    assert resolve_torch_device("meta") == torch.device("meta")