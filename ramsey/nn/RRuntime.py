"""Reproducible NumPy and PyTorch runtime initialization."""

from __future__ import annotations

from numbers import Integral

import numpy as np
import torch


def create_numpy_generator(
    seed: int,
) -> np.random.Generator:
    """
    Return a new NumPy generator initialized from a validated seed.

    Separate generators should be created for logically independent
    experiments so one workflow cannot silently consume another
    workflow's random-number sequence.

    Args:
        seed (int): Nonnegative integer seed. Validated with
            ``_validate_seed`` before being passed to
            ``numpy.random.default_rng``.

    Returns:
        numpy.random.Generator: A freshly constructed generator seeded
        independently of any other generator in the process.

    Raises:
        TypeError: If ``seed`` is not an integer (or is a ``bool``).
        ValueError: If ``seed`` is negative.
    """
    return np.random.default_rng(
        _validate_seed(seed)
    )


def seed_torch(
    seed: int,
) -> None:
    """
    Seed PyTorch's CPU and available CUDA random generators.

    This controls random-number initialization but does not enable
    deterministic algorithms, which can reduce performance and are
    a separate runtime choice.

    Args:
        seed (int): Nonnegative integer seed. Validated with
            ``_validate_seed`` before being applied to PyTorch's
            generators.

    Raises:
        TypeError: If ``seed`` is not an integer (or is a ``bool``).
        ValueError: If ``seed`` is negative.
    """
    seed = _validate_seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_torch_device(
    device: torch.device | str | None = "auto",
) -> torch.device:
    """
    Resolve automatic or unindexed CUDA device specifications.

    ``None`` and ``"auto"`` select the current CUDA device when CUDA
    is available and otherwise select the CPU. An explicit unindexed
    CUDA device is converted to the current indexed CUDA device so
    comparisons such as ``cuda`` versus ``cuda:0`` remain reliable.
    Other explicit PyTorch device specifications are preserved.

    Args:
        device (torch.device | str | None): Requested device.
            ``None`` or ``"auto"`` triggers automatic resolution; any
            other string or ``torch.device`` is treated as an
            explicit request. Defaults to ``"auto"``.

    Returns:
        torch.device: The resolved, comparison-stable device.
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device(
                "cuda",
                torch.cuda.current_device(),
            )

        return torch.device("cpu")

    resolved_device = torch.device(device)

    if (
        resolved_device.type == "cuda"
        and resolved_device.index is None
        and torch.cuda.is_available()
    ):
        return torch.device(
            "cuda",
            torch.cuda.current_device(),
        )

    return resolved_device


def _validate_seed(
    seed: int,
) -> int:
    """
    Return a nonnegative built-in integer seed.

    Args:
        seed (int): Candidate seed value, which must be an integral
            type (excluding ``bool``) and nonnegative.

    Returns:
        int: ``seed`` coerced to a built-in ``int``.

    Raises:
        TypeError: If ``seed`` is not an integer (or is a ``bool``).
        ValueError: If ``seed`` is negative.
    """
    if isinstance(seed, bool) or not isinstance(
        seed,
        Integral,
    ):
        raise TypeError("seed must be an integer.")

    seed = int(seed)

    if seed < 0:
        raise ValueError("seed cannot be negative.")

    return seed