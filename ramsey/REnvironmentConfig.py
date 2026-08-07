"""Immutable configuration for search environments and memory."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral


def _require_nonnegative_integer(
    name: str,
    value: int,
) -> int:
    """Validate and normalize a nonnegative integer configuration field.

    Args:
        name (str): Field name used to build a descriptive error message.
        value (int): Candidate value to validate.

    Returns:
        int: ``value`` coerced to a plain ``int``.

    Raises:
        TypeError: If ``value`` is a ``bool`` or is not an ``Integral``.
        ValueError: If ``value`` is negative.
    """
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")

    value = int(value)

    if value < 0:
        raise ValueError(f"{name} cannot be negative.")

    return value


@dataclass(frozen=True, slots=True)
class REnvironmentConfig:
    """
    Configuration governing an environment episode.

    This contains behavior belonging to the environment itself,
    rather than to a particular optimization objective.

    Attributes:
        max_steps (int): Maximum number of edge flips permitted in one
            episode before the environment reports it as truncated.
        use_aspiration (bool): Whether a blocked action is allowed anyway
            when it would establish a new best exact score for the
            episode (the aspiration criterion of tabu search).
    """

    # Maximum number of edge flips in one episode.
    max_steps: int = 1_000

    # Allow a blocked action when it establishes a new best score.
    use_aspiration: bool = True

    def __post_init__(self) -> None:
        """Validate and normalize the configured field values.

        Raises:
            TypeError: If ``max_steps`` is not an integer or
                ``use_aspiration`` is not a ``bool``.
            ValueError: If ``max_steps`` is not positive.
        """
        max_steps = _require_nonnegative_integer(
            "max_steps",
            self.max_steps,
        )

        if max_steps == 0:
            raise ValueError("max_steps must be positive.")

        if not isinstance(
            self.use_aspiration,
            bool,
        ):
            raise TypeError("use_aspiration must be boolean.")

        object.__setattr__(
            self,
            "max_steps",
            max_steps,
        )


@dataclass(frozen=True, slots=True)
class RTabuMemoryConfig:
    """
    Configuration for tabu and visited-state memory.

    Attributes:
        edge_tenure (int): Number of future steps for which a just-flipped
            edge remains tabu (blocked from being flipped again). Zero
            disables edge-tenure blocking.
        visited_state_window (int): Number of most recently visited exact
            colorings remembered for revisit blocking. Zero disables
            visited-state memory entirely.
    """

    # Number of future steps for which a flipped edge is blocked.
    edge_tenure: int = 20

    # Number of exact recently visited colorings to remember.
    # Set this to zero to disable visited-state memory.
    visited_state_window: int = 2_000

    def __post_init__(self) -> None:
        """Validate and normalize the configured field values.

        Raises:
            TypeError: If either field is not an integer.
            ValueError: If either field is negative.
        """
        object.__setattr__(
            self,
            "edge_tenure",
            _require_nonnegative_integer(
                "edge_tenure",
                self.edge_tenure,
            ),
        )

        object.__setattr__(
            self,
            "visited_state_window",
            _require_nonnegative_integer(
                "visited_state_window",
                self.visited_state_window,
            ),
        )
