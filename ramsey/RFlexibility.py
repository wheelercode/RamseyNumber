"""Objective-neutral measurements of local search flexibility."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Iterable

import numpy as np
from numpy.typing import NDArray


DEFAULT_FLEXIBILITY_BUDGETS = (0, 1, 2, 5, 10, 20)


def _read_only_copy(array: NDArray) -> NDArray:
    """Return an owned, read-only copy of ``array``.

    Args:
        array (numpy.ndarray): Source array or array-like value to copy.

    Returns:
        numpy.ndarray: A new array that owns its memory, with the
        ``writeable`` flag cleared.
    """
    result = np.asarray(array).copy()
    result.flags.writeable = False
    return result


def _normalize_budgets(
    budgets: Iterable[int],
) -> NDArray[np.int32]:
    """Validate and canonicalize a sequence of flexibility damage budgets.

    Args:
        budgets (Iterable[int]): Candidate damage budgets, which must be
            non-negative integers given in strictly increasing order.

    Returns:
        numpy.ndarray: ``int32`` array of the validated budget values,
        in the order supplied.

    Raises:
        TypeError: If any budget is not an integer (or is a ``bool``).
        ValueError: If ``budgets`` is empty, contains a negative value,
            or is not strictly increasing.
    """
    values = []

    for value in budgets:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("flexibility budgets must be integers.")

        value = int(value)

        if value < 0:
            raise ValueError("flexibility budgets cannot be negative.")

        values.append(value)

    if not values:
        raise ValueError("at least one flexibility budget is required.")

    result = np.asarray(values, dtype=np.int32)

    if np.any(result[1:] <= result[:-1]):
        raise ValueError(
            "flexibility budgets must be strictly increasing."
        )

    return result


@dataclass(frozen=True, slots=True, eq=False)
class RFlexibilityProfile:
    """The fraction of available actions within each damage budget.

    Produced by :func:`calculate_flexibility`. For each budget ``b`` in
    :attr:`budgets`, the profile records how many of the considered
    actions have an exact reward of at least ``-b`` (i.e. worsen the
    score by no more than ``b``), both as a raw count and as a fraction
    of the total.

    Attributes:
        budgets (numpy.ndarray): Read-only ``int32`` array of strictly
            increasing, non-negative damage budgets.
        counts (numpy.ndarray): Read-only ``int32`` array, parallel to
            ``budgets``, of how many considered actions have exact
            reward at least ``-budget``.
        fractions (numpy.ndarray): Read-only ``float64`` array, parallel
            to ``budgets``, of ``counts / number_of_actions``.
        number_of_actions (int): Total number of actions considered
            when computing the profile (after any ``action_mask`` was
            applied).
    """

    budgets: NDArray[np.int32]
    counts: NDArray[np.int32]
    fractions: NDArray[np.float64]
    number_of_actions: int

    def __post_init__(self) -> None:
        """Validate array shapes and value ranges, then freeze the arrays.

        Raises:
            ValueError: If ``budgets`` is not one-dimensional, if
                ``counts`` or ``fractions`` does not have one value per
                budget, if ``number_of_actions`` is not positive, if any
                count is out of range ``[0, number_of_actions]``, or if
                any fraction is out of range ``[0.0, 1.0]``.
        """
        budgets = np.asarray(self.budgets, dtype=np.int32)
        counts = np.asarray(self.counts, dtype=np.int32)
        fractions = np.asarray(self.fractions, dtype=np.float64)

        if budgets.ndim != 1:
            raise ValueError("budgets must be one-dimensional.")

        expected_shape = budgets.shape

        if counts.shape != expected_shape:
            raise ValueError("counts must have one value per budget.")

        if fractions.shape != expected_shape:
            raise ValueError("fractions must have one value per budget.")

        if self.number_of_actions <= 0:
            raise ValueError("number_of_actions must be positive.")

        if np.any(counts < 0) or np.any(counts > self.number_of_actions):
            raise ValueError("flexibility counts are out of range.")

        if np.any(fractions < 0.0) or np.any(fractions > 1.0):
            raise ValueError("flexibility fractions are out of range.")

        object.__setattr__(self, "budgets", _read_only_copy(budgets))
        object.__setattr__(self, "counts", _read_only_copy(counts))
        object.__setattr__(self, "fractions", _read_only_copy(fractions))
        object.__setattr__(self, "number_of_actions", int(self.number_of_actions))

    def fraction_at(self, budget: int) -> float:
        """Return F(budget), requiring that budget to be in the profile.

        Args:
            budget (int): Damage budget to look up. Must be one of the
                values in :attr:`budgets`.

        Returns:
            float: F(budget), the fraction of considered actions whose
            exact reward is at least ``-budget``.

        Raises:
            KeyError: If ``budget`` is not present in :attr:`budgets`.
        """
        indexes = np.flatnonzero(self.budgets == budget)

        if indexes.size == 0:
            raise KeyError(f"budget {budget} is not present in the profile.")

        return float(self.fractions[int(indexes[0])])

    def summary(self) -> dict[str, int | float]:
        """Return copy-friendly names such as flexibility_5.

        Returns:
            dict[str, int | float]: ``"flexibility_action_count"`` mapped
            to :attr:`number_of_actions`, plus one ``"flexibility_{b}"``
            entry per budget ``b`` in :attr:`budgets` mapped to its
            fraction.
        """
        result: dict[str, int | float] = {
            "flexibility_action_count": self.number_of_actions,
        }

        for budget, fraction in zip(self.budgets, self.fractions):
            result[f"flexibility_{int(budget)}"] = float(fraction)

        return result


def calculate_flexibility(
    rewards: NDArray[np.integer],
    budgets: Iterable[int] = DEFAULT_FLEXIBILITY_BUDGETS,
    *,
    action_mask: NDArray[np.bool_] | None = None,
) -> RFlexibilityProfile:
    """
    Calculate the local flexibility curve from exact action rewards.

    F(b) is the fraction of considered actions whose exact score reward
    is at least ``-b``. Thus F(0) counts improving and neutral actions;
    F(5) additionally permits actions that temporarily worsen the score
    by at most five.

    ``action_mask`` can restrict the measurement to actions available to
    a particular search mechanism. Omit it to measure the structural
    flexibility of the complete action space.

    Args:
        rewards (numpy.ndarray): One-dimensional array of exact
            immediate rewards (score reduction) for every candidate
            action, e.g. ``RActionAnalysis.immediate_rewards``.
        budgets (Iterable[int]): Strictly increasing, non-negative
            damage budgets ``b`` at which to evaluate F(b). Defaults to
            :data:`DEFAULT_FLEXIBILITY_BUDGETS`.
        action_mask (numpy.ndarray | None): Optional boolean array, the
            same shape as ``rewards``, selecting which actions to
            include in the measurement. Omit to consider every action
            in ``rewards``.

    Returns:
        RFlexibilityProfile: The flexibility profile giving, for every
        requested budget, the count and fraction of considered actions
        whose exact reward is at least ``-budget``.

    Raises:
        ValueError: If ``rewards`` is not one-dimensional, if
            ``budgets`` is empty, contains a negative or non-increasing
            value, if ``action_mask`` does not match the shape of
            ``rewards``, or if no actions remain to measure after
            masking.
        TypeError: If any budget is not an integer (or is a ``bool``).
    """
    rewards = np.asarray(rewards)

    if rewards.ndim != 1:
        raise ValueError("rewards must be one-dimensional.")

    normalized_budgets = _normalize_budgets(budgets)

    if action_mask is None:
        selected_rewards = rewards
    else:
        action_mask = np.asarray(action_mask, dtype=np.bool_)

        if action_mask.shape != rewards.shape:
            raise ValueError("action_mask must have the same shape as rewards.")

        selected_rewards = rewards[action_mask]

    if selected_rewards.size == 0:
        raise ValueError("flexibility requires at least one action.")

    counts = np.count_nonzero(
        selected_rewards[:, np.newaxis]
        >= -normalized_budgets[np.newaxis, :],
        axis=0,
    ).astype(np.int32)

    fractions = counts.astype(np.float64) / selected_rewards.size

    return RFlexibilityProfile(
        budgets=normalized_budgets,
        counts=counts,
        fractions=fractions,
        number_of_actions=int(selected_rewards.size),
    )