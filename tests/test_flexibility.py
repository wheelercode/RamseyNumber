"""Tests for objective-neutral local flexibility measurements."""

import numpy as np
import pytest

from ramsey.RFlexibility import calculate_flexibility


def test_flexibility_curve_counts_damage_budgets() -> None:
    rewards = np.asarray([5, 1, 0, -1, -2, -5, -8], dtype=np.int32)

    profile = calculate_flexibility(
        rewards,
        budgets=(0, 1, 2, 5),
    )

    assert np.array_equal(profile.counts, [3, 4, 5, 6])
    assert np.allclose(profile.fractions, np.asarray([3, 4, 5, 6]) / 7)


def test_action_mask_restricts_flexibility_measurement() -> None:
    rewards = np.asarray([2, 0, -1, -9], dtype=np.int32)
    mask = np.asarray([True, False, True, False])

    profile = calculate_flexibility(
        rewards,
        budgets=(0, 1),
        action_mask=mask,
    )

    assert profile.number_of_actions == 2
    assert np.array_equal(profile.counts, [1, 2])


def test_flexibility_budgets_must_be_strictly_increasing() -> None:
    with pytest.raises(ValueError):
        calculate_flexibility(
            np.asarray([0], dtype=np.int32),
            budgets=(0, 5, 2),
        )


def test_flexibility_results_are_read_only() -> None:
    profile = calculate_flexibility(
        np.asarray([1, 0, -1], dtype=np.int32),
        budgets=(0, 1),
    )

    assert not profile.budgets.flags.writeable
    assert not profile.counts.flags.writeable
    assert not profile.fractions.flags.writeable