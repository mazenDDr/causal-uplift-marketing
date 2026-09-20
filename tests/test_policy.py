from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_uplift.evaluation.policy import (
    bootstrap_policy_difference,
    evaluate_binary_policy,
    random_policy_scores,
    top_fraction_mask,
)


def test_top_fraction_selects_exact_budget() -> None:
    mask = top_fraction_mask(np.arange(100), 0.20)
    assert mask.sum() == 20
    assert mask[-20:].all()


def test_random_policy_is_independent_of_row_order() -> None:
    frame = pd.DataFrame({"source_row_id": [10, 20, 30]})
    shuffled = frame.iloc[[2, 0, 1]]
    original = dict(zip(frame["source_row_id"], random_policy_scores(frame, seed=42), strict=True))
    reordered = dict(
        zip(shuffled["source_row_id"], random_policy_scores(shuffled, seed=42), strict=True)
    )
    assert original == reordered


def test_policy_evaluation_uses_only_selected_rows() -> None:
    frame = pd.DataFrame(
        {
            "treatment": [0, 0, 1, 1, 0, 1],
            "conversion": [0, 0, 1, 1, 1, 0],
        }
    )
    result = evaluate_binary_policy(
        frame,
        np.array([True, True, True, True, False, False]),
        bootstrap_samples=20,
    )
    assert result["observed_uplift"] == pytest.approx(1.0)
    assert result["incremental_conversions_per_1000"] == pytest.approx(1000.0)


def test_paired_policy_difference_uses_the_same_bootstrap_rows() -> None:
    group_size = 100
    frame = pd.DataFrame(
        {
            "treatment": [0] * group_size + [1] * group_size,
            "conversion": [0] * group_size + [1] * (group_size // 2) + [0] * (group_size // 2),
        }
    )
    policy_a = np.array(
        [True] * (group_size // 2)
        + [False] * (group_size // 2)
        + [True] * (group_size // 2)
        + [False] * (group_size // 2)
    )
    policy_b = ~policy_a
    result = bootstrap_policy_difference(
        frame,
        policy_a,
        policy_b,
        samples=100,
        seed=7,
    )
    assert result["uplift_difference"] == pytest.approx(1.0)
