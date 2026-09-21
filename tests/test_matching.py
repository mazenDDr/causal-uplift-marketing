from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_uplift.causal.matching import (
    match_on_propensity,
    matched_frame,
    paired_att,
)
from causal_uplift.evaluation.balance import covariate_smds


def matching_frame() -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.DataFrame(
        {
            "source_row_id": np.arange(8),
            "treatment": [1, 1, 1, 1, 0, 0, 0, 0],
            "history": [10, 20, 30, 40, 11, 21, 80, 90],
            "conversion": [1, 0, 1, 1, 0, 0, 0, 0],
        }
    )
    return frame, np.array([0.20, 0.30, 0.40, 0.50, 0.21, 0.31, 0.80, 0.90])


def test_without_replacement_has_unique_controls_and_respects_caliper() -> None:
    frame, propensity = matching_frame()
    result = match_on_propensity(
        frame,
        propensity,
        replacement=False,
        caliper_sd=1.0,
        seed=4,
    )
    assert result.pairs["control_source_row_id"].is_unique
    assert (result.pairs["logit_distance"] <= result.caliper).all()


def test_with_replacement_can_reuse_nearest_control() -> None:
    frame = pd.DataFrame(
        {
            "source_row_id": [0, 1, 2, 3],
            "treatment": [1, 1, 1, 0],
            "history": [10, 11, 12, 10],
            "conversion": [1, 0, 1, 0],
        }
    )
    propensity = np.array([0.50, 0.51, 0.52, 0.50])
    result = match_on_propensity(
        frame,
        propensity,
        replacement=True,
        caliper_sd=4.0,
    )
    assert len(result.pairs) == 3
    assert result.pairs["control_source_row_id"].nunique() == 1


def test_one_to_two_matching_uses_two_distinct_controls_per_treated() -> None:
    frame, propensity = matching_frame()
    result = match_on_propensity(
        frame,
        propensity,
        replacement=True,
        matching_ratio=2,
        caliper_sd=2.0,
    )
    assert result.matching_ratio == 2
    assert len(result.pairs) == result.matched_treated * 2
    assert (result.pairs.groupby("pair_id")["control_source_row_id"].nunique() == 2).all()


def test_one_to_two_att_averages_controls_within_treated_customer() -> None:
    frame = pd.DataFrame(
        {
            "source_row_id": [0, 1, 2],
            "treatment": [1, 0, 0],
            "conversion": [1, 0, 1],
        }
    )
    result = match_on_propensity(
        frame,
        np.array([0.50, 0.49, 0.51]),
        replacement=True,
        matching_ratio=2,
        caliper_sd=2.0,
    )
    estimate = paired_att(frame, result.pairs, "conversion", bootstrap_samples=20, seed=3)
    assert estimate["estimate"] == pytest.approx(0.5)


def test_matching_preserves_source_outcomes_and_treatment() -> None:
    frame, propensity = matching_frame()
    original = frame.copy(deep=True)
    result = match_on_propensity(frame, propensity, replacement=True, caliper_sd=1.0)
    materialized = matched_frame(frame, result.pairs)
    pd.testing.assert_frame_equal(frame, original)
    for _, row in materialized.iterrows():
        source = original.loc[original["source_row_id"] == row["source_row_id"]].iloc[0]
        assert row["conversion"] == source["conversion"]
        assert row["treatment"] == source["treatment"]


def test_paired_att_and_bootstrap_recover_pair_difference() -> None:
    frame, propensity = matching_frame()
    result = match_on_propensity(frame, propensity, replacement=True, caliper_sd=1.0)
    estimate = paired_att(frame, result.pairs, "conversion", bootstrap_samples=100, seed=9)
    expected = np.mean(
        frame.iloc[result.pairs["treated_position"]]["conversion"].to_numpy()
        - frame.iloc[result.pairs["control_position"]]["conversion"].to_numpy()
    )
    assert estimate["estimate"] == pytest.approx(expected)
    assert estimate["ci_lower"] <= estimate["estimate"] <= estimate["ci_upper"]


def test_matching_reduces_synthetic_history_imbalance() -> None:
    rng = np.random.default_rng(12)
    history = rng.normal(size=2000)
    propensity = 1 / (1 + np.exp(-1.5 * history))
    treatment = rng.binomial(1, propensity)
    frame = pd.DataFrame(
        {
            "source_row_id": np.arange(len(history)),
            "treatment": treatment,
            "history": history,
            "conversion": rng.binomial(1, 0.1, len(history)),
        }
    )
    before = abs(covariate_smds(frame, ["history"])["history"])
    result = match_on_propensity(frame, propensity, replacement=True, caliper_sd=0.2)
    after = abs(covariate_smds(matched_frame(frame, result.pairs), ["history"])["history"])
    assert after < before
    assert after < 0.10
