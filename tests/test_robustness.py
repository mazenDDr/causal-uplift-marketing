from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_uplift.evaluation.robustness import (
    add_noise_covariates,
    joint_stratified_subsample,
    shuffle_treatment,
)


def robustness_frame(rows: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(8)
    return pd.DataFrame(
        {
            "source_row_id": np.arange(rows),
            "treatment": np.tile([0, 0, 1, 1], rows // 4),
            "conversion": np.tile([0, 1, 0, 1], rows // 4),
            "history": rng.normal(size=rows),
            "spend": rng.gamma(1, 10, rows),
        }
    )


def test_joint_subsample_preserves_rows_and_each_joint_stratum() -> None:
    frame = robustness_frame()
    sample = joint_stratified_subsample(frame, 0.25, seed=4)
    assert len(sample) == 100
    assert sample.groupby(["treatment", "conversion"]).size().eq(25).all()
    source = frame.set_index("source_row_id")
    actual = sample.set_index("source_row_id")
    pd.testing.assert_frame_equal(actual, source.loc[actual.index])


def test_noise_covariates_do_not_modify_existing_data() -> None:
    frame = robustness_frame()
    noisy, names = add_noise_covariates(frame, columns=3, seed=9)
    pd.testing.assert_frame_equal(noisy[frame.columns], frame)
    assert names == ("noise_0", "noise_1", "noise_2")
    assert np.isfinite(noisy.loc[:, names].to_numpy()).all()


def test_placebo_is_a_treatment_permutation_only() -> None:
    frame = robustness_frame()
    placebo = shuffle_treatment(frame, seed=11)
    pd.testing.assert_frame_equal(
        placebo.drop(columns="treatment"),
        frame.drop(columns="treatment"),
    )
    assert sorted(placebo["treatment"]) == sorted(frame["treatment"])
    assert not placebo["treatment"].equals(frame["treatment"])


def test_invalid_sample_fraction_is_rejected() -> None:
    with pytest.raises(ValueError, match="fraction"):
        joint_stratified_subsample(robustness_frame(), 0, seed=1)
