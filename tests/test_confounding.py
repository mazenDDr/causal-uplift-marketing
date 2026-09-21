from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_uplift.data.confounding import (
    ConfoundingConfig,
    desired_propensity,
    sample_observational,
)


def synthetic_rct(rows: int = 30_000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    history = rng.lognormal(mean=5.0, sigma=0.9, size=rows)
    return pd.DataFrame(
        {
            "source_row_id": np.arange(rows),
            "history": history,
            "recency": rng.integers(1, 13, rows),
            "mens": rng.binomial(1, 0.5, rows),
            "womens": rng.binomial(1, 0.5, rows),
            "newbie": rng.binomial(1, 0.25, rows),
            "channel": rng.choice(["Phone", "Web", "Multichannel"], rows),
            "treatment": rng.binomial(1, 0.5, rows),
            "visit": rng.binomial(1, 0.2, rows),
            "conversion": rng.binomial(1, 0.05, rows),
            "spend": rng.gamma(1.0, 10.0, rows),
        }
    )


def standardized_difference(frame: pd.DataFrame, column: str) -> float:
    treated = frame.loc[frame["treatment"] == 1, column].astype(float)
    control = frame.loc[frame["treatment"] == 0, column].astype(float)
    pooled = np.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2)
    return float((treated.mean() - control.mean()) / pooled)


def test_propensity_is_clipped() -> None:
    propensity = desired_propensity(synthetic_rct(), ConfoundingConfig(strength=8.0))
    assert propensity.min() >= 0.05
    assert propensity.max() <= 0.95


def test_sampling_never_changes_treatment_or_outcomes() -> None:
    source = synthetic_rct()
    selected = sample_observational(source, ConfoundingConfig(strength=1.0), seed=42)
    original = source.set_index("source_row_id")
    actual = selected.set_index("source_row_id")
    pd.testing.assert_frame_equal(
        actual[["treatment", "visit", "conversion", "spend"]],
        original.loc[actual.index, ["treatment", "visit", "conversion", "spend"]],
    )


def test_zero_strength_keeps_the_full_randomized_pool() -> None:
    source = synthetic_rct()
    selected = sample_observational(source, ConfoundingConfig(strength=0.0), seed=42)
    assert selected["source_row_id"].tolist() == source["source_row_id"].tolist()
    assert (selected["selection_probability"] == 1.0).all()


def test_stronger_confounding_creates_more_history_imbalance() -> None:
    source = synthetic_rct()
    weak = sample_observational(source, ConfoundingConfig(strength=0.5), seed=42)
    strong = sample_observational(source, ConfoundingConfig(strength=2.0), seed=42)
    assert abs(standardized_difference(strong, "history")) > abs(
        standardized_difference(weak, "history")
    )


def test_post_treatment_feature_cannot_replace_required_feature() -> None:
    source = synthetic_rct().drop(columns="history")
    with pytest.raises(ValueError, match="confounding features are missing"):
        desired_propensity(source, ConfoundingConfig(strength=1.0))


def test_campaign_affinity_column_changes_only_the_pre_treatment_score() -> None:
    source = synthetic_rct()
    mens_score = desired_propensity(
        source,
        ConfoundingConfig(strength=1.0, affinity_column="mens"),
    )
    womens_score = desired_propensity(
        source,
        ConfoundingConfig(strength=1.0, affinity_column="womens"),
    )
    assert not np.array_equal(mens_score, womens_score)
    assert np.corrcoef(womens_score, source["womens"])[0, 1] > 0


def test_invalid_campaign_affinity_is_rejected() -> None:
    with pytest.raises(ValueError, match="affinity_column"):
        ConfoundingConfig(strength=1.0, affinity_column="conversion")
