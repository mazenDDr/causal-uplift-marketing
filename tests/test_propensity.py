from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from causal_uplift.causal.propensity import (
    LogisticPropensityModel,
    propensity_diagnostics,
)


def propensity_frame(rows: int = 3000, seed: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    history = rng.normal(200, 60, rows)
    logit = -1.0 + 0.012 * (history - 200) + 0.8 * rng.binomial(1, 0.5, rows)
    treatment = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return pd.DataFrame(
        {
            "history": history,
            "recency": rng.integers(1, 13, rows),
            "history_segment": rng.choice(["low", "high"], rows),
            "mens": rng.binomial(1, 0.5, rows),
            "womens": rng.binomial(1, 0.5, rows),
            "zip_code": rng.choice(["Urban", "Rural"], rows),
            "newbie": rng.binomial(1, 0.3, rows),
            "channel": rng.choice(["Web", "Phone"], rows),
            "treatment": treatment,
        }
    )


def test_logistic_propensity_recovers_predictable_assignment() -> None:
    frame = propensity_frame()
    propensity = LogisticPropensityModel(seed=2).fit(frame).predict(frame)
    assert ((propensity >= 0) & (propensity <= 1)).all()
    assert roc_auc_score(frame["treatment"], propensity) > 0.65


def test_diagnostics_report_common_support_and_extremes() -> None:
    frame = pd.DataFrame({"treatment": [0, 0, 1, 1]})
    propensity = np.array([0.01, 0.40, 0.60, 0.99])
    diagnostics = propensity_diagnostics(frame, propensity)
    assert diagnostics["outside_overlap_threshold_fraction"] == pytest.approx(0.5)
    assert diagnostics["common_support_lower"] == pytest.approx(0.60)
    assert diagnostics["common_support_upper"] == pytest.approx(0.40)


def test_missing_pre_treatment_feature_is_rejected() -> None:
    frame = propensity_frame().drop(columns="history")
    with pytest.raises(ValueError, match="propensity features are missing"):
        LogisticPropensityModel().fit(frame)
