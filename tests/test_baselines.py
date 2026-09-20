from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_uplift.causal.baselines import (
    ResponseModel,
    TreatmentFeatureModel,
    validate_model_features,
)


def model_frame(rows: int = 500, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    treatment = rng.binomial(1, 0.5, rows)
    history = rng.normal(200, 50, rows)
    probability = 1 / (1 + np.exp(-(-4 + 0.008 * history + 1.2 * treatment)))
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
            "conversion": rng.binomial(1, probability),
        }
    )


def test_feature_guard_rejects_outcomes() -> None:
    with pytest.raises(ValueError, match="invalid model features"):
        validate_model_features(("history", "conversion"))


def test_response_model_returns_probabilities() -> None:
    frame = model_frame()
    probability = ResponseModel(seed=3).fit(frame).predict(frame)
    assert probability.shape == (len(frame),)
    assert ((probability >= 0) & (probability <= 1)).all()


def test_treatment_feature_model_scores_both_counterfactual_inputs() -> None:
    frame = model_frame()
    model = TreatmentFeatureModel(seed=3).fit(frame)
    treated, control = model.potential_predictions(frame)
    np.testing.assert_allclose(model.predict_uplift(frame), treated - control)
    assert ((treated >= 0) & (treated <= 1)).all()
    assert ((control >= 0) & (control <= 1)).all()
