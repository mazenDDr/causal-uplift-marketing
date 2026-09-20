from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("econml")

from causal_uplift.causal.dml import (  # noqa: E402
    fit_linear_dml,
    make_confounder_encoder,
    make_cross_fit_splits,
)
from causal_uplift.data.load import PRE_TREATMENT_COLUMNS  # noqa: E402


def synthetic_frame(rows: int = 2500, seed: int = 14) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    history = rng.normal(200, 50, rows)
    recency = rng.integers(1, 13, rows)
    mens = rng.binomial(1, 0.5, rows)
    treatment_probability = 1 / (1 + np.exp(-0.012 * (history - 200) + 0.5 * mens))
    treatment = rng.binomial(1, treatment_probability)
    baseline = 0.12 + 0.08 * (history > 200) + 0.03 * mens
    conversion = rng.binomial(1, baseline + 0.08 * treatment)
    return pd.DataFrame(
        {
            "history": history,
            "recency": recency,
            "history_segment": np.where(history > 200, "high", "low"),
            "mens": mens,
            "womens": rng.binomial(1, 0.5, rows),
            "zip_code": rng.choice(["Urban", "Rural", "Surburban"], rows),
            "newbie": rng.binomial(1, 0.3, rows),
            "channel": rng.choice(["Web", "Phone", "Multichannel"], rows),
            "treatment": treatment,
            "conversion": conversion,
        }
    )


def test_encoder_uses_only_pre_treatment_features() -> None:
    frame = synthetic_frame(200)
    encoder = make_confounder_encoder().fit(frame.loc[:, PRE_TREATMENT_COLUMNS])
    assert set(encoder.feature_names_in_) == set(PRE_TREATMENT_COLUMNS)
    assert "conversion" not in encoder.feature_names_in_
    assert "treatment" not in encoder.feature_names_in_


def test_cross_fit_splits_cover_each_row_once_without_train_overlap() -> None:
    frame = synthetic_frame(500)
    splits = make_cross_fit_splits(frame["conversion"], frame["treatment"], folds=5, seed=2)
    test_rows = np.concatenate([test for _, test in splits])
    assert len(splits) == 5
    assert sorted(test_rows) == list(range(len(frame)))
    assert all(set(train).isdisjoint(test) for train, test in splits)


def test_linear_dml_approximately_recovers_known_effect() -> None:
    frame = synthetic_frame()
    splits = make_cross_fit_splits(frame["conversion"], frame["treatment"], folds=5, seed=5)
    result = fit_linear_dml(
        frame,
        "conversion",
        candidate="hist_gradient_boosting",
        splits=splits,
        seed=5,
        min_samples_leaf=50,
        max_iter=80,
    )
    assert result.cross_fit_folds == 5
    assert result.estimate == pytest.approx(0.08, abs=0.04)
    assert result.ci_lower < result.estimate < result.ci_upper
