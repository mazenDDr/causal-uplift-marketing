from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("causalml")

from causal_uplift.causal.uplift import (  # noqa: E402
    fit_uplift_tree,
    make_uplift_encoder,
    treatment_labels,
    uplift_tree_leaves,
)
from causal_uplift.data.load import PRE_TREATMENT_COLUMNS  # noqa: E402


def synthetic_frame(rows: int = 3000, seed: int = 22) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    history = rng.normal(200, 50, rows)
    high_effect = history < 200
    treatment = rng.binomial(1, 0.5, rows)
    probability = 0.10 + treatment * np.where(high_effect, 0.15, 0.01)
    return pd.DataFrame(
        {
            "history": history,
            "recency": rng.integers(1, 13, rows),
            "history_segment": np.where(history > 200, "high", "low"),
            "mens": rng.binomial(1, 0.5, rows),
            "womens": rng.binomial(1, 0.5, rows),
            "zip_code": rng.choice(["Urban", "Rural", "Surburban"], rows),
            "newbie": rng.binomial(1, 0.3, rows),
            "channel": rng.choice(["Web", "Phone", "Multichannel"], rows),
            "treatment": treatment,
            "conversion": rng.binomial(1, probability),
        }
    )


def test_uplift_encoder_excludes_treatment_and_outcome() -> None:
    frame = synthetic_frame(200)
    encoder = make_uplift_encoder().fit(frame.loc[:, PRE_TREATMENT_COLUMNS])
    assert set(encoder.feature_names_in_) == set(PRE_TREATMENT_COLUMNS)
    assert "conversion" not in encoder.feature_names_in_
    assert "treatment" not in encoder.feature_names_in_


def test_treatment_labels_require_both_groups() -> None:
    labels = treatment_labels(
        np.array([0, 1, 0]),
        control_name="control",
        treatment_name="email",
    )
    assert labels.tolist() == ["control", "email", "control"]
    with pytest.raises(ValueError, match="both binary treatment groups"):
        treatment_labels(
            np.array([1, 1]),
            control_name="control",
            treatment_name="email",
        )


def test_uplift_tree_ranks_known_high_effect_group_higher() -> None:
    frame = synthetic_frame()
    model = fit_uplift_tree(
        frame,
        control_name="control",
        treatment_name="email",
        evaluation_function="KL",
        honesty=True,
        estimation_sample_size=0.5,
        max_depth=3,
        min_samples_leaf=100,
        min_samples_treatment=50,
        n_reg=50,
        seed=5,
    )
    score = model.predict(frame)
    leaves = uplift_tree_leaves(model, control_name="control", treatment_name="email")
    assert np.isfinite(score).all()
    assert score[frame["history"] < 200].mean() > score[frame["history"] >= 200].mean()
    assert leaves
    assert all(set(leaf["group_samples"]) == {"control", "email"} for leaf in leaves)
