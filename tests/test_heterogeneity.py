from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("econml")

from causal_uplift.causal.causal_forest import (  # noqa: E402
    customer_segments,
    effect_bins,
)
from causal_uplift.evaluation.heterogeneity import randomized_group_effects  # noqa: E402


def test_effect_bins_are_balanced_and_ordered() -> None:
    scores = np.array([0.4, -0.1, 0.2, 0.9, 0.5, 0.0, 0.3, 0.8, 0.7, 0.6])
    bins = effect_bins(scores, bins=5)
    assert np.bincount(bins)[1:].tolist() == [2, 2, 2, 2, 2]
    means = [scores[bins == label].mean() for label in range(1, 6)]
    assert means == sorted(means)


def test_customer_segments_use_fixed_training_thresholds() -> None:
    frame = pd.DataFrame({"history": [50, 150, 50, 150], "recency": [2, 2, 8, 8]})
    segments = customer_segments(frame, history_threshold=100, recency_threshold=5)
    assert segments.tolist() == [
        "recent / low history",
        "recent / high history",
        "inactive / low history",
        "inactive / high history",
    ]


def test_randomized_group_effects_recover_group_ordering() -> None:
    rng = np.random.default_rng(18)
    rows = 10000
    groups = np.where(np.arange(rows) < rows // 2, "low", "high")
    treatment = rng.binomial(1, 0.5, rows)
    probability = 0.10 + treatment * np.where(groups == "high", 0.12, 0.01)
    frame = pd.DataFrame(
        {
            "treatment": treatment,
            "conversion": rng.binomial(1, probability),
        }
    )
    predicted = np.where(groups == "high", 0.12, 0.01)
    results = randomized_group_effects(
        frame,
        groups,
        predicted,
        bootstrap_samples=100,
        seed=3,
    )
    observed = {row["group"]: row["observed_rct_uplift"] for row in results}
    assert observed["high"] > observed["low"]
    assert observed["high"] == pytest.approx(0.12, abs=0.04)
