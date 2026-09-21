from __future__ import annotations

import numpy as np
import pandas as pd

from causal_uplift.evaluation.uplift_metrics import (
    bootstrap_ranking_metrics,
    ranking_metrics,
    tie_aware_gain_curve,
)


def randomized_example(rows: int = 4000, seed: int = 8) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    segment = rng.binomial(1, 0.5, rows)
    treatment = rng.binomial(1, 0.5, rows)
    probability = 0.10 + treatment * np.where(segment == 1, 0.20, 0.0)
    frame = pd.DataFrame({"treatment": treatment, "conversion": rng.binomial(1, probability)})
    return frame, segment.astype(float)


def test_high_effect_ranking_beats_reverse_ranking() -> None:
    frame, score = randomized_example()
    fractions = np.linspace(0, 1, 11)
    good = ranking_metrics(frame["conversion"], frame["treatment"], score, fractions)
    reverse = ranking_metrics(frame["conversion"], frame["treatment"], -score, fractions)
    assert good["qini"] > 0
    assert good["qini"] > reverse["qini"]


def test_tie_aware_curve_is_invariant_to_row_order() -> None:
    frame, score = randomized_example(1000)
    fractions = np.linspace(0, 1, 21)
    original = tie_aware_gain_curve(frame["conversion"], frame["treatment"], score, fractions)
    order = np.random.default_rng(3).permutation(len(frame))
    shuffled = tie_aware_gain_curve(
        frame["conversion"].to_numpy()[order],
        frame["treatment"].to_numpy()[order],
        score[order],
        fractions,
    )
    np.testing.assert_allclose(original["gain"], shuffled["gain"])


def test_full_curve_gain_equals_randomized_ate() -> None:
    frame, score = randomized_example(2000)
    curve = tie_aware_gain_curve(
        frame["conversion"], frame["treatment"], score, np.linspace(0, 1, 11)
    )
    expected = (
        frame.loc[frame["treatment"] == 1, "conversion"].mean()
        - frame.loc[frame["treatment"] == 0, "conversion"].mean()
    )
    assert np.isclose(curve["gain"][-1], expected)
    assert np.isclose(curve["qini_gain"][-1], 0.0)


def test_bootstrap_reports_intervals_and_paired_difference() -> None:
    frame, score = randomized_example(1000)
    result = bootstrap_ranking_metrics(
        frame,
        {"response_model": -score, "uplift_model": score},
        np.linspace(0, 1, 6),
        samples=30,
        seed=7,
    )
    uplift = result["models"]["uplift_model"]
    assert uplift["qini"]["ci_lower"] <= uplift["qini"]["ci_upper"]
    assert len(uplift["curve"]["gain_ci_lower"]) == 6
    paired = result["paired_qini_difference_vs_reference"]["models"]["uplift_model"]
    assert paired["estimate"] > 0
