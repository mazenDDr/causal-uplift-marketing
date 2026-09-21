from __future__ import annotations

import numpy as np
import pandas as pd

from causal_uplift.evaluation.business import (
    bootstrap_profit_difference,
    cost_aware_mask,
    evaluate_business_policy,
    psm_segment_scores,
    ranked_policy_mask,
)


def test_ranked_policy_uses_explicit_tie_breaker() -> None:
    scores = np.array([1.0, 1.0, 0.0, 0.0])
    tie_breaker = np.array([0.1, 0.9, 0.2, 0.8])
    selected = ranked_policy_mask(scores, 0.5, tie_breaker=tie_breaker)
    assert selected.tolist() == [True, True, False, False]
    one = ranked_policy_mask(scores, 0.25, tie_breaker=tie_breaker)
    assert one.tolist() == [False, True, False, False]


def test_cost_aware_rule_uses_incremental_conversion_value() -> None:
    selected = cost_aware_mask(
        np.array([-0.01, 0.001, 0.01]),
        conversion_value=50.0,
        email_cost=0.25,
    )
    assert selected.tolist() == [False, False, True]


def test_business_metrics_and_identical_policy_difference() -> None:
    rows = 2000
    rng = np.random.default_rng(5)
    treatment = rng.binomial(1, 0.5, rows)
    frame = pd.DataFrame(
        {
            "treatment": treatment,
            "conversion": rng.binomial(1, 0.1 + 0.05 * treatment),
            "visit": rng.binomial(1, 0.3 + 0.1 * treatment),
            "spend": 5 + 2 * treatment + rng.normal(0, 0.2, rows),
        }
    )
    selected = np.arange(rows) % 2 == 0
    result = evaluate_business_policy(
        frame,
        selected,
        email_cost=0.5,
        bootstrap_samples=30,
        seed=4,
    )
    assert result["campaign_cost_per_1000_eligible"] == 250.0
    assert result["incremental_profit_per_1000_eligible"]["estimate"] > 600
    difference = bootstrap_profit_difference(
        frame,
        selected,
        selected,
        email_cost=0.5,
        samples=20,
        seed=4,
    )
    assert difference == {"estimate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}


def test_psm_segment_scores_use_only_matched_pair_differences() -> None:
    training = pd.DataFrame(
        {
            "history": [200, 100, 210, 90],
            "recency": [2, 8, 3, 9],
            "conversion": [1, 0, 0, 0],
        }
    )
    pairs = pd.DataFrame({"treated_position": [0, 1], "control_position": [2, 3]})
    score, segments = psm_segment_scores(
        training,
        pairs,
        training,
        history_threshold=150,
        recency_threshold=6,
    )
    assert score.tolist() == [1.0, 0.0, 1.0, 0.0]
    assert sum(row["matched_pairs"] for row in segments) == 2
