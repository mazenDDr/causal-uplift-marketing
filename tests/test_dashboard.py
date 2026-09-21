from __future__ import annotations

from pathlib import Path

import pytest

from causal_uplift.dashboard import (
    MODEL_SPECS,
    balance_rows,
    effect_summary,
    load_dashboard_results,
    policy_curve,
    policy_summary,
    qini_curve,
)


@pytest.fixture(scope="module")
def results() -> dict:
    return load_dashboard_results(Path(__file__).resolve().parents[1])


def test_dashboard_loads_only_committed_evidence(results: dict) -> None:
    assert results["foundation"]["rows"]["rct_evaluation"] == 17046
    assert set(results["ablation"]["samples"]) == {"randomized", "weak", "medium", "strong"}


def test_balance_uses_pre_and_post_match_values(results: dict) -> None:
    rows = balance_rows(results, "medium")
    assert len(rows) == 11
    assert max(row["before"] for row in rows) == pytest.approx(0.5843357664)
    assert max(row["after"] for row in rows) == pytest.approx(0.0256804763)


def test_each_model_has_a_measured_policy(results: dict) -> None:
    for model_name in MODEL_SPECS:
        policy, scope = policy_summary(results, "medium", model_name, 0.20, 0.05)
        profit = policy["incremental_profit_per_1000_eligible"]
        assert policy["targeted_customers"] == 3409
        assert profit["ci_lower"] <= profit["estimate"] <= profit["ci_upper"]
        assert scope == "full policy grid"


def test_cost_change_only_changes_cost_and_profit(results: dict) -> None:
    low, _ = policy_summary(results, "strong", "CausalForestDML", 0.20, 0.01)
    high, _ = policy_summary(results, "strong", "CausalForestDML", 0.20, 0.50)
    assert (
        low["incremental_revenue_per_1000_eligible"]
        == high["incremental_revenue_per_1000_eligible"]
    )
    assert low["campaign_cost_per_1000_eligible"] < high["campaign_cost_per_1000_eligible"]
    assert (
        low["incremental_profit_per_1000_eligible"]["estimate"]
        > high["incremental_profit_per_1000_eligible"]["estimate"]
    )


def test_non_medium_budget_is_not_extrapolated(results: dict) -> None:
    with pytest.raises(ValueError, match="20% budget"):
        policy_summary(results, "strong", "CausalForestDML", 0.30, 0.05)


def test_estimand_and_ranking_availability_are_explicit(results: dict) -> None:
    assert effect_summary(results, "medium", "Response model") is None
    assert effect_summary(results, "medium", "PSM segments")["absolute_error_vs_rct_ate"] is None
    assert qini_curve(results, "LinearDML") is None
    rows, qini = qini_curve(results, "CausalForestDML")
    assert len(rows) == 21
    assert qini["ci_lower"] < 0 < qini["ci_upper"]


def test_policy_curve_uses_every_frozen_budget(results: dict) -> None:
    rows = policy_curve(results, "Response model", 0.05)
    assert [row["budget"] for row in rows] == results["business"]["settings"]["budgets"]
