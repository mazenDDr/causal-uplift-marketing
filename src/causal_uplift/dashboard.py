from __future__ import annotations

import copy
import json
from pathlib import Path

RESULT_FILES = {
    "foundation": "foundation_summary.json",
    "propensity": "propensity_diagnostics_summary.json",
    "matching": "matching_summary.json",
    "ablation": "confounding_ablation_summary.json",
    "forest": "causal_forest_summary.json",
    "uplift_models": "uplift_models_summary.json",
    "uplift_metrics": "uplift_metrics_summary.json",
    "business": "business_policy_summary.json",
}

MODEL_SPECS = {
    "Naive pseudo-uplift": {
        "ate": "naive",
        "policy": "pseudo_uplift",
        "ranking": "pseudo_uplift",
        "estimand": "Raw association; not a causal estimand",
    },
    "Response model": {
        "ate": None,
        "policy": "response_model",
        "ranking": "response_model",
        "estimand": "Predicts conversion, not treatment effect",
    },
    "PSM segments": {
        "ate": "psm",
        "policy": "psm_segments",
        "ranking": None,
        "estimand": "ATT in matched observational treated customers",
    },
    "LinearDML": {
        "ate": "linear_dml",
        "policy": "linear_dml",
        "ranking": None,
        "estimand": "ATE on the randomized-holdout covariate distribution",
    },
    "CausalForestDML": {
        "ate": "causal_forest_dml",
        "policy": "causal_forest_dml",
        "ranking": "causal_forest_dml",
        "estimand": "ATE and heterogeneous effects on the holdout covariate distribution",
    },
    "Uplift random forest": {
        "ate": "uplift_random_forest",
        "policy": "uplift_random_forest",
        "ranking": "uplift_random_forest",
        "estimand": "Mean predicted uplift; no explicit propensity adjustment",
    },
}


def load_dashboard_results(repository: Path) -> dict:
    result_dir = repository / "experiments" / "results"
    results = {}
    for name, filename in RESULT_FILES.items():
        path = result_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"dashboard result is missing: {path}")
        results[name] = json.loads(path.read_text())
    return results


def balance_rows(results: dict, strength: str) -> list[dict]:
    before = results["propensity"]["standardized_mean_differences"][strength]
    matching_sample = results["matching"]["samples"][strength]
    variant = matching_sample["preferred_variant"]
    after = matching_sample["variants"][variant]["standardized_mean_differences"]
    return [
        {"covariate": covariate, "before": abs(value), "after": abs(after[covariate])}
        for covariate, value in before.items()
    ]


def propensity_quantiles(results: dict, strength: str) -> list[dict]:
    sample = results["propensity"]["samples"][strength]
    return [
        {
            "quantile": float(quantile),
            "treated": value,
            "control": sample["control_quantiles"][quantile],
        }
        for quantile, value in sample["treated_quantiles"].items()
    ]


def effect_summary(results: dict, strength: str, model_name: str) -> dict | None:
    spec = MODEL_SPECS[model_name]
    if spec["ate"] is None:
        return None
    effect = copy.deepcopy(results["ablation"]["samples"][strength]["ate_estimators"][spec["ate"]])
    effect["rct_estimate"] = results["ablation"]["rct_effect"]["estimate"]
    effect["rct_ci_lower"] = results["ablation"]["rct_effect"]["ci_lower"]
    effect["rct_ci_upper"] = results["ablation"]["rct_effect"]["ci_upper"]
    effect["model_estimand"] = spec["estimand"]
    return effect


def _profit_at_cost(record: dict, email_cost: float, *, recorded_cost: float) -> dict:
    adjusted = copy.deepcopy(record)
    targeted_cost = adjusted["targeted_fraction"] * 1000 * email_cost
    old_targeted_cost = adjusted["targeted_fraction"] * 1000 * recorded_cost
    stored_profit = adjusted["incremental_profit_per_1000_eligible"]
    revenue_interval = {
        "estimate": adjusted["incremental_revenue_per_1000_eligible"],
        "ci_lower": stored_profit["ci_lower"] + old_targeted_cost,
        "ci_upper": stored_profit["ci_upper"] + old_targeted_cost,
    }
    adjusted["campaign_cost_per_1000_eligible"] = targeted_cost
    adjusted["incremental_profit_per_1000_eligible"] = {
        name: value - targeted_cost for name, value in revenue_interval.items()
    }
    return adjusted


def policy_summary(
    results: dict,
    strength: str,
    model_name: str,
    budget: float,
    email_cost: float,
) -> tuple[dict, str]:
    policy = MODEL_SPECS[model_name]["policy"]
    if strength == "medium":
        record = copy.deepcopy(results["business"]["top_k_policies"][policy][f"{budget:.2f}"])
        stored_cost = results["business"]["settings"]["primary_email_cost"]
        return _profit_at_cost(record, email_cost, recorded_cost=stored_cost), "full policy grid"

    if abs(budget - 0.20) > 1e-9:
        raise ValueError("non-medium confounding results are measured only at a 20% budget")
    if policy == "linear_dml":
        policy = "random"
        note = "20% ablation; constant-effect LinearDML uses the seeded random ranking"
    else:
        note = "20% confounding ablation"
    record = copy.deepcopy(results["ablation"]["samples"][strength]["policies"][policy])
    stored_cost = results["ablation"]["policy_setting"]["email_cost"]
    return _profit_at_cost(record, email_cost, recorded_cost=stored_cost), note


def policy_curve(results: dict, model_name: str, email_cost: float) -> list[dict]:
    policy = MODEL_SPECS[model_name]["policy"]
    stored_cost = results["business"]["settings"]["primary_email_cost"]
    rows = []
    for budget, record in results["business"]["top_k_policies"][policy].items():
        adjusted = _profit_at_cost(record, email_cost, recorded_cost=stored_cost)
        rows.append(
            {
                "budget": float(budget),
                "profit": adjusted["incremental_profit_per_1000_eligible"]["estimate"],
            }
        )
    return rows


def qini_curve(results: dict, model_name: str) -> tuple[list[dict], dict] | None:
    ranking = MODEL_SPECS[model_name]["ranking"]
    if ranking is None:
        return None
    model = results["uplift_metrics"]["models"][ranking]
    curve = model["curve"]
    overall = results["foundation"]["rct_effects"]["conversion"]["estimate"]
    rows = [
        {
            "fraction": fraction,
            "model_gain": gain,
            "random_gain": fraction * overall,
        }
        for fraction, gain in zip(curve["fraction"], curve["gain"], strict=True)
    ]
    return rows, model["qini"]


def heterogeneity_rows(results: dict, strength: str, model_name: str) -> list[dict]:
    if model_name == "CausalForestDML":
        return [
            {
                "segment": row["group"],
                "predicted": row["predicted_cate_mean"],
                "observed": row["observed_rct_uplift"],
                "ci_lower": row["observed_ci_lower"],
                "ci_upper": row["observed_ci_upper"],
            }
            for row in results["forest"]["subgroups"]
        ]
    if model_name == "Uplift random forest":
        deciles = results["uplift_models"]["models"]["uplift_random_forest"]["deciles"]
        return [
            {
                "segment": f"Decile {row['group']}",
                "predicted": row["predicted_cate_mean"],
                "observed": row["observed_rct_uplift"],
                "ci_lower": row["observed_ci_lower"],
                "ci_upper": row["observed_ci_upper"],
            }
            for row in deciles
        ]
    if model_name == "PSM segments":
        return [
            {
                "segment": row["segment"],
                "predicted": row["conversion_att"],
                "observed": None,
                "ci_lower": None,
                "ci_upper": None,
            }
            for row in results["ablation"]["samples"][strength]["psm_segments"]
        ]
    return []
