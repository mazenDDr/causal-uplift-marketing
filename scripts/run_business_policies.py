from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from causal_uplift.evaluation.business import (
    bootstrap_profit_difference,
    cost_aware_mask,
    evaluate_business_policy,
    psm_segment_scores,
    ranked_policy_mask,
)
from causal_uplift.evaluation.policy import bootstrap_policy_difference, random_policy_scores
from causal_uplift.visualization.plots import (
    plot_policy_profit_curves,
    plot_prediction_vs_uplift_decision,
    plot_primary_policy_intervals,
)

DISPLAY_NAMES = {
    "random": "Random",
    "response_model": "Response model",
    "pseudo_uplift": "Treatment-as-feature",
    "psm_segments": "PSM segments",
    "linear_dml": "LinearDML (constant)",
    "causal_forest_dml": "CausalForestDML",
    "uplift_tree": "Uplift tree",
    "uplift_random_forest": "Uplift random forest",
}


def _load_score_file(
    evaluation: pd.DataFrame,
    path: str,
    columns: tuple[str, ...],
) -> dict[str, np.ndarray]:
    scored = pd.read_csv(path).sort_values("source_row_id")
    if not np.array_equal(scored["source_row_id"].to_numpy(), evaluation["source_row_id"]):
        raise AssertionError(f"scores in {path} do not align with the frozen RCT holdout")
    if "treatment" in scored and not np.array_equal(scored["treatment"], evaluation["treatment"]):
        raise AssertionError(f"treatment labels in {path} changed")
    if "conversion" in scored and not np.array_equal(
        scored["conversion"], evaluation["conversion"]
    ):
        raise AssertionError(f"outcomes in {path} changed")
    return {column: scored[column].to_numpy(dtype=float) for column in columns}


def _profit_at_cost(metrics: dict[str, object], email_cost: float) -> dict[str, float]:
    fraction = metrics["targeted_fraction"]
    spend = metrics["spend"]
    if spend is None:
        return {"estimate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
    cost = fraction * email_cost * 1000
    return {
        "estimate": spend["estimate"] * fraction * 1000 - cost,
        "ci_lower": spend["ci_lower"] * fraction * 1000 - cost,
        "ci_upper": spend["ci_upper"] * fraction * 1000 - cost,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    policy_config = config["business_policy"]
    seed = int(config["project"]["seed"])
    bootstrap_samples = int(policy_config["bootstrap_samples"])
    primary_budget = float(policy_config["primary_budget"])
    failure_budget = float(policy_config["decision_failure_budget"])
    primary_cost = float(policy_config["primary_email_cost"])
    email_costs = [float(value) for value in policy_config["email_costs"]]
    budgets = [float(value) for value in policy_config["budgets"]]

    evaluation = pd.read_csv("data/processed/rct_evaluation.csv").sort_values("source_row_id")
    naive = _load_score_file(
        evaluation,
        "outputs/naive_baselines/rct_policy_scores.csv",
        ("response_score", "pseudo_uplift_score"),
    )
    causal_forest = _load_score_file(
        evaluation,
        "outputs/causal_forest/rct_cate_scores.csv",
        ("predicted_cate",),
    )
    uplift = _load_score_file(
        evaluation,
        "outputs/uplift_models/rct_uplift_scores.csv",
        ("uplift_tree_score", "uplift_forest_score"),
    )
    random_score = random_policy_scores(evaluation, seed=seed)

    training = pd.read_csv("data/observational/medium.csv")
    pairs = pd.read_csv("outputs/matching/medium_with_replacement_pairs.csv")
    history_threshold = float(training["history"].median())
    recency_threshold = float(training["recency"].median())
    psm_score, psm_segments = psm_segment_scores(
        training,
        pairs,
        evaluation,
        history_threshold=history_threshold,
        recency_threshold=recency_threshold,
    )
    dml_summary = json.loads(Path("experiments/results/linear_dml_summary.json").read_text())
    dml_sample = dml_summary["samples"]["medium"]
    dml_candidate = dml_sample["selected_nuisance_candidate"]
    dml_effect = float(dml_sample["candidates"][dml_candidate]["ate"]["estimate"])

    scores = {
        "random": random_score,
        "response_model": naive["response_score"],
        "pseudo_uplift": naive["pseudo_uplift_score"],
        "psm_segments": psm_score,
        "linear_dml": np.full(len(evaluation), dml_effect),
        "causal_forest_dml": causal_forest["predicted_cate"],
        "uplift_tree": uplift["uplift_tree_score"],
        "uplift_random_forest": uplift["uplift_forest_score"],
    }
    started = time.perf_counter()
    top_k: dict[str, dict[str, object]] = {}
    masks: dict[str, dict[str, np.ndarray]] = {}
    for name, score in scores.items():
        top_k[name] = {}
        masks[name] = {}
        for budget in budgets:
            mask = ranked_policy_mask(score, budget, tie_breaker=random_score)
            metrics = evaluate_business_policy(
                evaluation,
                mask,
                email_cost=primary_cost,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
            metrics["profit_by_email_cost"] = {
                f"{cost:.2f}": _profit_at_cost(metrics, cost) for cost in email_costs
            }
            key = f"{budget:.2f}"
            top_k[name][key] = metrics
            masks[name][key] = mask

    primary_key = f"{primary_budget:.2f}"
    headline = {
        name: {"display_name": DISPLAY_NAMES[name], **results[primary_key]}
        for name, results in top_k.items()
    }
    paired_profit = {}
    for reference in ("random", "response_model"):
        paired_profit[reference] = {}
        for name in scores:
            if name == reference:
                continue
            paired_profit[reference][name] = bootstrap_profit_difference(
                evaluation,
                masks[name][primary_key],
                masks[reference][primary_key],
                email_cost=primary_cost,
                samples=bootstrap_samples,
                seed=seed,
            )

    failure_key = f"{failure_budget:.2f}"
    decision_failure_comparison = {}
    for name in scores:
        if name == "response_model":
            continue
        decision_failure_comparison[name] = {
            "conversion_uplift_difference_vs_response": bootstrap_policy_difference(
                evaluation,
                masks[name][failure_key],
                masks["response_model"][failure_key],
                samples=bootstrap_samples,
                seed=seed,
            ),
            "profit_difference_vs_response_per_1000_eligible": bootstrap_profit_difference(
                evaluation,
                masks[name][failure_key],
                masks["response_model"][failure_key],
                email_cost=primary_cost,
                samples=bootstrap_samples,
                seed=seed,
            ),
        }

    personalized_scores = {
        name: scores[name]
        for name in (
            "pseudo_uplift",
            "psm_segments",
            "linear_dml",
            "causal_forest_dml",
            "uplift_tree",
            "uplift_random_forest",
        )
    }
    cost_aware = {}
    for name, score in personalized_scores.items():
        cost_aware[name] = {}
        for cost in email_costs:
            selected = cost_aware_mask(
                score,
                conversion_value=float(policy_config["conversion_value"]),
                email_cost=cost,
            )
            cost_aware[name][f"{cost:.2f}"] = evaluate_business_policy(
                evaluation,
                selected,
                email_cost=cost,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )

    descriptive_best_budget = {}
    for name, results in top_k.items():
        descriptive_best_budget[name] = {}
        for cost in email_costs:
            cost_key = f"{cost:.2f}"
            winner_key = max(
                results,
                key=lambda key: results[key]["profit_by_email_cost"][cost_key]["estimate"],
            )
            descriptive_best_budget[name][cost_key] = {
                "budget": float(winner_key),
                **results[winner_key]["profit_by_email_cost"][cost_key],
            }

    figure_dir = Path("experiments/figures")
    curve_path = figure_dir / "policy_profit_curves.svg"
    interval_path = figure_dir / "policy_20pct_profit.svg"
    failure_path = figure_dir / "prediction_vs_uplift_policy.svg"
    plot_policy_profit_curves(
        top_k,
        curve_path,
        primary_cost=primary_cost,
        primary_budget=primary_budget,
    )
    plot_primary_policy_intervals(
        headline,
        interval_path,
        budget=primary_budget,
        email_cost=primary_cost,
    )
    response_failure_mask = masks["response_model"][failure_key]
    uplift_failure_mask = masks["uplift_tree"][failure_key]
    decision_overlap = {
        "both": int((response_failure_mask & uplift_failure_mask).sum()),
        "response_only": int((response_failure_mask & ~uplift_failure_mask).sum()),
        "uplift_only": int((uplift_failure_mask & ~response_failure_mask).sum()),
        "neither": int((~response_failure_mask & ~uplift_failure_mask).sum()),
        "jaccard": float(
            (response_failure_mask & uplift_failure_mask).sum()
            / (response_failure_mask | uplift_failure_mask).sum()
        ),
    }
    plot_prediction_vs_uplift_decision(
        scores["response_model"],
        scores["uplift_tree"],
        response_failure_mask,
        uplift_failure_mask,
        top_k["response_model"][failure_key],
        top_k["uplift_tree"][failure_key],
        decision_failure_comparison["uplift_tree"][
            "profit_difference_vs_response_per_1000_eligible"
        ],
        failure_path,
        budget=failure_budget,
    )
    summary = {
        "seed": seed,
        "evaluation_data": "untouched randomized holdout",
        "evaluation_rows": len(evaluation),
        "assumptions": {
            "incremental_revenue": "randomized difference in Hillstrom spend",
            "campaign_cost": "targeted customers multiplied by configurable email cost",
            "cost_aware_rule": "conversion_value * predicted_conversion_uplift > email_cost",
            "conversion_value": float(policy_config["conversion_value"]),
            "profit": "incremental spend minus campaign cost",
            "bootstrap_scope": "conditional on frozen scores and policy masks",
        },
        "settings": policy_config,
        "psm_segmentation": {
            "history_threshold_from_training": history_threshold,
            "recency_threshold_from_training": recency_threshold,
            "segments": psm_segments,
        },
        "linear_dml": {
            "selected_nuisance_candidate": dml_candidate,
            "constant_conversion_effect": dml_effect,
            "ranking_note": "constant effect; top-k ties use the same seeded order as random",
        },
        "headline_20_percent": headline,
        "paired_profit_differences_at_primary_setting": paired_profit,
        "decision_failure_comparison": {
            "budget": failure_budget,
            "email_cost": primary_cost,
            "selection_note": (
                "The 5% budget came from the prespecified budget grid but was selected for this "
                "diagnostic after inspecting the policy curves; intervals are not adjusted for "
                "that multiplicity."
            ),
            "response_policy": top_k["response_model"][failure_key],
            "alternatives_vs_response": decision_failure_comparison,
            "response_vs_uplift_tree_decisions": decision_overlap,
        },
        "top_k_policies": top_k,
        "cost_aware_policies": cost_aware,
        "descriptive_rct_best_budget": descriptive_best_budget,
        "best_budget_warning": (
            "These point-estimate maxima use the evaluation outcomes and are descriptive only; "
            "they are not deployment-selected budgets."
        ),
        "runtime_seconds": time.perf_counter() - started,
        "figures": {
            "profit_curves": str(curve_path),
            "primary_intervals": str(interval_path),
            "prediction_vs_uplift": str(failure_path),
        },
    }
    payload = json.dumps(summary, indent=2) + "\n"
    output_dir = Path("outputs/business_policies")
    result_path = Path("experiments/results/business_policy_summary.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(payload)
    result_path.write_text(payload)
    print(payload)
    print(f"wrote {result_path}")


if __name__ == "__main__":
    main()
