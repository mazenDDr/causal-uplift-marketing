from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from causal_uplift.evaluation.failures import ratio, validate_failure_cases
from causal_uplift.visualization.plots import plot_failure_taxonomy

RESULTS = Path("experiments/results")
FIGURES = Path("experiments/figures")


def load_json(name: str) -> dict[str, Any]:
    return json.loads((RESULTS / name).read_text())


def main() -> None:
    confounding = load_json("confounding_ablation_summary.json")
    overlap = load_json("overlap_stress_summary.json")
    uplift_models = load_json("uplift_models_summary.json")
    uplift_metrics = load_json("uplift_metrics_summary.json")

    rct_ate = confounding["rct_effect"]["estimate"]
    strong = confounding["samples"]["strong"]
    severe = overlap["aggregate"]["severe"]
    healthy = overlap["aggregate"]["healthy"]
    forest = uplift_models["models"]["uplift_random_forest"]
    forest_qini = uplift_metrics["models"]["uplift_random_forest"]["qini"]

    naive = strong["ate_estimators"]["naive"]
    predicted_decile_spread = (
        forest["deciles"][-1]["predicted_cate_mean"] - forest["deciles"][0]["predicted_cate_mean"]
    )
    cases = [
        {
            "id": "naive_confounding",
            "method": "Naive association",
            "failure": "Selection bias is mistaken for campaign lift",
            "symptom": (
                "The strong-confounding estimate overstates the randomized conversion ATE by "
                f"{ratio(naive['estimate'] - rct_ate, rct_ate):.1%}."
            ),
            "cause": (
                "Historically preferred customers have higher baseline purchase propensity, so "
                "the treated/control difference mixes treatment effect with pre-existing demand."
            ),
            "diagnostic": (
                "Track covariate imbalance and compare observational estimates with a held-out "
                "randomized benchmark as selection strength increases."
            ),
            "fix": (
                "Adjust for observed pre-treatment confounders, preserve overlap, and retain an "
                "experimental benchmark when one is available."
            ),
            "estimand": "ATE on the RCT covariate distribution",
            "evidence": {
                "rct_ate": rct_ate,
                "strong_confounding_naive_estimate": naive["estimate"],
                "ate_error": naive["absolute_error_vs_rct_ate"],
                "relative_overestimate": ratio(naive["estimate"] - rct_ate, rct_ate),
                "linear_dml_ate_error": strong["ate_estimators"]["linear_dml"][
                    "absolute_error_vs_rct_ate"
                ],
            },
            "source_artifacts": ["confounding_ablation_summary.json"],
        },
        {
            "id": "psm_overlap",
            "method": "Propensity-score matching",
            "failure": "Matching runs after the comparable population collapses",
            "symptom": (
                "Severe overlap stress retains a mean 1,987 matched pairs, discards 8,569 rows, "
                "and passes the balance rule in only 2 of 5 resamples."
            ),
            "cause": (
                "When propensities approach zero or one, nearby controls do not exist for many "
                "treated customers; matching must discard them or accept poor matches."
            ),
            "diagnostic": (
                "Inspect propensity support, retained treated fraction, discarded rows, and "
                "post-match standardized mean differences before estimating ATT."
            ),
            "fix": (
                "Restrict the estimand to supported customers, tighten design diagnostics, and "
                "collect or randomize additional comparable examples rather than extrapolating."
            ),
            "estimand": "ATT in retained matched treated customers",
            "evidence": {
                "outside_overlap_fraction": severe["desired_outside_overlap_fraction_mean"],
                "matched_pairs_mean": severe["matching"]["matched_pairs_mean"],
                "matched_pairs_healthy_mean": healthy["matching"]["matched_pairs_mean"],
                "discarded_rows_mean": severe["matching"]["discarded_rows_outside_overlap_mean"],
                "balance_pass_rate": severe["matching"]["balance_pass_rate"],
                "max_absolute_smd_mean": severe["matching"]["max_absolute_smd_mean"],
            },
            "source_artifacts": ["overlap_stress_summary.json"],
        },
        {
            "id": "dml_overlap",
            "method": "LinearDML",
            "failure": "Orthogonalization becomes unstable without treatment variation",
            "symptom": (
                "From healthy to severe overlap, mean interval width doubles and only 3 of 5 "
                "severe intervals cover the randomized point estimate."
            ),
            "cause": (
                "Residualized treatment contains little usable variation in unsupported regions; "
                "flexible nuisance models cannot create missing counterfactual comparisons."
            ),
            "diagnostic": (
                "Monitor propensity extremes, effective sample size, interval width, repeated-seed "
                "dispersion, and randomized-benchmark coverage."
            ),
            "fix": (
                "Trim or redefine the target population, simplify unsupported heterogeneity "
                "claims, add data with overlap, or run a randomized intervention."
            ),
            "estimand": "ATE on the RCT covariate distribution",
            "evidence": {
                "healthy_ci_width": healthy["linear_dml"]["analytic_ci_width_mean"],
                "severe_ci_width": severe["linear_dml"]["analytic_ci_width_mean"],
                "ci_width_ratio": ratio(
                    severe["linear_dml"]["analytic_ci_width_mean"],
                    healthy["linear_dml"]["analytic_ci_width_mean"],
                ),
                "healthy_replicate_sd": healthy["linear_dml"]["replicate_sd"],
                "severe_replicate_sd": severe["linear_dml"]["replicate_sd"],
                "severe_rct_coverage_rate": severe["linear_dml"]["rct_ate_coverage_rate"],
                "fitted_effective_sample_fraction": severe["fitted_effective_sample_fraction_mean"],
            },
            "source_artifacts": ["overlap_stress_summary.json"],
        },
        {
            "id": "uplift_forest_ranking",
            "method": "Uplift random forest",
            "failure": "A granular observational ranking does not generalize to the RCT",
            "symptom": (
                "The model predicts a positive top-minus-bottom decile spread, but the observed "
                "randomized difference is negative and its interval includes zero."
            ),
            "cause": (
                "Effect-difference splits can fit rare-outcome noise and observational selection; "
                "the uplift objective is not itself propensity adjustment."
            ),
            "diagnostic": (
                "Freeze scores, then check RCT deciles, top-minus-bottom uplift, Qini, "
                "and bootstrap intervals instead of classification accuracy or score granularity."
            ),
            "fix": (
                "Regularize the forest, consider propensity-aware estimators, gather stronger "
                "experimental signal, and do not deploy a ranking that fails held-out validation."
            ),
            "estimand": "CATE ranking on the randomized holdout",
            "evidence": {
                "unique_score_values": forest["score_summary"]["unique_values"],
                "predicted_top_minus_bottom": predicted_decile_spread,
                "observed_top_minus_bottom": forest["top_minus_bottom_decile"]["uplift_difference"],
                "observed_ci_lower": forest["top_minus_bottom_decile"]["ci_lower"],
                "observed_ci_upper": forest["top_minus_bottom_decile"]["ci_upper"],
                "decile_spearman": forest["decile_spearman"]["correlation"],
                "qini_per_1000": forest_qini["estimate_per_1000"],
                "qini_ci_per_1000": [
                    forest_qini["ci_lower_per_1000"],
                    forest_qini["ci_upper_per_1000"],
                ],
            },
            "interpretation_limit": (
                "This is evidence of failed held-out ranking and overfit-like behavior, not a "
                "training-versus-test Qini gap because training Qini was intentionally not used."
            ),
            "source_artifacts": [
                "uplift_models_summary.json",
                "uplift_metrics_summary.json",
            ],
        },
    ]
    validate_failure_cases(cases)

    figure_path = FIGURES / "failure_taxonomy.svg"
    plot_failure_taxonomy(cases, figure_path)
    summary = {
        "purpose": (
            "Measured failure cases; each claim is derived from an earlier frozen experiment."
        ),
        "rct_outcomes_reused_for_model_selection": False,
        "cases": cases,
        "figures": {"taxonomy": str(figure_path)},
        "source_artifacts": sorted(
            {source for case in cases for source in case["source_artifacts"]}
        ),
    }
    output = RESULTS / "failure_analysis_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {output}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
