from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from causal_uplift.causal.baselines import ResponseModel, TreatmentFeatureModel
from causal_uplift.causal.causal_forest import fit_causal_forest
from causal_uplift.causal.dml import make_cross_fit_splits
from causal_uplift.causal.uplift import fit_uplift_forest
from causal_uplift.data.load import PRE_TREATMENT_COLUMNS
from causal_uplift.evaluation.ablation import summarize_ate_estimators
from causal_uplift.evaluation.ate import bootstrap_difference_in_means, difference_in_means
from causal_uplift.evaluation.business import (
    bootstrap_profit_difference,
    evaluate_business_policy,
    psm_segment_scores,
    ranked_policy_mask,
)
from causal_uplift.evaluation.policy import random_policy_scores
from causal_uplift.visualization.plots import (
    plot_confounding_ate_ablation,
    plot_confounding_policy_ablation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    seed = int(config["project"]["seed"])
    strengths = list(config["confounding"]["strengths"])
    ablation_config = config["confounding_ablation"]
    budget = float(ablation_config["policy_budget"])
    email_cost = float(ablation_config["email_cost"])
    bootstrap_samples = int(ablation_config["bootstrap_samples"])
    dml_config = config["dml"]
    forest_config = config["causal_forest"]
    uplift_config = config["uplift_models"]

    dml_summary = json.loads(Path("experiments/results/linear_dml_summary.json").read_text())
    matching_summary = json.loads(Path("experiments/results/matching_summary.json").read_text())
    if dml_summary["selection_uses_rct"]:
        raise AssertionError("ablation cannot use nuisance configurations selected on the RCT")

    score_columns = ["source_row_id", *PRE_TREATMENT_COLUMNS]
    scoring = pd.read_csv("data/processed/rct_evaluation.csv", usecols=score_columns).sort_values(
        "source_row_id"
    )
    score_output = pd.DataFrame({"source_row_id": scoring["source_row_id"]})
    fitted: dict[str, dict[str, object]] = {}
    fit_started = time.perf_counter()
    for strength in strengths:
        training = pd.read_csv(Path("data/observational") / f"{strength}.csv")
        nuisance_candidate = dml_summary["samples"][strength]["selected_nuisance_candidate"]
        splits = make_cross_fit_splits(
            training["conversion"],
            training["treatment"],
            folds=int(dml_config["cross_fit_folds"]),
            seed=seed,
        )
        strength_started = time.perf_counter()
        response = ResponseModel(seed=seed).fit(training)
        pseudo = TreatmentFeatureModel(seed=seed).fit(training)
        causal_forest = fit_causal_forest(
            training,
            "conversion",
            nuisance_candidate=nuisance_candidate,
            splits=splits,
            seed=seed,
            n_estimators=int(forest_config["n_estimators"]),
            min_samples_leaf=int(forest_config["min_samples_leaf"]),
            max_depth=int(forest_config["max_depth"]),
            max_samples=float(forest_config["max_samples"]),
            max_features=float(forest_config["max_features"]),
            subforest_size=int(forest_config["subforest_size"]),
            honest=bool(forest_config["honest"]),
            nuisance_random_forest_trees=int(dml_config["random_forest_trees"]),
            nuisance_max_iter=int(dml_config["max_iter"]),
        )
        uplift_forest = fit_uplift_forest(
            training,
            control_name=str(uplift_config["control_name"]),
            treatment_name=str(uplift_config["treatment_name"]),
            evaluation_function=str(uplift_config["evaluation_function"]),
            honesty=bool(uplift_config["honesty"]),
            estimation_sample_size=float(uplift_config["estimation_sample_size"]),
            seed=seed,
            **uplift_config["forest"],
        )

        preferred = matching_summary["samples"][strength]["preferred_variant"]
        pairs = pd.read_csv(Path("outputs/matching") / f"{strength}_{preferred}_pairs.csv")
        psm_score, psm_segments = psm_segment_scores(
            training,
            pairs,
            scoring,
            history_threshold=float(training["history"].median()),
            recency_threshold=float(training["recency"].median()),
        )
        response_score = response.predict(scoring)
        pseudo_score = pseudo.predict_uplift(scoring)
        causal_score = causal_forest.predict(scoring)
        uplift_score = uplift_forest.predict(scoring)
        causal_ate, causal_ate_lower, causal_ate_upper = causal_forest.ate_interval(scoring)
        selected_dml = dml_summary["samples"][strength]["selected_nuisance_candidate"]
        dml_ate = dml_summary["samples"][strength]["candidates"][selected_dml]["ate"]
        psm_att = matching_summary["samples"][strength]["variants"][preferred]["att"]["conversion"]
        fitted[strength] = {
            "training_rows": len(training),
            "nuisance_candidate": nuisance_candidate,
            "preferred_matching_variant": preferred,
            "naive_association": difference_in_means(training, "conversion"),
            "linear_dml_ate": dml_ate,
            "causal_forest_ate": {
                "estimate": causal_ate,
                "ci_lower": causal_ate_lower,
                "ci_upper": causal_ate_upper,
            },
            "uplift_forest_mean_effect": float(uplift_score.mean()),
            "psm_att": psm_att,
            "psm_segments": psm_segments,
            "scores": {
                "response_model": response_score,
                "pseudo_uplift": pseudo_score,
                "psm_segments": psm_score,
                "causal_forest_dml": causal_score,
                "uplift_random_forest": uplift_score,
            },
            "fit_and_score_runtime_seconds": time.perf_counter() - strength_started,
        }
        for name, values in fitted[strength]["scores"].items():
            score_output[f"{strength}_{name}"] = values
        del response, pseudo, causal_forest, uplift_forest
        gc.collect()

    # Outcomes are opened only after every specification is fitted and every score is frozen.
    evaluation = pd.read_csv("data/processed/rct_evaluation.csv").sort_values("source_row_id")
    if not np.array_equal(evaluation["source_row_id"], scoring["source_row_id"]):
        raise AssertionError("RCT covariates and outcomes do not align")
    rct_effect = bootstrap_difference_in_means(
        evaluation,
        "conversion",
        samples=bootstrap_samples,
        seed=seed,
    )
    random_score = random_policy_scores(evaluation, seed=seed)
    random_mask = ranked_policy_mask(random_score, budget, tie_breaker=random_score)
    random_policy = evaluate_business_policy(
        evaluation,
        random_mask,
        email_cost=email_cost,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    samples: dict[str, dict[str, object]] = {}
    for strength in strengths:
        result = fitted[strength]
        result["ate_estimators"] = summarize_ate_estimators(
            rct_ate=rct_effect["estimate"],
            naive=result["naive_association"],
            linear_dml=result["linear_dml_ate"]["estimate"],
            causal_forest=result["causal_forest_ate"]["estimate"],
            uplift_forest=result["uplift_forest_mean_effect"],
            psm_att=result["psm_att"]["estimate"],
        )
        policies = {"random": random_policy}
        paired_vs_random = {}
        for name, score in result.pop("scores").items():
            mask = ranked_policy_mask(score, budget, tie_breaker=random_score)
            policies[name] = evaluate_business_policy(
                evaluation,
                mask,
                email_cost=email_cost,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
            paired_vs_random[name] = bootstrap_profit_difference(
                evaluation,
                mask,
                random_mask,
                email_cost=email_cost,
                samples=bootstrap_samples,
                seed=seed,
            )
        result["policies"] = policies
        result["paired_profit_difference_vs_random"] = paired_vs_random
        samples[strength] = result

    figure_dir = Path("experiments/figures")
    ate_path = figure_dir / "confounding_ate_ablation.svg"
    policy_path = figure_dir / "confounding_policy_ablation.svg"
    plot_confounding_ate_ablation(samples, ate_path)
    plot_confounding_policy_ablation(
        samples,
        policy_path,
        budget=budget,
        email_cost=email_cost,
    )
    summary = {
        "seed": seed,
        "strengths": strengths,
        "rct_effect": rct_effect,
        "policy_setting": {"budget": budget, "email_cost": email_cost},
        "model_selection_uses_rct": False,
        "rct_outcomes_loaded_after_all_scores_frozen": True,
        "psm_estimand_warning": (
            "PSM estimates ATT in matched observational treated customers; its gap from the "
            "overall RCT ATE is a reference gap, not a like-for-like ATE error."
        ),
        "linear_dml_policy_warning": (
            "LinearDML is constant-effect in this project and therefore has no personalized "
            "ranking; random policy is its honest top-k equivalent."
        ),
        "samples": samples,
        "runtime_seconds": time.perf_counter() - fit_started,
        "figures": {"ate_error": str(ate_path), "policy_value": str(policy_path)},
    }
    payload = json.dumps(summary, indent=2) + "\n"
    output_dir = Path("outputs/confounding_ablation")
    result_path = Path("experiments/results/confounding_ablation_summary.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    score_output.to_csv(output_dir / "rct_scores.csv", index=False)
    (output_dir / "summary.json").write_text(payload)
    result_path.write_text(payload)
    print(payload)
    print(f"wrote {result_path}")


if __name__ == "__main__":
    main()
