from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from causal_uplift.causal.causal_forest import (
    customer_segments,
    effect_bins,
    fit_causal_forest,
)
from causal_uplift.causal.dml import make_cross_fit_splits
from causal_uplift.evaluation.ate import bootstrap_difference_in_means
from causal_uplift.evaluation.heterogeneity import randomized_group_effects
from causal_uplift.evaluation.policy import bootstrap_policy_difference
from causal_uplift.visualization.plots import plot_causal_forest_validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    seed = int(config["project"]["seed"])
    forest_config = config["causal_forest"]
    dml_config = config["dml"]
    training_sample = str(forest_config["training_sample"])

    dml_summary = json.loads(Path("experiments/results/linear_dml_summary.json").read_text())
    if dml_summary["selection_uses_rct"]:
        raise AssertionError("causal forest cannot reuse nuisance selection tuned on the RCT")
    nuisance_candidate = dml_summary["samples"][training_sample]["selected_nuisance_candidate"]
    training = pd.read_csv(Path("data/observational") / f"{training_sample}.csv")
    splits = make_cross_fit_splits(
        training["conversion"],
        training["treatment"],
        folds=int(dml_config["cross_fit_folds"]),
        seed=seed,
    )
    history_threshold = float(training["history"].median())
    recency_threshold = float(training["recency"].median())
    started = time.perf_counter()
    forest = fit_causal_forest(
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
    fit_runtime = time.perf_counter() - started

    # Randomized outcomes are opened only after the estimator and segment rules are frozen.
    evaluation = pd.read_csv("data/processed/rct_evaluation.csv")
    score_started = time.perf_counter()
    cate = forest.predict(evaluation)
    cate_lower, cate_upper = forest.predict_interval(evaluation)
    score_runtime = time.perf_counter() - score_started
    ate, ate_lower, ate_upper = forest.ate_interval(evaluation)
    rct_effect = bootstrap_difference_in_means(
        evaluation,
        "conversion",
        samples=int(config["evaluation"]["bootstrap_samples"]),
        seed=seed,
    )
    bins = effect_bins(cate, bins=int(forest_config["cate_bins"]))
    deciles = randomized_group_effects(
        evaluation,
        bins,
        cate,
        bootstrap_samples=int(config["evaluation"]["bootstrap_samples"]),
        seed=seed,
    )
    deciles.sort(key=lambda row: int(row["group"]))
    segments = customer_segments(
        evaluation,
        history_threshold=history_threshold,
        recency_threshold=recency_threshold,
    )
    subgroups = randomized_group_effects(
        evaluation,
        segments,
        cate,
        bootstrap_samples=int(config["evaluation"]["bootstrap_samples"]),
        seed=seed,
    )
    segment_order = {
        "recent / high history": 0,
        "recent / low history": 1,
        "inactive / high history": 2,
        "inactive / low history": 3,
    }
    subgroups.sort(key=lambda row: segment_order[row["group"]])
    top = bins == bins.max()
    bottom = bins == bins.min()
    top_minus_bottom = bootstrap_policy_difference(
        evaluation,
        top,
        bottom,
        samples=int(config["evaluation"]["bootstrap_samples"]),
        seed=seed,
    )
    rank_correlation = spearmanr(
        [row["predicted_cate_mean"] for row in deciles],
        [row["observed_rct_uplift"] for row in deciles],
    )

    figure_path = Path("experiments/figures/causal_forest_validation.svg")
    plot_causal_forest_validation(deciles, subgroups, figure_path)
    summary = {
        "seed": seed,
        "outcome": "conversion",
        "training_sample": training_sample,
        "training_rows": len(training),
        "evaluation_rows": len(evaluation),
        "rct_used_for_model_or_hyperparameter_selection": False,
        "nuisance_candidate": nuisance_candidate,
        "nuisance_selection_source": "versioned T05 observational cross-fit result",
        "forest_parameters": forest_config,
        "fit_runtime_seconds": fit_runtime,
        "score_and_interval_runtime_seconds": score_runtime,
        "ate_on_rct_covariate_distribution": {
            "estimate": ate,
            "ci_lower": ate_lower,
            "ci_upper": ate_upper,
            "absolute_error_vs_rct": abs(ate - rct_effect["estimate"]),
        },
        "overall_rct_effect": rct_effect,
        "cate_distribution": {
            "mean": float(cate.mean()),
            "standard_deviation": float(cate.std()),
            "minimum": float(cate.min()),
            "maximum": float(cate.max()),
            "negative_fraction": float((cate < 0).mean()),
            "mean_pointwise_interval_width": float(np.mean(cate_upper - cate_lower)),
        },
        "cate_deciles": deciles,
        "decile_spearman": {
            "correlation": float(rank_correlation.statistic),
            "p_value": float(rank_correlation.pvalue),
            "note": "descriptive only; ten noisy aggregate bins are not a model-selection set",
        },
        "top_minus_bottom_decile": top_minus_bottom,
        "segment_thresholds_from_training": {
            "history_median": history_threshold,
            "recency_median": recency_threshold,
        },
        "subgroups": subgroups,
        "aggregated_feature_importance": forest.aggregated_feature_importance(),
        "interpretation_warning": (
            "Feature importance and CATE association do not identify causal mechanisms."
        ),
        "figure": str(figure_path),
    }
    payload = json.dumps(summary, indent=2) + "\n"
    output_dir = Path("outputs/causal_forest")
    result_path = Path("experiments/results/causal_forest_summary.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "source_row_id": evaluation["source_row_id"],
            "treatment": evaluation["treatment"],
            "conversion": evaluation["conversion"],
            "predicted_cate": cate,
            "cate_ci_lower": cate_lower,
            "cate_ci_upper": cate_upper,
            "cate_decile": bins,
            "customer_segment": segments,
        }
    ).to_csv(output_dir / "rct_cate_scores.csv", index=False)
    (output_dir / "summary.json").write_text(payload)
    result_path.write_text(payload)
    print(payload)
    print(f"wrote {result_path}")


if __name__ == "__main__":
    main()
