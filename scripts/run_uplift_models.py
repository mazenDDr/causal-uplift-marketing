from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from causal_uplift.causal.causal_forest import effect_bins
from causal_uplift.causal.uplift import (
    fit_uplift_forest,
    fit_uplift_tree,
    uplift_tree_leaves,
)
from causal_uplift.evaluation.heterogeneity import randomized_group_effects
from causal_uplift.evaluation.policy import bootstrap_policy_difference
from causal_uplift.visualization.plots import (
    plot_uplift_model_validation,
    plot_uplift_tree_structure,
)


def _evaluate_ranking(
    evaluation: pd.DataFrame,
    score: np.ndarray,
    *,
    bins: int,
    bootstrap_samples: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    labels = effect_bins(score, bins=bins)
    deciles = randomized_group_effects(
        evaluation,
        labels,
        score,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    deciles.sort(key=lambda row: int(row["group"]))
    correlation = spearmanr(
        [row["predicted_cate_mean"] for row in deciles],
        [row["observed_rct_uplift"] for row in deciles],
    )
    top_minus_bottom = bootstrap_policy_difference(
        evaluation,
        labels == labels.max(),
        labels == labels.min(),
        samples=bootstrap_samples,
        seed=seed,
    )
    return labels, {
        "score_summary": {
            "mean": float(score.mean()),
            "standard_deviation": float(score.std()),
            "minimum": float(score.min()),
            "maximum": float(score.max()),
            "negative_fraction": float((score < 0).mean()),
            "unique_values": int(np.unique(np.round(score, 12)).size),
        },
        "deciles": deciles,
        "decile_spearman": {
            "correlation": float(correlation.statistic),
            "p_value": float(correlation.pvalue),
            "note": "descriptive only; randomized outcomes were not used for model selection",
        },
        "top_minus_bottom_decile": top_minus_bottom,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    seed = int(config["project"]["seed"])
    uplift_config = config["uplift_models"]
    training_sample = str(uplift_config["training_sample"])
    training = pd.read_csv(Path("data/observational") / f"{training_sample}.csv")
    common = {
        "control_name": str(uplift_config["control_name"]),
        "treatment_name": str(uplift_config["treatment_name"]),
        "evaluation_function": str(uplift_config["evaluation_function"]),
        "honesty": bool(uplift_config["honesty"]),
        "estimation_sample_size": float(uplift_config["estimation_sample_size"]),
        "seed": seed,
    }

    tree_started = time.perf_counter()
    tree = fit_uplift_tree(training, **common, **uplift_config["tree"])
    tree_runtime = time.perf_counter() - tree_started
    forest_started = time.perf_counter()
    forest = fit_uplift_forest(training, **common, **uplift_config["forest"])
    forest_runtime = time.perf_counter() - forest_started

    figure_dir = Path("experiments/figures")
    tree_figure_path = figure_dir / "uplift_tree.svg"
    plot_uplift_tree_structure(
        tree.model.fitted_uplift_tree,
        tree.feature_names,
        tree_figure_path,
        control_name=common["control_name"],
        treatment_name=common["treatment_name"],
    )
    leaves = uplift_tree_leaves(
        tree,
        control_name=common["control_name"],
        treatment_name=common["treatment_name"],
    )

    # Randomized outcomes are opened only after both model specifications are fitted and frozen.
    evaluation = pd.read_csv("data/processed/rct_evaluation.csv")
    tree_score = tree.predict(evaluation)
    forest_score = forest.predict(evaluation)
    bootstrap_samples = int(config["evaluation"]["bootstrap_samples"])
    bins = int(uplift_config["cate_bins"])
    tree_bins, tree_result = _evaluate_ranking(
        evaluation,
        tree_score,
        bins=bins,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    tree_result["ranking_warning"] = (
        "The tree produces one score per leaf. Exact deciles therefore split tied leaf scores; "
        "interpret its leaf estimates and randomized intervals alongside the decile chart."
    )
    forest_bins, forest_result = _evaluate_ranking(
        evaluation,
        forest_score,
        bins=bins,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    forest_minus_tree_top_decile = bootstrap_policy_difference(
        evaluation,
        forest_bins == forest_bins.max(),
        tree_bins == tree_bins.max(),
        samples=bootstrap_samples,
        seed=seed,
    )
    decision_overlap = {
        "top_decile_overlap_rows": int(
            ((forest_bins == forest_bins.max()) & (tree_bins == tree_bins.max())).sum()
        ),
        "top_decile_jaccard": float(
            ((forest_bins == forest_bins.max()) & (tree_bins == tree_bins.max())).sum()
            / ((forest_bins == forest_bins.max()) | (tree_bins == tree_bins.max())).sum()
        ),
        "score_spearman": float(spearmanr(tree_score, forest_score).statistic),
    }
    validation_figure_path = figure_dir / "uplift_model_validation.svg"
    plot_uplift_model_validation(
        {
            "uplift_tree": tree_result["deciles"],
            "uplift_random_forest": forest_result["deciles"],
        },
        validation_figure_path,
    )

    summary = {
        "seed": seed,
        "outcome": "conversion",
        "training_sample": training_sample,
        "training_rows": len(training),
        "evaluation_rows": len(evaluation),
        "rct_used_for_model_or_hyperparameter_selection": False,
        "propensity_adjustment": False,
        "propensity_warning": (
            "These uplift trees optimize treatment-effect splits but do not correct observational "
            "assignment with an explicit propensity model."
        ),
        "package": {"causalml": importlib.metadata.version("causalml")},
        "parameters": uplift_config,
        "models": {
            "uplift_tree": {
                "fit_runtime_seconds": tree_runtime,
                "leaf_count": len(leaves),
                "leaves": leaves,
                **tree_result,
            },
            "uplift_random_forest": {
                "fit_runtime_seconds": forest_runtime,
                **forest_result,
            },
        },
        "forest_minus_tree_top_decile": forest_minus_tree_top_decile,
        "model_agreement": decision_overlap,
        "figures": {
            "tree": str(tree_figure_path),
            "randomized_validation": str(validation_figure_path),
        },
    }
    payload = json.dumps(summary, indent=2) + "\n"
    output_dir = Path("outputs/uplift_models")
    result_path = Path("experiments/results/uplift_models_summary.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "source_row_id": evaluation["source_row_id"],
            "treatment": evaluation["treatment"],
            "conversion": evaluation["conversion"],
            "uplift_tree_score": tree_score,
            "uplift_tree_decile": tree_bins,
            "uplift_forest_score": forest_score,
            "uplift_forest_decile": forest_bins,
        }
    ).to_csv(output_dir / "rct_uplift_scores.csv", index=False)
    (output_dir / "summary.json").write_text(payload)
    result_path.write_text(payload)
    print(payload)
    print(f"wrote {result_path}")


if __name__ == "__main__":
    main()
