from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import pandas as pd
import yaml

from causal_uplift.causal.dml import (
    fit_linear_dml,
    make_cross_fit_splits,
    nuisance_cross_fit_metrics,
)
from causal_uplift.evaluation.ate import (
    bootstrap_difference_in_means,
    difference_in_means,
)
from causal_uplift.visualization.plots import plot_ate_error_by_strength


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    seed = int(config["project"]["seed"])
    dml_config = config["dml"]
    candidates = list(dml_config["nuisance_candidates"])
    fit_arguments = {
        "seed": seed,
        "random_forest_trees": int(dml_config["random_forest_trees"]),
        "min_samples_leaf": int(dml_config["min_samples_leaf"]),
        "max_iter": int(dml_config["max_iter"]),
    }

    samples: dict[str, dict[str, object]] = {}
    for strength in config["confounding"]["strengths"]:
        frame = pd.read_csv(Path("data/observational") / f"{strength}.csv")
        splits = make_cross_fit_splits(
            frame["conversion"],
            frame["treatment"],
            folds=int(dml_config["cross_fit_folds"]),
            seed=seed,
        )
        candidate_results: dict[str, dict[str, object]] = {}
        for candidate in candidates:
            nuisance_started = time.perf_counter()
            nuisance = nuisance_cross_fit_metrics(
                frame,
                "conversion",
                candidate=candidate,
                splits=splits,
                **fit_arguments,
            )
            nuisance_runtime = time.perf_counter() - nuisance_started
            dml_started = time.perf_counter()
            estimate = fit_linear_dml(
                frame,
                "conversion",
                candidate=candidate,
                splits=splits,
                **fit_arguments,
            )
            dml_runtime = time.perf_counter() - dml_started
            candidate_results[candidate] = {
                "nuisance_cross_fit_metrics": nuisance,
                "nuisance_validation_runtime_seconds": nuisance_runtime,
                "dml_fit_runtime_seconds": dml_runtime,
                "ate": {
                    "estimate": estimate.estimate,
                    "ci_lower": estimate.ci_lower,
                    "ci_upper": estimate.ci_upper,
                },
            }
        selected = min(
            candidates,
            key=lambda name: candidate_results[name]["nuisance_cross_fit_metrics"][
                "selection_loss"
            ],
        )
        estimates = [candidate_results[name]["ate"]["estimate"] for name in candidates]
        samples[strength] = {
            "rows": len(frame),
            "cross_fit_folds": len(splits),
            "every_row_scored_out_of_fold_once": (
                sum(len(test) for _, test in splits) == len(frame)
                and len(set().union(*(set(test) for _, test in splits))) == len(frame)
            ),
            "selected_nuisance_candidate": selected,
            "selection_rule": "lowest sum of outcome and treatment Brier loss relative to null",
            "selection_complete_before_rct_loaded": True,
            "nuisance_configuration_ate_range": max(estimates) - min(estimates),
            "naive_association": difference_in_means(frame, "conversion"),
            "candidates": candidate_results,
        }

    # Open randomized outcomes only after nuisance selection finishes for every sample.
    evaluation = pd.read_csv("data/processed/rct_evaluation.csv")
    rct_effect = bootstrap_difference_in_means(
        evaluation,
        "conversion",
        samples=int(config["evaluation"]["bootstrap_samples"]),
        seed=seed,
    )
    rct_ate = rct_effect["estimate"]
    errors: dict[str, dict[str, float]] = {
        "naive": {},
        **{candidate: {} for candidate in candidates},
    }
    for strength, sample in samples.items():
        sample["naive_absolute_error_vs_rct"] = abs(sample["naive_association"] - rct_ate)
        errors["naive"][strength] = sample["naive_absolute_error_vs_rct"]
        for candidate, result in sample["candidates"].items():
            error = abs(result["ate"]["estimate"] - rct_ate)
            result["ate"]["absolute_error_vs_rct"] = error
            errors[candidate][strength] = error
        selected = sample["selected_nuisance_candidate"]
        sample["selected_dml_absolute_error_vs_rct"] = sample["candidates"][selected]["ate"][
            "absolute_error_vs_rct"
        ]

    figure_path = Path("experiments/figures/dml_ate_error.svg")
    plot_ate_error_by_strength(errors, figure_path)
    summary = {
        "seed": seed,
        "outcome": "conversion",
        "method": "LinearDML with constant final treatment effect",
        "cross_fitting": {
            "folds": int(dml_config["cross_fit_folds"]),
            "stratification": "joint treatment/outcome state",
            "same_observation_used_for_nuisance_fit_and_residual": False,
        },
        "selection_uses_rct": False,
        "rct_effect": rct_effect,
        "packages": {
            package: importlib.metadata.version(package)
            for package in ("econml", "numpy", "scikit-learn")
        },
        "samples": samples,
        "absolute_ate_errors": errors,
        "figure": str(figure_path),
    }
    payload = json.dumps(summary, indent=2) + "\n"
    output_dir = Path("outputs/dml")
    result_path = Path("experiments/results/linear_dml_summary.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(payload)
    result_path.write_text(payload)
    print(payload)
    print(f"wrote {result_path}")


if __name__ == "__main__":
    main()
