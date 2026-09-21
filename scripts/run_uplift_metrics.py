from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from causal_uplift.evaluation.uplift_metrics import (
    bootstrap_ranking_metrics,
)
from causal_uplift.visualization.plots import (
    plot_uplift_metric_intervals,
    plot_uplift_qini_curves,
)

SCORE_SOURCES = {
    "response_model": ("outputs/naive_baselines/rct_policy_scores.csv", "response_score"),
    "pseudo_uplift": (
        "outputs/naive_baselines/rct_policy_scores.csv",
        "pseudo_uplift_score",
    ),
    "causal_forest_dml": ("outputs/causal_forest/rct_cate_scores.csv", "predicted_cate"),
    "uplift_tree": ("outputs/uplift_models/rct_uplift_scores.csv", "uplift_tree_score"),
    "uplift_random_forest": (
        "outputs/uplift_models/rct_uplift_scores.csv",
        "uplift_forest_score",
    ),
}


def _load_scores(evaluation: pd.DataFrame) -> dict[str, np.ndarray]:
    scores = {}
    expected_ids = evaluation["source_row_id"].to_numpy()
    for name, (path, column) in SCORE_SOURCES.items():
        scored = pd.read_csv(path).sort_values("source_row_id")
        if not np.array_equal(scored["source_row_id"].to_numpy(), expected_ids):
            raise AssertionError(f"{name} scores do not align with the frozen RCT holdout")
        if "treatment" in scored and not np.array_equal(
            scored["treatment"].to_numpy(), evaluation["treatment"].to_numpy()
        ):
            raise AssertionError(f"{name} treatment labels changed")
        if "conversion" in scored and not np.array_equal(
            scored["conversion"].to_numpy(), evaluation["conversion"].to_numpy()
        ):
            raise AssertionError(f"{name} outcomes changed")
        scores[name] = scored[column].to_numpy(dtype=float)
    return scores


def _add_per_1000(summary: dict[str, object]) -> None:
    for result in summary["models"].values():
        for metric in ("auuc", "qini"):
            values = result[metric]
            values["estimate_per_1000"] = values["estimate"] * 1000
            values["ci_lower_per_1000"] = values["ci_lower"] * 1000
            values["ci_upper_per_1000"] = values["ci_upper"] * 1000
    paired = summary["paired_qini_difference_vs_reference"]["models"]
    for values in paired.values():
        values["estimate_per_1000"] = values["estimate"] * 1000
        values["ci_lower_per_1000"] = values["ci_lower"] * 1000
        values["ci_upper_per_1000"] = values["ci_upper"] * 1000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    metric_config = config["uplift_metrics"]
    seed = int(config["project"]["seed"])
    evaluation = pd.read_csv("data/processed/rct_evaluation.csv").sort_values("source_row_id")
    scores = _load_scores(evaluation)
    fractions = np.asarray(metric_config["fractions"], dtype=float)

    started = time.perf_counter()
    result = bootstrap_ranking_metrics(
        evaluation,
        scores,
        fractions,
        samples=int(metric_config["bootstrap_samples"]),
        seed=seed,
        reference=str(metric_config["reference_model"]),
    )
    runtime = time.perf_counter() - started
    _add_per_1000(result)
    summary = {
        "seed": seed,
        "outcome": "conversion",
        "evaluation_data": "untouched randomized holdout",
        "evaluation_rows": len(evaluation),
        "bootstrap": {
            "samples": int(metric_config["bootstrap_samples"]),
            "method": "treatment-stratified paired resampling",
            "confidence_level": 0.95,
            "runtime_seconds": runtime,
        },
        "definitions": {
            "gain": (
                "targeted fraction multiplied by its randomized treatment-control conversion "
                "difference"
            ),
            "auuc": "trapezoidal area under the cumulative gain curve",
            "qini": "area between cumulative gain and the random-targeting line",
            "tie_handling": (
                "all rows tied at a targeting boundary receive the same fractional selection weight"
            ),
            "uncertainty_scope": (
                "conditional on the fitted scores; bootstrap resamples the randomized evaluation "
                "rows within treatment arms"
            ),
        },
        **result,
    }
    figure_dir = Path("experiments/figures")
    curve_path = figure_dir / "uplift_qini_curves.svg"
    interval_path = figure_dir / "uplift_metric_intervals.svg"
    plot_uplift_qini_curves(summary["models"], curve_path)
    plot_uplift_metric_intervals(summary["models"], interval_path)
    summary["figures"] = {"curves": str(curve_path), "intervals": str(interval_path)}

    payload = json.dumps(summary, indent=2) + "\n"
    output_dir = Path("outputs/uplift_metrics")
    result_path = Path("experiments/results/uplift_metrics_summary.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(payload)
    result_path.write_text(payload)
    print(payload)
    print(f"wrote {result_path}")


if __name__ == "__main__":
    main()
