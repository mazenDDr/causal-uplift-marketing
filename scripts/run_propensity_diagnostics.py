from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import yaml

from causal_uplift.causal.propensity import (
    LogisticPropensityModel,
    propensity_diagnostics,
)
from causal_uplift.data.load import PRE_TREATMENT_COLUMNS
from causal_uplift.evaluation.balance import covariate_smds
from causal_uplift.visualization.plots import (
    plot_love_by_strength,
    plot_propensity_overlap,
)

BALANCE_COLUMNS = (
    "history",
    "recency",
    "mens",
    "womens",
    "newbie",
    "channel",
    "zip_code",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    seed = int(config["project"]["seed"])
    propensity_config = config["propensity"]

    frames: dict[str, pd.DataFrame] = {}
    scores = {}
    diagnostics = {}
    smds = {}
    score_output = Path("outputs/propensity")
    score_output.mkdir(parents=True, exist_ok=True)
    for name in config["confounding"]["strengths"]:
        frame = pd.read_csv(Path("data/observational") / f"{name}.csv")
        started = time.perf_counter()
        model = LogisticPropensityModel(
            seed=seed,
            max_iter=int(propensity_config["max_iter"]),
        ).fit(frame)
        fit_seconds = time.perf_counter() - started
        score = model.predict(frame)
        sample_diagnostics = propensity_diagnostics(
            frame,
            score,
            overlap_min=float(propensity_config["overlap_min"]),
            overlap_max=float(propensity_config["overlap_max"]),
        )
        sample_smds = covariate_smds(frame, BALANCE_COLUMNS)
        sample_diagnostics["rows"] = len(frame)
        sample_diagnostics["treated_share"] = float(frame["treatment"].mean())
        sample_diagnostics["fit_runtime_seconds"] = fit_seconds
        sample_diagnostics["max_absolute_smd"] = max(abs(value) for value in sample_smds.values())
        sample_diagnostics["covariates_above_0_10_absolute_smd"] = sum(
            abs(value) >= 0.10 for value in sample_smds.values()
        )
        frames[name] = frame
        scores[name] = score
        diagnostics[name] = sample_diagnostics
        smds[name] = sample_smds
        pd.DataFrame(
            {
                "source_row_id": frame["source_row_id"],
                "treatment": frame["treatment"],
                "estimated_propensity": score,
            }
        ).to_csv(score_output / f"{name}_scores.csv", index=False)

    figure_dir = Path("experiments/figures")
    plot_propensity_overlap(
        frames,
        scores,
        figure_dir / "propensity_overlap.svg",
        overlap_min=float(propensity_config["overlap_min"]),
        overlap_max=float(propensity_config["overlap_max"]),
    )
    plot_love_by_strength(smds, figure_dir / "pre_matching_love_plot.svg")

    summary = {
        "seed": seed,
        "model": "LogisticRegression",
        "features": list(PRE_TREATMENT_COLUMNS),
        "overlap_thresholds": {
            "minimum": float(propensity_config["overlap_min"]),
            "maximum": float(propensity_config["overlap_max"]),
        },
        "samples": diagnostics,
        "standardized_mean_differences": smds,
        "figures": {
            "propensity_overlap": "experiments/figures/propensity_overlap.svg",
            "pre_matching_love_plot": "experiments/figures/pre_matching_love_plot.svg",
        },
    }
    payload = json.dumps(summary, indent=2) + "\n"
    result_path = Path("experiments/results/propensity_diagnostics_summary.json")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(payload)
    (score_output / "summary.json").write_text(payload)
    print(payload)
    print(f"wrote {result_path}")


if __name__ == "__main__":
    main()
