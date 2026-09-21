from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from causal_uplift.causal.dml import fit_linear_dml, make_cross_fit_splits
from causal_uplift.causal.matching import match_on_propensity, matched_frame, paired_att
from causal_uplift.causal.propensity import LogisticPropensityModel, propensity_diagnostics
from causal_uplift.data.confounding import ConfoundingConfig, sample_observational
from causal_uplift.evaluation.ate import bootstrap_difference_in_means, difference_in_means
from causal_uplift.evaluation.balance import covariate_smds
from causal_uplift.evaluation.overlap import inverse_probability_weight_diagnostics
from causal_uplift.visualization.plots import (
    plot_overlap_estimate_stability,
    plot_overlap_stress_diagnostics,
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


def _mean(rows: list[dict[str, object]], *keys: str) -> float:
    values = []
    for row in rows:
        value: object = row
        for key in keys:
            value = value[key]  # type: ignore[index]
        values.append(float(value))
    return float(np.mean(values))


def _sample_sd(rows: list[dict[str, object]], *keys: str) -> float:
    values = []
    for row in rows:
        value: object = row
        for key in keys:
            value = value[key]  # type: ignore[index]
        values.append(float(value))
    return float(np.std(values, ddof=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    seed = int(config["project"]["seed"])
    stress_config = config["overlap_stress"]
    propensity_config = config["propensity"]
    matching_config = config["matching"]
    dml_config = config["dml"]
    coefficients = config["confounding"]["coefficients"]
    overlap_min = float(matching_config["overlap_min"])
    overlap_max = float(matching_config["overlap_max"])
    bootstrap_samples = int(stress_config["bootstrap_samples"])
    nuisance_candidate = str(stress_config["nuisance_candidate"])
    training_pool = pd.read_csv("data/processed/randomized_training_pool.csv")

    started = time.perf_counter()
    samples: dict[str, list[dict[str, object]]] = {}
    for level_name, level in stress_config["levels"].items():
        samples[level_name] = []
        for replicate_seed in stress_config["seeds"]:
            replicate_started = time.perf_counter()
            frame = sample_observational(
                training_pool,
                ConfoundingConfig(
                    strength=float(level["strength"]),
                    clip_min=float(level["clip_min"]),
                    clip_max=float(level["clip_max"]),
                    **{key: float(value) for key, value in coefficients.items()},
                ),
                seed=int(replicate_seed),
            )
            treatment = frame["treatment"].to_numpy(dtype=int)
            desired_propensity = frame["desired_propensity"].to_numpy(dtype=float)
            fitted_propensity = (
                LogisticPropensityModel(
                    seed=int(replicate_seed),
                    max_iter=int(propensity_config["max_iter"]),
                )
                .fit(frame)
                .predict(frame)
            )
            before_balance = covariate_smds(frame, BALANCE_COLUMNS)
            match = match_on_propensity(
                frame,
                fitted_propensity,
                replacement=True,
                caliper_sd=float(matching_config["caliper_sd"]),
                overlap_min=overlap_min,
                overlap_max=overlap_max,
                seed=int(replicate_seed),
            )
            after_balance = covariate_smds(matched_frame(frame, match.pairs), BALANCE_COLUMNS)
            att = paired_att(
                frame,
                match.pairs,
                "conversion",
                bootstrap_samples=bootstrap_samples,
                seed=int(replicate_seed),
            )
            splits = make_cross_fit_splits(
                frame["conversion"],
                frame["treatment"],
                folds=int(dml_config["cross_fit_folds"]),
                seed=int(replicate_seed),
            )
            dml = fit_linear_dml(
                frame,
                "conversion",
                candidate=nuisance_candidate,
                splits=splits,
                seed=int(replicate_seed),
                random_forest_trees=int(dml_config["random_forest_trees"]),
                min_samples_leaf=int(dml_config["min_samples_leaf"]),
                max_iter=int(dml_config["max_iter"]),
            )
            treated_rows = int(treatment.sum())
            samples[level_name].append(
                {
                    "seed": int(replicate_seed),
                    "rows": len(frame),
                    "treated_rows": treated_rows,
                    "treated_share": float(treatment.mean()),
                    "desired_propensity": propensity_diagnostics(
                        frame,
                        desired_propensity,
                        overlap_min=overlap_min,
                        overlap_max=overlap_max,
                    ),
                    "fitted_propensity": propensity_diagnostics(
                        frame,
                        fitted_propensity,
                        overlap_min=overlap_min,
                        overlap_max=overlap_max,
                    ),
                    "desired_weights": inverse_probability_weight_diagnostics(
                        treatment,
                        desired_propensity,
                        overlap_min=overlap_min,
                        overlap_max=overlap_max,
                    ),
                    "fitted_weights": inverse_probability_weight_diagnostics(
                        treatment,
                        fitted_propensity,
                        overlap_min=overlap_min,
                        overlap_max=overlap_max,
                    ),
                    "balance_before_max_absolute_smd": max(
                        abs(value) for value in before_balance.values()
                    ),
                    "matching": {
                        "matched_pairs": match.matched_treated,
                        "matched_treated_fraction": match.matched_treated / treated_rows,
                        "eligible_treated": match.eligible_treated,
                        "eligible_controls": match.eligible_controls,
                        "discarded_rows_outside_overlap": match.discarded_outside_overlap,
                        "max_absolute_smd": max(abs(value) for value in after_balance.values()),
                        "balance_passes_0_10": max(abs(value) for value in after_balance.values())
                        < 0.10,
                        "att": att,
                    },
                    "naive_association": difference_in_means(frame, "conversion"),
                    "linear_dml": {
                        "estimate": dml.estimate,
                        "ci_lower": dml.ci_lower,
                        "ci_upper": dml.ci_upper,
                    },
                    "runtime_seconds": time.perf_counter() - replicate_started,
                }
            )

    # The randomized outcomes are opened only after every observational fit is frozen.
    rct = pd.read_csv("data/processed/rct_evaluation.csv")
    rct_effect = bootstrap_difference_in_means(
        rct,
        "conversion",
        samples=int(config["evaluation"]["bootstrap_samples"]),
        seed=seed,
    )
    rct_ate = rct_effect["estimate"]
    aggregate: dict[str, dict[str, object]] = {}
    for level_name, rows in samples.items():
        for row in rows:
            row["naive_absolute_error_vs_rct_ate"] = abs(float(row["naive_association"]) - rct_ate)
            dml = row["linear_dml"]
            dml["absolute_error_vs_rct_ate"] = abs(float(dml["estimate"]) - rct_ate)
            dml["covers_rct_ate"] = float(dml["ci_lower"]) <= rct_ate <= float(dml["ci_upper"])
            matching = row["matching"]
            matching["att"]["reference_gap_vs_overall_rct_ate"] = (
                float(matching["att"]["estimate"]) - rct_ate
            )

        aggregate[level_name] = {
            "rows_mean": _mean(rows, "rows"),
            "rows_min": min(int(row["rows"]) for row in rows),
            "rows_max": max(int(row["rows"]) for row in rows),
            "desired_outside_overlap_fraction_mean": _mean(
                rows, "desired_propensity", "outside_overlap_threshold_fraction"
            ),
            "fitted_outside_overlap_fraction_mean": _mean(
                rows, "fitted_propensity", "outside_overlap_threshold_fraction"
            ),
            "desired_weight_p99_mean": _mean(rows, "desired_weights", "weight_p99"),
            "desired_weight_max_max": max(
                float(row["desired_weights"]["weight_max"]) for row in rows
            ),
            "fitted_weight_p99_mean": _mean(rows, "fitted_weights", "weight_p99"),
            "fitted_weight_max_max": max(
                float(row["fitted_weights"]["weight_max"]) for row in rows
            ),
            "desired_effective_sample_fraction_mean": _mean(
                rows, "desired_weights", "effective_sample_fraction"
            ),
            "fitted_effective_sample_fraction_mean": _mean(
                rows, "fitted_weights", "effective_sample_fraction"
            ),
            "matching": {
                "matched_pairs_mean": _mean(rows, "matching", "matched_pairs"),
                "matched_pairs_min": min(int(row["matching"]["matched_pairs"]) for row in rows),
                "matched_treated_fraction_mean": _mean(
                    rows, "matching", "matched_treated_fraction"
                ),
                "discarded_rows_outside_overlap_mean": _mean(
                    rows, "matching", "discarded_rows_outside_overlap"
                ),
                "max_absolute_smd_mean": _mean(rows, "matching", "max_absolute_smd"),
                "balance_pass_rate": float(
                    np.mean([row["matching"]["balance_passes_0_10"] for row in rows])
                ),
                "att_mean": _mean(rows, "matching", "att", "estimate"),
                "att_replicate_sd": _sample_sd(rows, "matching", "att", "estimate"),
                "att_ci_width_mean": _mean(rows, "matching", "att", "ci_upper")
                - _mean(rows, "matching", "att", "ci_lower"),
            },
            "naive": {
                "estimate_mean": _mean(rows, "naive_association"),
                "replicate_sd": _sample_sd(rows, "naive_association"),
                "absolute_error_mean": _mean(rows, "naive_absolute_error_vs_rct_ate"),
            },
            "linear_dml": {
                "estimate_mean": _mean(rows, "linear_dml", "estimate"),
                "replicate_sd": _sample_sd(rows, "linear_dml", "estimate"),
                "analytic_ci_width_mean": _mean(rows, "linear_dml", "ci_upper")
                - _mean(rows, "linear_dml", "ci_lower"),
                "absolute_error_mean": _mean(rows, "linear_dml", "absolute_error_vs_rct_ate"),
                "rct_ate_coverage_rate": float(
                    np.mean([row["linear_dml"]["covers_rct_ate"] for row in rows])
                ),
            },
        }

    figure_dir = Path("experiments/figures")
    diagnostics_path = figure_dir / "overlap_stress_diagnostics.svg"
    stability_path = figure_dir / "overlap_estimate_stability.svg"
    plot_overlap_stress_diagnostics(aggregate, diagnostics_path)
    plot_overlap_estimate_stability(samples, aggregate, rct_effect, stability_path)
    summary = {
        "seed": seed,
        "stress_levels": stress_config["levels"],
        "replicate_seeds": stress_config["seeds"],
        "nuisance_candidate_frozen_before_rct": nuisance_candidate,
        "overlap_thresholds": {"minimum": overlap_min, "maximum": overlap_max},
        "rct_outcomes_loaded_after_all_observational_fits": True,
        "rct_effect": rct_effect,
        "ipw_note": "IPW weights are diagnostics only; no IPW effect estimate is reported.",
        "psm_estimand_warning": (
            "Matched estimates are ATT in retained treated customers. Their RCT ATE gap is a "
            "reference only, not a like-for-like ATE error."
        ),
        "samples": samples,
        "aggregate": aggregate,
        "runtime_seconds": time.perf_counter() - started,
        "figures": {
            "diagnostics": str(diagnostics_path),
            "estimate_stability": str(stability_path),
        },
    }
    payload = json.dumps(summary, indent=2) + "\n"
    output_dir = Path("outputs/overlap_stress")
    result_path = Path("experiments/results/overlap_stress_summary.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(payload)
    result_path.write_text(payload)
    print(payload)
    print(f"wrote {result_path}")


if __name__ == "__main__":
    main()
