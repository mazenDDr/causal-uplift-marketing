from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import yaml

from causal_uplift.causal.matching import (
    match_on_propensity,
    matched_frame,
    paired_att,
)
from causal_uplift.causal.propensity import LogisticPropensityModel
from causal_uplift.evaluation.ate import difference_in_means
from causal_uplift.evaluation.balance import covariate_smds
from causal_uplift.visualization.plots import plot_matching_balance

BALANCE_COLUMNS = (
    "history",
    "recency",
    "mens",
    "womens",
    "newbie",
    "channel",
    "zip_code",
)
OUTCOMES = ("conversion", "visit", "spend")


def _balance_summary(smds: dict[str, float]) -> dict[str, float | int]:
    return {
        "max_absolute_smd": max(abs(value) for value in smds.values()),
        "covariates_above_0_10_absolute_smd": sum(abs(value) >= 0.10 for value in smds.values()),
    }


def _preferred_variant(variants: dict[str, dict[str, object]]) -> str:
    """Choose by balance, then retention; outcomes are intentionally absent."""
    passing = [
        name for name, result in variants.items() if result["balance"]["max_absolute_smd"] < 0.10
    ]
    candidates = passing or list(variants)
    if passing:
        return max(candidates, key=lambda name: variants[name]["matched_treated"])
    return min(
        candidates,
        key=lambda name: (
            variants[name]["balance"]["max_absolute_smd"],
            -variants[name]["matched_treated"],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    seed = int(config["project"]["seed"])
    propensity_config = config["propensity"]
    matching_config = config["matching"]
    pair_output = Path("outputs/matching")
    pair_output.mkdir(parents=True, exist_ok=True)

    rct = pd.read_csv("data/processed/rct_evaluation.csv")
    rct_ate = {outcome: difference_in_means(rct, outcome) for outcome in OUTCOMES}
    samples: dict[str, object] = {}
    before_smds: dict[str, dict[str, float]] = {}
    after_smds: dict[str, dict[str, float]] = {}

    for strength in config["confounding"]["strengths"]:
        frame = pd.read_csv(Path("data/observational") / f"{strength}.csv")
        score = (
            LogisticPropensityModel(
                seed=seed,
                max_iter=int(propensity_config["max_iter"]),
            )
            .fit(frame)
            .predict(frame)
        )
        before = covariate_smds(frame, BALANCE_COLUMNS)
        before_smds[strength] = before
        variants: dict[str, dict[str, object]] = {}

        for variant in matching_config["variants"]:
            replacement = variant == "with_replacement"
            started = time.perf_counter()
            result = match_on_propensity(
                frame,
                score,
                replacement=replacement,
                caliper_sd=float(matching_config["caliper_sd"]),
                overlap_min=float(matching_config["overlap_min"]),
                overlap_max=float(matching_config["overlap_max"]),
                seed=seed,
            )
            matched = matched_frame(frame, result.pairs)
            balance = covariate_smds(matched, BALANCE_COLUMNS)
            outcome_results = {
                outcome: paired_att(
                    frame,
                    result.pairs,
                    outcome,
                    bootstrap_samples=int(matching_config["bootstrap_samples"]),
                    seed=seed,
                )
                for outcome in OUTCOMES
            }
            for outcome, estimate in outcome_results.items():
                estimate["gap_vs_overall_rct_ate"] = estimate["estimate"] - rct_ate[outcome]
            reused_controls = len(result.pairs) - result.pairs["control_source_row_id"].nunique()
            variants[variant] = {
                "replacement": replacement,
                "matched_treated": len(result.pairs),
                "eligible_treated": result.eligible_treated,
                "eligible_controls": result.eligible_controls,
                "unmatched_eligible_treated": result.eligible_treated - len(result.pairs),
                "discarded_rows_outside_overlap": result.discarded_outside_overlap,
                "unique_matched_controls": result.pairs["control_source_row_id"].nunique(),
                "reused_control_matches": reused_controls,
                "caliper_logit_distance": result.caliper,
                "maximum_matched_logit_distance": float(result.pairs["logit_distance"].max()),
                "balance": _balance_summary(balance),
                "standardized_mean_differences": balance,
                "att": outcome_results,
                "runtime_seconds": time.perf_counter() - started,
            }
            result.pairs.to_csv(pair_output / f"{strength}_{variant}_pairs.csv", index=False)

        preferred = _preferred_variant(variants)
        after_smds[strength] = variants[preferred]["standardized_mean_differences"]
        samples[strength] = {
            "rows": len(frame),
            "balance_before": _balance_summary(before),
            "preferred_variant": preferred,
            "selection_rule": (
                "pass max |SMD| < 0.10, then retain most treated; otherwise lowest max |SMD|"
            ),
            "variants": variants,
        }

    figure_path = Path("experiments/figures/matching_balance.svg")
    plot_matching_balance(before_smds, after_smds, figure_path)
    summary = {
        "seed": seed,
        "method": "1:1 nearest-neighbor matching on logit propensity",
        "propensity_model": "LogisticRegression",
        "selection_uses_outcomes_or_rct": False,
        "settings": matching_config,
        "rct_ate_reference": rct_ate,
        "rct_reference_note": (
            "PSM estimates ATT in the selected observational population; the overall RCT ATE "
            "gap is reported as a reference, not labeled estimator error."
        ),
        "samples": samples,
        "figure": str(figure_path),
    }
    payload = json.dumps(summary, indent=2) + "\n"
    result_path = Path("experiments/results/matching_summary.json")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(payload)
    (pair_output / "summary.json").write_text(payload)
    print(payload)
    print(f"wrote {result_path}")


if __name__ == "__main__":
    main()
