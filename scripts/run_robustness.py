from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from causal_uplift.causal.dml import (
    fit_linear_dml,
    make_confounder_encoder,
    make_cross_fit_splits,
)
from causal_uplift.causal.matching import match_on_propensity, matched_frame, paired_att
from causal_uplift.causal.propensity import propensity_diagnostics
from causal_uplift.data.confounding import ConfoundingConfig, sample_observational
from causal_uplift.data.load import PRE_TREATMENT_COLUMNS
from causal_uplift.evaluation.ate import bootstrap_difference_in_means
from causal_uplift.evaluation.balance import covariate_smds
from causal_uplift.evaluation.robustness import (
    add_noise_covariates,
    joint_stratified_subsample,
    shuffle_treatment,
)
from causal_uplift.visualization.plots import (
    plot_robustness_dml,
    plot_robustness_matching,
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


def _propensity_scores(frame: pd.DataFrame, name: str, *, seed: int) -> np.ndarray:
    encoder = make_confounder_encoder()
    features = encoder.fit_transform(frame.loc[:, PRE_TREATMENT_COLUMNS])
    if name == "logistic":
        model = LogisticRegression(max_iter=2000, random_state=seed)
    elif name == "gradient_boosting":
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            min_samples_leaf=100,
            l2_regularization=1.0,
            random_state=seed,
        )
    elif name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=100,
            n_jobs=-1,
            random_state=seed,
        )
    else:
        raise ValueError(f"unknown propensity model: {name}")
    model.fit(features, frame["treatment"])
    return model.predict_proba(features)[:, 1]


def _fit_dml(
    frame: pd.DataFrame,
    *,
    candidate: str,
    seed: int,
    dml_config: dict[str, object],
    additional_numeric_features: tuple[str, ...] = (),
) -> dict[str, float]:
    splits = make_cross_fit_splits(
        frame["conversion"],
        frame["treatment"],
        folds=int(dml_config["cross_fit_folds"]),
        seed=seed,
    )
    result = fit_linear_dml(
        frame,
        "conversion",
        candidate=candidate,
        splits=splits,
        seed=seed,
        random_forest_trees=int(dml_config["random_forest_trees"]),
        min_samples_leaf=int(dml_config["min_samples_leaf"]),
        max_iter=int(dml_config["max_iter"]),
        additional_numeric_features=additional_numeric_features,
    )
    return {
        "estimate": result.estimate,
        "ci_lower": result.ci_lower,
        "ci_upper": result.ci_upper,
        "ci_width": result.ci_upper - result.ci_lower,
    }


def _matching_result(
    frame: pd.DataFrame,
    propensity: np.ndarray,
    *,
    replacement: bool,
    ratio: int,
    caliper: float,
    seed: int,
    overlap_min: float,
    overlap_max: float,
    bootstrap_samples: int,
) -> dict[str, object]:
    result = match_on_propensity(
        frame,
        propensity,
        replacement=replacement,
        matching_ratio=ratio,
        caliper_sd=caliper,
        overlap_min=overlap_min,
        overlap_max=overlap_max,
        seed=seed,
    )
    balance = covariate_smds(matched_frame(frame, result.pairs), BALANCE_COLUMNS)
    return {
        "replacement": replacement,
        "matching_ratio": ratio,
        "caliper_sd": caliper,
        "matched_treated": result.matched_treated,
        "control_matches": len(result.pairs),
        "discarded_rows_outside_overlap": result.discarded_outside_overlap,
        "max_absolute_smd": max(abs(value) for value in balance.values()),
        "balance_passes_0_10": max(abs(value) for value in balance.values()) < 0.10,
        "att": paired_att(
            frame,
            result.pairs,
            "conversion",
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
    }


def _replicate_summary(rows: list[dict[str, float]], reference: float) -> dict[str, float]:
    estimates = np.array([row["estimate"] for row in rows])
    return {
        "estimate_mean": float(estimates.mean()),
        "estimate_sd": float(estimates.std(ddof=1)),
        "absolute_error_mean": float(np.mean(np.abs(estimates - reference))),
        "ci_width_mean": float(np.mean([row["ci_width"] for row in rows])),
        "reference_coverage_rate": float(
            np.mean([row["ci_lower"] <= reference <= row["ci_upper"] for row in rows])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    seed = int(config["project"]["seed"])
    robustness = config["robustness"]
    matching_config = config["matching"]
    dml_config = config["dml"]
    bootstrap_samples = int(robustness["bootstrap_samples"])
    fixed_candidate = str(robustness["fixed_nuisance_candidate"])
    frame = pd.read_csv(Path("data/observational") / f"{robustness['training_sample']}.csv")
    started = time.perf_counter()

    propensity_results: dict[str, dict[str, object]] = {}
    propensity_scores: dict[str, np.ndarray] = {}
    for name in robustness["propensity_models"]:
        score = _propensity_scores(frame, str(name), seed=seed)
        propensity_scores[str(name)] = score
        propensity_results[str(name)] = {
            "diagnostics": propensity_diagnostics(
                frame,
                score,
                overlap_min=float(matching_config["overlap_min"]),
                overlap_max=float(matching_config["overlap_max"]),
            ),
            "matching": _matching_result(
                frame,
                score,
                replacement=True,
                ratio=1,
                caliper=float(matching_config["caliper_sd"]),
                seed=seed,
                overlap_min=float(matching_config["overlap_min"]),
                overlap_max=float(matching_config["overlap_max"]),
                bootstrap_samples=bootstrap_samples,
            ),
        }

    matching_grid = []
    for caliper in robustness["matching_calipers"]:
        for replacement in robustness["matching_replacement"]:
            for ratio in robustness["matching_ratios"]:
                matching_grid.append(
                    _matching_result(
                        frame,
                        propensity_scores["logistic"],
                        replacement=bool(replacement),
                        ratio=int(ratio),
                        caliper=float(caliper),
                        seed=seed,
                        overlap_min=float(matching_config["overlap_min"]),
                        overlap_max=float(matching_config["overlap_max"]),
                        bootstrap_samples=bootstrap_samples,
                    )
                )

    nuisance_results = {
        str(candidate): _fit_dml(
            frame,
            candidate=str(candidate),
            seed=seed,
            dml_config=dml_config,
        )
        for candidate in robustness["nuisance_candidates"]
    }

    training_pool = pd.read_csv("data/processed/randomized_training_pool.csv")
    coefficients = config["confounding"]["coefficients"]
    strength = float(config["confounding"]["strengths"][robustness["training_sample"]])
    seed_results = []
    for replicate_seed in robustness["seeds"]:
        observational = sample_observational(
            training_pool,
            ConfoundingConfig(
                strength=strength,
                clip_min=float(config["confounding"]["clip_min"]),
                clip_max=float(config["confounding"]["clip_max"]),
                **{key: float(value) for key, value in coefficients.items()},
            ),
            seed=int(replicate_seed),
        )
        result = _fit_dml(
            observational,
            candidate=fixed_candidate,
            seed=int(replicate_seed),
            dml_config=dml_config,
        )
        result.update({"seed": int(replicate_seed), "rows": len(observational)})
        seed_results.append(result)

    noisy_frame, noise_names = add_noise_covariates(
        frame,
        columns=int(robustness["noise_features"]),
        seed=seed,
    )
    noise_result = {
        "baseline": nuisance_results[fixed_candidate],
        "with_noise": _fit_dml(
            noisy_frame,
            candidate=fixed_candidate,
            seed=seed,
            dml_config=dml_config,
            additional_numeric_features=noise_names,
        ),
        "noise_features": list(noise_names),
    }
    noise_result["absolute_estimate_change"] = abs(
        noise_result["with_noise"]["estimate"] - noise_result["baseline"]["estimate"]
    )

    placebo_results = []
    for placebo_seed in robustness["seeds"]:
        placebo = shuffle_treatment(frame, seed=int(placebo_seed))
        result = _fit_dml(
            placebo,
            candidate=fixed_candidate,
            seed=int(placebo_seed),
            dml_config=dml_config,
        )
        result["seed"] = int(placebo_seed)
        placebo_results.append(result)

    sample_size_results: dict[str, list[dict[str, float]]] = {}
    for fraction in robustness["sample_fractions"]:
        fraction_key = f"{float(fraction):.2f}"
        sample_size_results[fraction_key] = []
        for sample_seed in robustness["seeds"]:
            sample = joint_stratified_subsample(frame, float(fraction), seed=int(sample_seed))
            result = _fit_dml(
                sample,
                candidate=fixed_candidate,
                seed=int(sample_seed),
                dml_config=dml_config,
            )
            result.update({"seed": int(sample_seed), "rows": len(sample)})
            sample_size_results[fraction_key].append(result)

    # The randomized benchmark is opened only after every robustness specification is frozen.
    rct = pd.read_csv("data/processed/rct_evaluation.csv")
    rct_effect = bootstrap_difference_in_means(
        rct,
        "conversion",
        samples=int(config["evaluation"]["bootstrap_samples"]),
        seed=seed,
    )
    rct_ate = rct_effect["estimate"]
    for result in nuisance_results.values():
        result["absolute_error_vs_rct_ate"] = abs(result["estimate"] - rct_ate)
    for result in propensity_results.values():
        result["matching"]["att"]["reference_gap_vs_overall_rct_ate"] = (
            result["matching"]["att"]["estimate"] - rct_ate
        )
    for result in matching_grid:
        result["att"]["reference_gap_vs_overall_rct_ate"] = result["att"]["estimate"] - rct_ate

    aggregate = {
        "selection_seed": _replicate_summary(seed_results, rct_ate),
        "placebo": _replicate_summary(placebo_results, 0.0),
        "sample_size": {
            fraction: _replicate_summary(results, rct_ate)
            for fraction, results in sample_size_results.items()
        },
    }
    figure_dir = Path("experiments/figures")
    dml_path = figure_dir / "robustness_dml.svg"
    matching_path = figure_dir / "robustness_matching.svg"
    plot_robustness_dml(
        nuisance_results,
        seed_results,
        placebo_results,
        sample_size_results,
        noise_result,
        aggregate,
        rct_effect,
        dml_path,
    )
    plot_robustness_matching(propensity_results, matching_grid, matching_path)

    summary = {
        "seed": seed,
        "settings": robustness,
        "rct_effect": rct_effect,
        "rct_outcomes_loaded_after_all_specifications_frozen": True,
        "propensity_models": propensity_results,
        "matching_grid": matching_grid,
        "nuisance_models": nuisance_results,
        "selection_seed_results": seed_results,
        "noise_covariate_result": noise_result,
        "placebo_results": placebo_results,
        "sample_size_results": sample_size_results,
        "aggregate": aggregate,
        "psm_estimand_warning": (
            "Every matching estimate is ATT in its retained treated population; RCT ATE gaps "
            "are references, not like-for-like estimator errors."
        ),
        "runtime_seconds": time.perf_counter() - started,
        "figures": {"dml": str(dml_path), "matching": str(matching_path)},
    }
    payload = json.dumps(summary, indent=2) + "\n"
    output_dir = Path("outputs/robustness")
    result_path = Path("experiments/results/robustness_summary.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(payload)
    result_path.write_text(payload)
    print(payload)
    print(f"wrote {result_path}")


if __name__ == "__main__":
    main()
