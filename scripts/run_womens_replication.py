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
from causal_uplift.causal.causal_forest import effect_bins, fit_causal_forest
from causal_uplift.causal.dml import fit_linear_dml, make_cross_fit_splits
from causal_uplift.causal.matching import match_on_propensity, matched_frame, paired_att
from causal_uplift.causal.propensity import LogisticPropensityModel, propensity_diagnostics
from causal_uplift.causal.uplift import fit_uplift_forest, fit_uplift_tree
from causal_uplift.data.confounding import ConfoundingConfig, sample_observational
from causal_uplift.data.load import (
    OUTCOME_COLUMNS,
    PRE_TREATMENT_COLUMNS,
    campaign_frame,
    load_hillstrom,
)
from causal_uplift.data.preprocess import randomized_train_evaluation_split
from causal_uplift.evaluation.ablation import summarize_ate_estimators
from causal_uplift.evaluation.ate import bootstrap_difference_in_means, difference_in_means
from causal_uplift.evaluation.balance import covariate_smds
from causal_uplift.evaluation.business import (
    bootstrap_profit_difference,
    evaluate_business_policy,
    psm_segment_scores,
    ranked_policy_mask,
)
from causal_uplift.evaluation.policy import bootstrap_policy_difference, random_policy_scores
from causal_uplift.evaluation.uplift_metrics import bootstrap_ranking_metrics
from causal_uplift.visualization.plots import plot_campaign_replication

BALANCE_COLUMNS = ("history", "recency", "mens", "womens", "newbie", "channel", "zip_code")
CAMPAIGN = "Womens E-Mail"
CONTROL = "No E-Mail"
AFFINITY_COLUMN = "womens"


def balance_summary(smds: dict[str, float]) -> dict[str, float | int | bool]:
    maximum = max(abs(value) for value in smds.values())
    return {
        "max_absolute_smd": maximum,
        "covariates_above_0_10_absolute_smd": sum(abs(value) >= 0.10 for value in smds.values()),
        "passes_0_10": maximum < 0.10,
    }


def select_match(variants: dict[str, dict[str, object]]) -> str:
    """Apply the frozen balance-first matching rule without reading outcomes."""
    passing = [name for name, result in variants.items() if result["balance"]["passes_0_10"]]
    if passing:
        return max(passing, key=lambda name: variants[name]["matched_treated"])
    return min(
        variants,
        key=lambda name: (
            variants[name]["balance"]["max_absolute_smd"],
            -variants[name]["matched_treated"],
        ),
    )


def add_per_1000(metrics: dict[str, object]) -> None:
    for result in metrics["models"].values():
        for metric in ("auuc", "qini"):
            values = result[metric]
            for field in ("estimate", "ci_lower", "ci_upper"):
                values[f"{field}_per_1000"] = values[field] * 1000
    for values in metrics["paired_qini_difference_vs_reference"]["models"].values():
        for field in ("estimate", "ci_lower", "ci_upper"):
            values[f"{field}_per_1000"] = values[field] * 1000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    seed = int(config["project"]["seed"])
    bootstrap_samples = int(config["uplift_metrics"]["bootstrap_samples"])
    medium_strength = float(config["confounding"]["strengths"]["medium"])
    started = time.perf_counter()

    raw_path = Path(config["data"]["raw_path"])
    campaign = campaign_frame(
        load_hillstrom(raw_path),
        treatment_label=CAMPAIGN,
        control_label=CONTROL,
    )
    training_pool, evaluation_for_split = randomized_train_evaluation_split(
        campaign,
        evaluation_fraction=float(config["project"]["evaluation_fraction"]),
        seed=seed,
    )
    evaluation_ids = evaluation_for_split["source_row_id"].to_numpy()
    scoring = evaluation_for_split.loc[:, ["source_row_id", *PRE_TREATMENT_COLUMNS]].copy()
    del campaign, evaluation_for_split

    coefficients = config["confounding"]["coefficients"]
    training = sample_observational(
        training_pool,
        ConfoundingConfig(
            strength=medium_strength,
            clip_min=float(config["confounding"]["clip_min"]),
            clip_max=float(config["confounding"]["clip_max"]),
            affinity_column=AFFINITY_COLUMN,
            **{name: float(value) for name, value in coefficients.items()},
        ),
        seed=seed + list(config["confounding"]["strengths"]).index("medium"),
    )
    if not set(training["source_row_id"]).issubset(set(training_pool["source_row_id"])):
        raise AssertionError("observational sampling fabricated source rows")

    propensity = (
        LogisticPropensityModel(seed=seed, max_iter=int(config["propensity"]["max_iter"]))
        .fit(training)
        .predict(training)
    )
    before_smd = covariate_smds(training, BALANCE_COLUMNS)
    variants: dict[str, dict[str, object]] = {}
    pair_frames: dict[str, pd.DataFrame] = {}
    for variant in config["matching"]["variants"]:
        match = match_on_propensity(
            training,
            propensity,
            replacement=variant == "with_replacement",
            caliper_sd=float(config["matching"]["caliper_sd"]),
            overlap_min=float(config["matching"]["overlap_min"]),
            overlap_max=float(config["matching"]["overlap_max"]),
            seed=seed,
        )
        pair_frames[variant] = match.pairs
        after_smd = covariate_smds(matched_frame(training, match.pairs), BALANCE_COLUMNS)
        variants[variant] = {
            "replacement": variant == "with_replacement",
            "matched_treated": match.matched_treated,
            "eligible_treated": match.eligible_treated,
            "eligible_controls": match.eligible_controls,
            "discarded_rows_outside_overlap": match.discarded_outside_overlap,
            "caliper_logit_distance": match.caliper,
            "maximum_matched_logit_distance": float(match.pairs["logit_distance"].max()),
            "balance": balance_summary(after_smd),
            "standardized_mean_differences": after_smd,
        }
    preferred_match = select_match(variants)
    pairs = pair_frames[preferred_match]

    mens_dml = json.loads(Path("experiments/results/linear_dml_summary.json").read_text())
    if mens_dml["selection_uses_rct"]:
        raise AssertionError("frozen nuisance choice cannot have used the Men's RCT")
    nuisance_candidate = mens_dml["samples"]["medium"]["selected_nuisance_candidate"]
    dml_config = config["dml"]
    splits = make_cross_fit_splits(
        training["conversion"],
        training["treatment"],
        folds=int(dml_config["cross_fit_folds"]),
        seed=seed,
    )
    dml = fit_linear_dml(
        training,
        "conversion",
        candidate=nuisance_candidate,
        splits=splits,
        seed=seed,
        random_forest_trees=int(dml_config["random_forest_trees"]),
        min_samples_leaf=int(dml_config["min_samples_leaf"]),
        max_iter=int(dml_config["max_iter"]),
    )
    response = ResponseModel(seed=seed).fit(training)
    pseudo = TreatmentFeatureModel(seed=seed).fit(training)

    forest_config = config["causal_forest"]
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

    uplift_config = config["uplift_models"]
    uplift_common = {
        "control_name": str(uplift_config["control_name"]),
        "treatment_name": "womens_email",
        "evaluation_function": str(uplift_config["evaluation_function"]),
        "honesty": bool(uplift_config["honesty"]),
        "estimation_sample_size": float(uplift_config["estimation_sample_size"]),
        "seed": seed,
    }
    uplift_tree = fit_uplift_tree(training, **uplift_common, **uplift_config["tree"])
    uplift_forest = fit_uplift_forest(training, **uplift_common, **uplift_config["forest"])

    psm_score, psm_segments = psm_segment_scores(
        training,
        pairs,
        scoring,
        history_threshold=float(training["history"].median()),
        recency_threshold=float(training["recency"].median()),
    )
    scores = {
        "response_model": response.predict(scoring),
        "pseudo_uplift": pseudo.predict_uplift(scoring),
        "psm_segments": psm_score,
        "causal_forest_dml": causal_forest.predict(scoring),
        "uplift_tree": uplift_tree.predict(scoring),
        "uplift_random_forest": uplift_forest.predict(scoring),
    }
    causal_ate, causal_lower, causal_upper = causal_forest.ate_interval(scoring)
    model_fit_seconds = time.perf_counter() - started

    # Reload randomized outcomes only after every model, matching rule, and score is frozen.
    evaluation_campaign = campaign_frame(
        load_hillstrom(raw_path),
        treatment_label=CAMPAIGN,
        control_label=CONTROL,
    )
    _, evaluation = randomized_train_evaluation_split(
        evaluation_campaign,
        evaluation_fraction=float(config["project"]["evaluation_fraction"]),
        seed=seed,
    )
    evaluation = evaluation.sort_values("source_row_id").reset_index(drop=True)
    scoring = scoring.sort_values("source_row_id").reset_index(drop=True)
    order = np.argsort(evaluation_ids)
    if not np.array_equal(evaluation["source_row_id"].to_numpy(), evaluation_ids[order]):
        raise AssertionError("the randomized Women's holdout changed after model fitting")
    scores = {name: np.asarray(score)[order] for name, score in scores.items()}

    rct_effects = {
        outcome: bootstrap_difference_in_means(
            evaluation,
            outcome,
            samples=bootstrap_samples,
            seed=seed,
        )
        for outcome in OUTCOME_COLUMNS
    }
    matching_att = {
        outcome: paired_att(
            training,
            pairs,
            outcome,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        for outcome in OUTCOME_COLUMNS
    }
    ate_estimators = summarize_ate_estimators(
        rct_ate=rct_effects["conversion"]["estimate"],
        naive=difference_in_means(training, "conversion"),
        linear_dml=dml.estimate,
        causal_forest=causal_ate,
        uplift_forest=float(scores["uplift_random_forest"].mean()),
        psm_att=matching_att["conversion"]["estimate"],
    )
    ate_estimators["linear_dml"].update(ci_lower=dml.ci_lower, ci_upper=dml.ci_upper)
    ate_estimators["causal_forest_dml"].update(
        ci_lower=causal_lower,
        ci_upper=causal_upper,
    )

    ranking = bootstrap_ranking_metrics(
        evaluation,
        scores,
        np.asarray(config["uplift_metrics"]["fractions"], dtype=float),
        samples=bootstrap_samples,
        seed=seed,
        reference="response_model",
    )
    add_per_1000(ranking)
    top_bottom = {}
    for name, score in scores.items():
        bins = effect_bins(score, bins=int(uplift_config["cate_bins"]))
        top_bottom[name] = bootstrap_policy_difference(
            evaluation,
            bins == bins.max(),
            bins == bins.min(),
            samples=bootstrap_samples,
            seed=seed,
        )

    random_score = random_policy_scores(evaluation, seed=seed)
    policies: dict[str, dict[str, object]] = {}
    for budget in (0.05, 0.20):
        budget_key = f"{budget:.2f}"
        random_mask = ranked_policy_mask(random_score, budget, tie_breaker=random_score)
        budget_policies = {
            "random": evaluate_business_policy(
                evaluation,
                random_mask,
                email_cost=float(config["business_policy"]["primary_email_cost"]),
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
        }
        paired_vs_random = {}
        paired_vs_response = {}
        response_mask = ranked_policy_mask(
            scores["response_model"],
            budget,
            tie_breaker=random_score,
        )
        for name, score in scores.items():
            mask = ranked_policy_mask(score, budget, tie_breaker=random_score)
            budget_policies[name] = evaluate_business_policy(
                evaluation,
                mask,
                email_cost=float(config["business_policy"]["primary_email_cost"]),
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
            paired_vs_random[name] = bootstrap_profit_difference(
                evaluation,
                mask,
                random_mask,
                email_cost=float(config["business_policy"]["primary_email_cost"]),
                samples=bootstrap_samples,
                seed=seed,
            )
            if name != "response_model":
                paired_vs_response[name] = bootstrap_profit_difference(
                    evaluation,
                    mask,
                    response_mask,
                    email_cost=float(config["business_policy"]["primary_email_cost"]),
                    samples=bootstrap_samples,
                    seed=seed,
                )
        policies[budget_key] = {
            "models": budget_policies,
            "paired_profit_difference_vs_random": paired_vs_random,
            "paired_profit_difference_vs_response": paired_vs_response,
        }

    mens_ablation = json.loads(
        Path("experiments/results/confounding_ablation_summary.json").read_text()
    )
    mens_metrics = json.loads(Path("experiments/results/uplift_metrics_summary.json").read_text())
    mens_reference = {
        "campaign": config["data"]["campaign"],
        "rct_conversion_effect": mens_ablation["rct_effect"],
        "medium_ate_estimators": mens_ablation["samples"]["medium"]["ate_estimators"],
        "qini": {name: values["qini"] for name, values in mens_metrics["models"].items()},
        "medium_policy_at_20_percent": mens_ablation["samples"]["medium"]["policies"],
        "source_artifacts": [
            "confounding_ablation_summary.json",
            "uplift_metrics_summary.json",
        ],
    }
    del response, pseudo, causal_forest, uplift_tree, uplift_forest
    gc.collect()

    summary = {
        "seed": seed,
        "campaign": CAMPAIGN,
        "control": CONTROL,
        "campaign_affinity_column": AFFINITY_COLUMN,
        "frozen_from_mens_pipeline": {
            "split_seed": seed,
            "evaluation_fraction": float(config["project"]["evaluation_fraction"]),
            "confounding_strength": medium_strength,
            "confounding_coefficients": coefficients,
            "semantic_label_substitutions": {
                "campaign": {"from": config["data"]["campaign"], "to": CAMPAIGN},
                "affinity_column": {"from": "mens", "to": AFFINITY_COLUMN},
                "uplift_treatment_name": {
                    "from": uplift_config["treatment_name"],
                    "to": uplift_common["treatment_name"],
                },
            },
            "nuisance_candidate": nuisance_candidate,
            "nuisance_selection_source": "Men's medium-confounding training-only selection",
            "model_hyperparameters": {
                "dml": dml_config,
                "causal_forest": forest_config,
                "uplift_models": uplift_config,
            },
            "womens_rct_used_for_selection": False,
        },
        "rows": {
            "campaign_total": len(evaluation_campaign),
            "randomized_training_pool": len(training_pool),
            "observational_training": len(training),
            "rct_evaluation": len(evaluation),
        },
        "rct_effects": rct_effects,
        "observational_design": {
            "propensity": propensity_diagnostics(
                training,
                propensity,
                overlap_min=float(config["propensity"]["overlap_min"]),
                overlap_max=float(config["propensity"]["overlap_max"]),
            ),
            "balance_before": balance_summary(before_smd),
            "standardized_mean_differences_before": before_smd,
            "matching_selection_rule": (
                "pass max |SMD| < 0.10, then retain most treated; otherwise lowest max |SMD|"
            ),
            "preferred_matching_variant": preferred_match,
            "matching_variants": variants,
            "matching_att": matching_att,
            "psm_segments": psm_segments,
            "psm_estimand_warning": (
                "PSM estimates ATT in retained treated customers; its RCT ATE gap is not an "
                "ATE error."
            ),
        },
        "ate_estimators": ate_estimators,
        "ranking": ranking,
        "top_minus_bottom_decile": top_bottom,
        "policies": policies,
        "mens_reference": mens_reference,
        "runtime_seconds_before_rct_evaluation": model_fit_seconds,
        "runtime_seconds_total": time.perf_counter() - started,
    }
    figure_path = Path("experiments/figures/womens_replication.svg")
    plot_campaign_replication(summary, figure_path)
    summary["figure"] = str(figure_path)

    output_dir = Path("outputs/womens_replication")
    result_path = Path("experiments/results/womens_replication_summary.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(summary, indent=2) + "\n"
    (output_dir / "summary.json").write_text(payload)
    result_path.write_text(payload)
    pd.DataFrame({"source_row_id": evaluation["source_row_id"], **scores}).to_csv(
        output_dir / "rct_scores.csv",
        index=False,
    )
    print(payload)
    print(f"wrote {result_path} and {figure_path}")


if __name__ == "__main__":
    main()
