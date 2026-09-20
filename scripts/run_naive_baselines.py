from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from causal_uplift.causal.baselines import (
    ResponseModel,
    TreatmentFeatureModel,
    classification_metrics,
)
from causal_uplift.data.load import OUTCOME_COLUMNS
from causal_uplift.evaluation.ate import bootstrap_difference_in_means
from causal_uplift.evaluation.policy import (
    bootstrap_policy_difference,
    evaluate_binary_policy,
    random_policy_scores,
    top_fraction_mask,
)


def _timed_fit(model: ResponseModel | TreatmentFeatureModel, frame: pd.DataFrame) -> float:
    started = time.perf_counter()
    model.fit(frame)
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--training-sample", default="medium")
    parser.add_argument("--budget", type=float, default=0.20)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    seed = int(config["project"]["seed"])
    bootstrap_samples = int(config["evaluation"]["bootstrap_samples"])

    evaluation = pd.read_csv("data/processed/rct_evaluation.csv")
    observational_paths = {
        name: Path("data/observational") / f"{name}.csv"
        for name in config["confounding"]["strengths"]
    }
    observational = {name: pd.read_csv(path) for name, path in observational_paths.items()}
    training = observational[args.training_sample]

    train, development = train_test_split(
        training,
        test_size=0.25,
        random_state=seed,
        stratify=training["conversion"],
    )
    response_dev = ResponseModel(seed=seed).fit(train)
    treatment_dev = TreatmentFeatureModel(seed=seed).fit(train)
    response_dev_metrics = classification_metrics(
        development["conversion"], response_dev.predict(development)
    )
    treatment_dev_probability = treatment_dev.pipeline.predict_proba(development)[:, 1]
    treatment_dev_metrics = classification_metrics(
        development["conversion"], treatment_dev_probability
    )

    response = ResponseModel(seed=seed)
    treatment_feature = TreatmentFeatureModel(seed=seed)
    response_runtime = _timed_fit(response, training)
    treatment_runtime = _timed_fit(treatment_feature, training)
    response_score = response.predict(evaluation)
    pseudo_uplift_score = treatment_feature.predict_uplift(evaluation)
    random_score = random_policy_scores(evaluation, seed=seed)

    masks = {
        "random": top_fraction_mask(random_score, args.budget),
        "response_model": top_fraction_mask(response_score, args.budget),
        "treatment_feature_pseudo_uplift": top_fraction_mask(pseudo_uplift_score, args.budget),
    }
    policies = {
        name: evaluate_binary_policy(
            evaluation,
            mask,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        for name, mask in masks.items()
    }
    policy_differences = {
        "response_minus_random": bootstrap_policy_difference(
            evaluation,
            masks["response_model"],
            masks["random"],
            samples=bootstrap_samples,
            seed=seed,
        ),
        "pseudo_uplift_minus_random": bootstrap_policy_difference(
            evaluation,
            masks["treatment_feature_pseudo_uplift"],
            masks["random"],
            samples=bootstrap_samples,
            seed=seed,
        ),
        "pseudo_uplift_minus_response": bootstrap_policy_difference(
            evaluation,
            masks["treatment_feature_pseudo_uplift"],
            masks["response_model"],
            samples=bootstrap_samples,
            seed=seed,
        ),
    }

    associations: dict[str, dict[str, dict[str, float]]] = {}
    for name, frame in observational.items():
        associations[name] = {}
        for outcome in OUTCOME_COLUMNS:
            estimate = bootstrap_difference_in_means(
                frame,
                outcome,
                samples=bootstrap_samples,
                seed=seed,
            )
            associations[name][outcome] = estimate

    rct_effects = {
        outcome: bootstrap_difference_in_means(
            evaluation,
            outcome,
            samples=bootstrap_samples,
            seed=seed,
        )
        for outcome in OUTCOME_COLUMNS
    }
    for sample in associations.values():
        for outcome, estimate in sample.items():
            estimate["absolute_error_vs_rct"] = abs(
                estimate["estimate"] - rct_effects[outcome]["estimate"]
            )

    response_only = masks["response_model"] & ~masks["treatment_feature_pseudo_uplift"]
    pseudo_only = masks["treatment_feature_pseudo_uplift"] & ~masks["response_model"]
    decisions_differ = masks["response_model"] != masks["treatment_feature_pseudo_uplift"]
    disagreement = {
        "different_decision_rows": int(decisions_differ.sum()),
        "decision_disagreement_fraction": float(decisions_differ.mean()),
        "response_only": evaluate_binary_policy(
            evaluation,
            response_only,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        "pseudo_uplift_only": evaluate_binary_policy(
            evaluation,
            pseudo_only,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
    }

    summary = {
        "seed": seed,
        "training_sample": args.training_sample,
        "campaign_budget": args.budget,
        "rct_effects": rct_effects,
        "naive_associations": associations,
        "model_development_metrics": {
            "response_model": response_dev_metrics,
            "treatment_feature_model": treatment_dev_metrics,
        },
        "fit_runtime_seconds": {
            "response_model": response_runtime,
            "treatment_feature_model": treatment_runtime,
        },
        "score_summaries": {
            "response_probability": {
                "mean": float(response_score.mean()),
                "standard_deviation": float(response_score.std()),
            },
            "pseudo_uplift": {
                "mean": float(pseudo_uplift_score.mean()),
                "standard_deviation": float(pseudo_uplift_score.std()),
                "minimum": float(pseudo_uplift_score.min()),
                "maximum": float(pseudo_uplift_score.max()),
            },
        },
        "policies": policies,
        "paired_policy_differences": policy_differences,
        "response_vs_pseudo_uplift": disagreement,
    }

    output_dir = Path("outputs/naive_baselines")
    result_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    scored = pd.DataFrame(
        {
            "source_row_id": evaluation["source_row_id"],
            "treatment": evaluation["treatment"],
            "conversion": evaluation["conversion"],
            "response_score": response_score,
            "pseudo_uplift_score": pseudo_uplift_score,
            **{f"target_{name}": mask.astype(int) for name, mask in masks.items()},
        }
    )
    scored.to_csv(output_dir / "rct_policy_scores.csv", index=False)
    payload = json.dumps(summary, indent=2) + "\n"
    (output_dir / "summary.json").write_text(payload)
    result_path = result_dir / "naive_baselines_summary.json"
    result_path.write_text(payload)
    print(payload)
    print(f"wrote {result_path}")


if __name__ == "__main__":
    main()
