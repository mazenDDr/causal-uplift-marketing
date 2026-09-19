from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from causal_uplift.data.confounding import ConfoundingConfig, sample_observational
from causal_uplift.data.load import OUTCOME_COLUMNS, campaign_frame, load_hillstrom
from causal_uplift.data.preprocess import randomized_train_evaluation_split
from causal_uplift.evaluation.ate import bootstrap_difference_in_means
from causal_uplift.evaluation.balance import covariate_smds

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
    data_config = config["data"]
    confounding_config = config["confounding"]
    raw_path = Path(data_config["raw_path"])
    metadata_path = raw_path.with_suffix(".metadata.json")
    frame = campaign_frame(
        load_hillstrom(raw_path),
        treatment_label=data_config["campaign"],
        control_label=data_config["control"],
    )
    training_pool, evaluation = randomized_train_evaluation_split(
        frame,
        evaluation_fraction=float(config["project"]["evaluation_fraction"]),
        seed=seed,
    )

    processed_dir = Path("data/processed")
    observational_dir = Path("data/observational")
    output_dir = Path("outputs/foundation")
    results_dir = Path("experiments/results")
    for directory in (processed_dir, observational_dir, output_dir, results_dir):
        directory.mkdir(parents=True, exist_ok=True)
    training_pool.to_csv(processed_dir / "randomized_training_pool.csv", index=False)
    evaluation.to_csv(processed_dir / "rct_evaluation.csv", index=False)

    coefficients = confounding_config["coefficients"]
    summary: dict[str, object] = {
        "seed": seed,
        "campaign": data_config["campaign"],
        "control": data_config["control"],
        "dataset": json.loads(metadata_path.read_text()) if metadata_path.exists() else None,
        "rows": {
            "campaign_total": len(frame),
            "randomized_training_pool": len(training_pool),
            "rct_evaluation": len(evaluation),
        },
        "rct_effects": {},
        "observational_samples": {},
    }
    for outcome in OUTCOME_COLUMNS:
        summary["rct_effects"][outcome] = bootstrap_difference_in_means(
            evaluation,
            outcome,
            samples=int(config["evaluation"]["bootstrap_samples"]),
            seed=seed,
        )

    for offset, (name, strength) in enumerate(confounding_config["strengths"].items()):
        observational = sample_observational(
            training_pool,
            ConfoundingConfig(
                strength=float(strength),
                clip_min=float(confounding_config["clip_min"]),
                clip_max=float(confounding_config["clip_max"]),
                **{key: float(value) for key, value in coefficients.items()},
            ),
            seed=seed + offset,
        )
        observational.to_csv(observational_dir / f"{name}.csv", index=False)
        smds = covariate_smds(observational, BALANCE_COLUMNS)
        raw_conversion = bootstrap_difference_in_means(
            observational,
            "conversion",
            samples=int(config["evaluation"]["bootstrap_samples"]),
            seed=seed,
        )
        summary["observational_samples"][name] = {
            "strength": float(strength),
            "rows": len(observational),
            "treated_share": float(observational["treatment"].mean()),
            "max_absolute_smd": max(abs(value) for value in smds.values()),
            "smd": smds,
            "raw_conversion_association": raw_conversion,
            "raw_conversion_ate_error": abs(
                raw_conversion["estimate"] - summary["rct_effects"]["conversion"]["estimate"]
            ),
        }

    output_path = output_dir / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    result_path = results_dir / "foundation_summary.json"
    result_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {output_path} and {result_path}")


if __name__ == "__main__":
    main()
