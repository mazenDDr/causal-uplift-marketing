from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from causal_uplift.evaluation.runner import (
    ExperimentStep,
    atomic_write_json,
    can_resume,
    completed_step_fingerprint,
    hash_files,
    step_fingerprint,
    topological_steps,
    validate_steps,
)


def python_step(
    name: str,
    script: str,
    *,
    dependencies: tuple[str, ...],
    outputs: tuple[str, ...],
) -> ExperimentStep:
    return ExperimentStep(name, script, dependencies, outputs)


STEPS = (
    python_step(
        "foundation",
        "scripts/run_foundation.py",
        dependencies=(),
        outputs=(
            "experiments/results/foundation_summary.json",
            "data/processed/randomized_training_pool.csv",
            "data/processed/rct_evaluation.csv",
            "data/observational/randomized.csv",
            "data/observational/weak.csv",
            "data/observational/medium.csv",
            "data/observational/strong.csv",
        ),
    ),
    python_step(
        "naive_baselines",
        "scripts/run_naive_baselines.py",
        dependencies=("foundation",),
        outputs=(
            "experiments/results/naive_baselines_summary.json",
            "outputs/naive_baselines/rct_policy_scores.csv",
        ),
    ),
    python_step(
        "propensity_diagnostics",
        "scripts/run_propensity_diagnostics.py",
        dependencies=("foundation",),
        outputs=(
            "experiments/results/propensity_diagnostics_summary.json",
            "experiments/figures/propensity_overlap.svg",
            "experiments/figures/pre_matching_love_plot.svg",
            *tuple(
                f"outputs/propensity/{strength}_scores.csv"
                for strength in ("randomized", "weak", "medium", "strong")
            ),
        ),
    ),
    python_step(
        "matching",
        "scripts/run_matching.py",
        dependencies=("foundation",),
        outputs=(
            "experiments/results/matching_summary.json",
            "experiments/figures/matching_balance.svg",
            *tuple(
                f"outputs/matching/{strength}_{variant}_pairs.csv"
                for strength in ("randomized", "weak", "medium", "strong")
                for variant in ("with_replacement", "without_replacement")
            ),
        ),
    ),
    python_step(
        "linear_dml",
        "scripts/run_linear_dml.py",
        dependencies=("foundation",),
        outputs=(
            "experiments/results/linear_dml_summary.json",
            "experiments/figures/dml_ate_error.svg",
        ),
    ),
    python_step(
        "causal_forest",
        "scripts/run_causal_forest.py",
        dependencies=("linear_dml",),
        outputs=(
            "experiments/results/causal_forest_summary.json",
            "experiments/figures/causal_forest_validation.svg",
            "outputs/causal_forest/rct_cate_scores.csv",
        ),
    ),
    python_step(
        "uplift_models",
        "scripts/run_uplift_models.py",
        dependencies=("foundation",),
        outputs=(
            "experiments/results/uplift_models_summary.json",
            "experiments/figures/uplift_tree.svg",
            "experiments/figures/uplift_model_validation.svg",
            "outputs/uplift_models/rct_uplift_scores.csv",
        ),
    ),
    python_step(
        "uplift_metrics",
        "scripts/run_uplift_metrics.py",
        dependencies=("naive_baselines", "causal_forest", "uplift_models"),
        outputs=(
            "experiments/results/uplift_metrics_summary.json",
            "experiments/figures/uplift_qini_curves.svg",
            "experiments/figures/uplift_metric_intervals.svg",
        ),
    ),
    python_step(
        "business_policies",
        "scripts/run_business_policies.py",
        dependencies=("matching", "linear_dml", "uplift_metrics"),
        outputs=(
            "experiments/results/business_policy_summary.json",
            "experiments/figures/policy_profit_curves.svg",
            "experiments/figures/policy_20pct_profit.svg",
            "experiments/figures/prediction_vs_uplift_policy.svg",
        ),
    ),
    python_step(
        "confounding_ablation",
        "scripts/run_confounding_ablation.py",
        dependencies=("matching", "linear_dml"),
        outputs=(
            "experiments/results/confounding_ablation_summary.json",
            "experiments/figures/confounding_ate_ablation.svg",
            "experiments/figures/confounding_policy_ablation.svg",
            "outputs/confounding_ablation/rct_scores.csv",
        ),
    ),
    python_step(
        "overlap_stress",
        "scripts/run_overlap_stress.py",
        dependencies=("foundation",),
        outputs=(
            "experiments/results/overlap_stress_summary.json",
            "experiments/figures/overlap_stress_diagnostics.svg",
            "experiments/figures/overlap_estimate_stability.svg",
        ),
    ),
    python_step(
        "robustness",
        "scripts/run_robustness.py",
        dependencies=("foundation",),
        outputs=(
            "experiments/results/robustness_summary.json",
            "experiments/figures/robustness_dml.svg",
            "experiments/figures/robustness_matching.svg",
        ),
    ),
    python_step(
        "failure_analysis",
        "scripts/build_failure_analysis.py",
        dependencies=(
            "confounding_ablation",
            "overlap_stress",
            "uplift_models",
            "uplift_metrics",
        ),
        outputs=(
            "experiments/results/failure_analysis_summary.json",
            "experiments/figures/failure_taxonomy.svg",
        ),
    ),
    python_step(
        "womens_replication",
        "scripts/run_womens_replication.py",
        dependencies=("linear_dml", "confounding_ablation", "uplift_metrics"),
        outputs=(
            "experiments/results/womens_replication_summary.json",
            "experiments/figures/womens_replication.svg",
            "outputs/womens_replication/rct_scores.csv",
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the causal-uplift experiment DAG")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument(
        "--steps",
        default="all",
        help="comma-separated terminal steps, including their dependencies; default: all",
    )
    parser.add_argument("--force", action="store_true", help="rerun even resumable steps")
    parser.add_argument("--dry-run", action="store_true", help="print the resolved DAG and stop")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/experiment_runner/manifest.json"),
    )
    return parser.parse_args()


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"schema_version": 1, "steps": {}, "invocations": []}
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("steps"), dict):
        raise ValueError(f"unsupported experiment manifest: {path}")
    manifest.setdefault("invocations", [])
    return manifest


def package_versions() -> dict[str, str]:
    versions = {}
    for package in ("causalml", "econml", "numpy", "pandas", "scikit-learn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def public_summary(
    manifest: dict,
    invocation: dict,
    *,
    code_fingerprint: str,
    input_fingerprint: str,
) -> dict:
    selected = invocation["resolved_steps"]
    successful_runtime = sum(
        manifest["steps"][name]["runtime_seconds"]
        for name in selected
        if manifest["steps"][name]["status"] == "succeeded"
    )
    return {
        "schema_version": 1,
        "purpose": "Resumable, fingerprinted execution audit for the complete experiment DAG.",
        "code_fingerprint": code_fingerprint,
        "input_fingerprint": input_fingerprint,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": package_versions(),
        },
        "latest_invocation": invocation,
        "recent_invocations": manifest["invocations"][-2:],
        "successful_step_runtime_total_seconds": successful_runtime,
        "steps": {name: manifest["steps"][name] for name in selected},
        "resume_rule": (
            "Skip only after prior success with the same code/config/upstream fingerprint and "
            "matching byte hashes for every declared output."
        ),
    }


def main() -> None:
    args = parse_args()
    repository = Path.cwd().resolve()
    validate_steps(STEPS)
    requested = [step.name for step in STEPS]
    if args.steps != "all":
        requested = [name.strip() for name in args.steps.split(",") if name.strip()]
    resolved = topological_steps(STEPS, requested)
    if args.dry_run:
        for index, step in enumerate(resolved, start=1):
            dependencies = ", ".join(step.dependencies) or "none"
            print(f"{index:02d}. {step.name} <- {dependencies}")
        return

    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    raw_data = Path("data/raw/hillstrom.csv")
    if not raw_data.is_file():
        raise FileNotFoundError(
            "data/raw/hillstrom.csv is missing; run `python scripts/download_hillstrom.py` first"
        )

    code_paths = [Path("pyproject.toml"), *Path("src").rglob("*.py"), *Path("scripts").glob("*.py")]
    code_fingerprint = hash_files(code_paths, root=repository)
    input_fingerprint = hash_files([args.config, raw_data], root=repository)
    manifest = load_manifest(args.manifest)
    invocation = {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "requested_steps": requested,
        "resolved_steps": [step.name for step in resolved],
        "force": args.force,
        "actions": [],
    }
    manifest["invocations"].append(invocation)
    manifest["invocations"] = manifest["invocations"][-20:]
    dependency_fingerprints: dict[str, str] = {}
    invocation_started = time.perf_counter()
    log_dir = args.manifest.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    for position, step in enumerate(resolved, start=1):
        fingerprint = step_fingerprint(
            step,
            code_fingerprint=code_fingerprint,
            config_fingerprint=input_fingerprint,
            dependency_fingerprints=dependency_fingerprints,
        )
        outputs = [repository / path for path in step.outputs]
        previous = manifest["steps"].get(step.name)
        if not args.force and can_resume(
            previous,
            fingerprint=fingerprint,
            outputs=outputs,
            root=repository,
        ):
            print(f"[{position:02d}/{len(resolved):02d}] skip {step.name}: fingerprint unchanged")
            invocation["actions"].append({"step": step.name, "action": "skipped"})
            dependency_fingerprints[step.name] = completed_step_fingerprint(
                fingerprint, previous["output_fingerprint"]
            )
            continue

        command = [sys.executable, step.script]
        if step.script != "scripts/build_failure_analysis.py":
            command.extend(["--config", str(args.config)])
        log_path = log_dir / f"{step.name}.log"
        print(f"[{position:02d}/{len(resolved):02d}] run  {step.name}")
        step_started = time.perf_counter()
        record = {
            "status": "running",
            "fingerprint": fingerprint,
            "command": command,
            "dependencies": list(step.dependencies),
            "outputs": list(step.outputs),
            "log": str(log_path),
            "started_at_utc": datetime.now(UTC).isoformat(),
        }
        manifest["steps"][step.name] = record
        atomic_write_json(args.manifest, manifest)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = "src"
        with log_path.open("w") as log:
            completed = subprocess.run(
                command,
                cwd=repository,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        runtime = time.perf_counter() - step_started
        record.update(
            status="succeeded" if completed.returncode == 0 else "failed",
            returncode=completed.returncode,
            runtime_seconds=runtime,
            finished_at_utc=datetime.now(UTC).isoformat(),
        )
        missing_outputs = [
            str(path.relative_to(repository)) for path in outputs if not path.is_file()
        ]
        if not missing_outputs:
            record["output_fingerprint"] = hash_files(outputs, root=repository)
        record["missing_outputs"] = missing_outputs
        invocation["actions"].append(
            {"step": step.name, "action": "ran", "runtime_seconds": runtime}
        )
        atomic_write_json(args.manifest, manifest)
        if completed.returncode or missing_outputs:
            record["status"] = "failed"
            atomic_write_json(args.manifest, manifest)
            tail = "\n".join(log_path.read_text().splitlines()[-30:])
            raise RuntimeError(f"{step.name} failed; log tail:\n{tail}")
        dependency_fingerprints[step.name] = completed_step_fingerprint(
            fingerprint, record["output_fingerprint"]
        )

    invocation["finished_at_utc"] = datetime.now(UTC).isoformat()
    invocation["runtime_seconds"] = time.perf_counter() - invocation_started
    invocation["ran_steps"] = sum(action["action"] == "ran" for action in invocation["actions"])
    invocation["skipped_steps"] = sum(
        action["action"] == "skipped" for action in invocation["actions"]
    )
    atomic_write_json(args.manifest, manifest)
    result_path = Path("experiments/results/experiment_runner_summary.json")
    atomic_write_json(
        result_path,
        public_summary(
            manifest,
            invocation,
            code_fingerprint=code_fingerprint,
            input_fingerprint=input_fingerprint,
        ),
    )
    print(
        f"completed {len(resolved)} steps: {invocation['ran_steps']} ran, "
        f"{invocation['skipped_steps']} resumed in {invocation['runtime_seconds']:.2f}s"
    )
    print(f"wrote {args.manifest} and {result_path}")


if __name__ == "__main__":
    main()
