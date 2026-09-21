from __future__ import annotations

import json
from pathlib import Path

METHODS = (
    {
        "label": "Naive association",
        "target": "Raw association",
        "ate": "naive",
        "ranking": None,
        "policy": None,
    },
    {
        "label": "Response model",
        "target": "Purchase probability",
        "ate": None,
        "ranking": "response_model",
        "policy": "response_model",
    },
    {
        "label": "Treatment-as-feature",
        "target": "Pseudo-uplift",
        "ate": None,
        "ranking": "pseudo_uplift",
        "policy": "pseudo_uplift",
    },
    {
        "label": "PSM segments",
        "target": "Matched ATT",
        "ate": "psm",
        "ranking": None,
        "policy": "psm_segments",
    },
    {
        "label": "LinearDML",
        "target": "Constant ATE",
        "ate": "linear_dml",
        "ranking": None,
        "policy": "linear_dml",
    },
    {
        "label": "CausalForestDML",
        "target": "CATE",
        "ate": "causal_forest_dml",
        "ranking": "causal_forest_dml",
        "policy": "causal_forest_dml",
    },
    {
        "label": "Uplift tree",
        "target": "CATE ranking",
        "ate": None,
        "ranking": "uplift_tree",
        "policy": "uplift_tree",
    },
    {
        "label": "Uplift random forest",
        "target": "CATE ranking",
        "ate": "uplift_random_forest",
        "ranking": "uplift_random_forest",
        "policy": "uplift_random_forest",
    },
)


def load_readme_results(repository: Path) -> dict:
    result_dir = repository / "experiments" / "results"
    filenames = {
        "ablation": "confounding_ablation_summary.json",
        "business": "business_policy_summary.json",
        "uplift": "uplift_metrics_summary.json",
    }
    return {
        name: json.loads((result_dir / filename).read_text())
        for name, filename in filenames.items()
    }


def _currency(value: float) -> str:
    rounded = int(round(value))
    return f"-${abs(rounded):,}" if rounded < 0 else f"${rounded:,}"


def _profit_cell(record: dict) -> str:
    profit = record["incremental_profit_per_1000_eligible"]
    return (
        f"{_currency(profit['estimate'])} "
        f"[{_currency(profit['ci_lower'])}, {_currency(profit['ci_upper'])}]"
    )


def _qini_cell(record: dict) -> str:
    return (
        f"{record['estimate_per_1000']:.3f} "
        f"[{record['ci_lower_per_1000']:.3f}, {record['ci_upper_per_1000']:.3f}]"
    )


def comparison_markdown(results: dict) -> str:
    medium = results["ablation"]["samples"]["medium"]["ate_estimators"]
    policies = results["business"]["headline_20_percent"]
    rankings = results["uplift"]["models"]
    lines = [
        "<!-- BEGIN GENERATED FINAL COMPARISON -->",
        "| Method | What it estimates | ATE error / 1,000 | Qini / 1,000 (95% CI) | "
        "Profit / 1,000 eligible (95% CI) |",
        "|---|---|---:|---:|---:|",
    ]
    for method in METHODS:
        if method["ate"] == "psm":
            ate = "ATT, not ATE"
        elif method["ate"] is None:
            ate = "—"
        else:
            error = medium[method["ate"]]["absolute_error_vs_rct_ate"]
            ate = f"{error * 1000:.2f}"
        qini = "—" if method["ranking"] is None else _qini_cell(rankings[method["ranking"]]["qini"])
        profit = "—" if method["policy"] is None else _profit_cell(policies[method["policy"]])
        lines.append(f"| {method['label']} | {method['target']} | {ate} | {qini} | {profit} |")
    lines.extend(
        [
            "",
            "*Setting: medium-confounding training data, 20% targeting, $0.05/email, and the "
            "untouched randomized holdout. PSM remains ATT rather than being mislabeled as ATE. "
            "Every Qini interval and every paired profit difference versus random and response "
            "includes zero, so the table does not declare a targeting winner.*",
            "<!-- END GENERATED FINAL COMPARISON -->",
        ]
    )
    return "\n".join(lines)


def replace_generated_comparison(readme: Path, markdown: str) -> None:
    text = readme.read_text()
    start = "<!-- BEGIN GENERATED FINAL COMPARISON -->"
    end = "<!-- END GENERATED FINAL COMPARISON -->"
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError("README must contain exactly one generated comparison block")
    prefix, remainder = text.split(start, maxsplit=1)
    _, suffix = remainder.split(end, maxsplit=1)
    readme.write_text(f"{prefix}{markdown}{suffix}")


def plot_headline_summary(results: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    matplotlib.rcParams["svg.hashsalt"] = "causal-uplift-readme"
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    figure.patch.set_facecolor("#fbfaf6")
    colors = {
        "naive": "#c44e52",
        "linear_dml": "#4c72b0",
        "causal_forest_dml": "#55a868",
        "uplift_random_forest": "#8172b3",
    }

    strengths = results["ablation"]["strengths"]
    x = np.arange(len(strengths))
    labels = {
        "naive": "Naive association",
        "linear_dml": "LinearDML",
        "causal_forest_dml": "CausalForestDML",
        "uplift_random_forest": "Uplift random forest",
    }
    axis = axes[0]
    for method, label in labels.items():
        errors = [
            results["ablation"]["samples"][strength]["ate_estimators"][method][
                "absolute_error_vs_rct_ate"
            ]
            * 1000
            for strength in strengths
        ]
        axis.plot(x, errors, marker="o", linewidth=2.2, label=label, color=colors[method])
    axis.set_xticks(x, ["RCT", "Weak", "Medium", "Strong"])
    axis.set_ylabel("Absolute ATE error / 1,000")
    axis.set_title("Causal adjustment protects average-effect estimates", loc="left")
    axis.grid(axis="y", color="#dddddd", linewidth=0.8)
    axis.legend(frameon=False, fontsize=9)

    policy_names = [
        "response_model",
        "psm_segments",
        "causal_forest_dml",
        "uplift_random_forest",
    ]
    policy_labels = ["Response", "PSM segments", "Causal forest", "Uplift forest"]
    records = [results["business"]["headline_20_percent"][name] for name in policy_names]
    estimates = np.array(
        [record["incremental_profit_per_1000_eligible"]["estimate"] for record in records]
    )
    lower = estimates - np.array(
        [record["incremental_profit_per_1000_eligible"]["ci_lower"] for record in records]
    )
    upper = (
        np.array([record["incremental_profit_per_1000_eligible"]["ci_upper"] for record in records])
        - estimates
    )
    positions = np.arange(len(policy_names))
    axis = axes[1]
    axis.errorbar(
        estimates,
        positions,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        color="#4c72b0",
        ecolor="#8aa4c2",
        capsize=4,
        markersize=7,
    )
    axis.axvline(0, color="#555555", linewidth=1)
    axis.set_yticks(positions, policy_labels)
    axis.invert_yaxis()
    axis.set_xlabel("Incremental profit / 1,000 eligible customers ($)")
    axis.set_title("Individualized policy value remains uncertain", loc="left")
    axis.grid(axis="x", color="#dddddd", linewidth=0.8)
    axis.text(
        0.02,
        -0.22,
        "20% target · $0.05/email · all paired differences vs random/response include zero",
        transform=axis.transAxes,
        fontsize=9,
        color="#555555",
    )

    for axis in axes:
        axis.set_facecolor("#fbfaf6")
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Average effects recovered; customer rankings still required randomized validation",
        fontsize=16,
        fontweight="bold",
        x=0.06,
        ha="left",
    )
    figure.subplots_adjust(top=0.82, bottom=0.22, left=0.08, right=0.98, wspace=0.33)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="svg", metadata={"Date": None}, facecolor=figure.get_facecolor())
    figure.savefig(output.with_suffix(".png"), dpi=170, facecolor=figure.get_facecolor())
    plt.close(figure)
