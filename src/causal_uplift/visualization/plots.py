from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save_figure(figure: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    if output.suffix == ".svg":
        normalized = "\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n"
        output.write_text(normalized)
    figure.savefig(output.with_suffix(".png"), dpi=160, bbox_inches="tight")


def plot_propensity_overlap(
    frames: dict[str, pd.DataFrame],
    scores: dict[str, np.ndarray],
    output: Path,
    *,
    overlap_min: float = 0.05,
    overlap_max: float = 0.95,
) -> None:
    """Plot treated/control propensity distributions for each stress level."""
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    bins = np.linspace(0, 1, 31)
    for axis, (name, frame) in zip(axes.flat, frames.items(), strict=True):
        score = scores[name]
        treatment = frame["treatment"].to_numpy(dtype=int)
        axis.hist(
            score[treatment == 0],
            bins=bins,
            density=True,
            alpha=0.55,
            label="Control",
            color="#2c7fb8",
        )
        axis.hist(
            score[treatment == 1],
            bins=bins,
            density=True,
            alpha=0.55,
            label="Men's email",
            color="#d95f0e",
        )
        axis.axvline(overlap_min, color="#555555", linestyle="--", linewidth=1)
        axis.axvline(overlap_max, color="#555555", linestyle="--", linewidth=1)
        axis.set_title(name.capitalize())
        axis.set_ylabel("Density")
        axis.grid(axis="y", alpha=0.2)
    axes[-1, 0].set_xlabel("Estimated treatment propensity")
    axes[-1, 1].set_xlabel("Estimated treatment propensity")
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Treatment assignment becomes more predictable as confounding increases")
    figure.tight_layout()
    _save_figure(figure, output)
    plt.close(figure)


def plot_love_by_strength(smds: dict[str, dict[str, float]], output: Path) -> None:
    """Plot absolute pre-matching SMDs for every confounding level."""
    features = list(next(iter(smds.values())))
    figure_height = max(6.0, len(features) * 0.48)
    figure, axis = plt.subplots(figsize=(11, figure_height))
    positions = np.arange(len(features))
    offsets = np.linspace(-0.24, 0.24, len(smds))
    colors = ["#4d4d4d", "#74a9cf", "#fdae6b", "#d7301f"]
    for offset, color, (name, values) in zip(offsets, colors, smds.items(), strict=True):
        axis.scatter(
            [abs(values[feature]) for feature in features],
            positions + offset,
            s=42,
            label=name.capitalize(),
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
    axis.axvline(0.10, color="#8c2d04", linestyle="--", linewidth=1.5, label="|SMD| = 0.10")
    axis.set_yticks(positions, labels=features)
    axis.invert_yaxis()
    axis.set_xlabel("Absolute standardized mean difference")
    axis.set_title("Covariate imbalance before matching")
    axis.grid(axis="x", alpha=0.2)
    axis.legend(frameon=False, ncol=3, loc="lower right")
    figure.tight_layout()
    _save_figure(figure, output)
    plt.close(figure)


def plot_matching_balance(
    before: dict[str, dict[str, float]],
    after: dict[str, dict[str, float]],
    output: Path,
) -> None:
    """Compare absolute SMDs before and after preferred matching by stress level."""
    figure, axes = plt.subplots(2, 2, figsize=(13, 12), sharex=True)
    for axis, name in zip(axes.flat, before, strict=True):
        features = list(before[name])
        positions = np.arange(len(features))
        before_values = [abs(before[name][feature]) for feature in features]
        after_values = [abs(after[name][feature]) for feature in features]
        for position, start, end in zip(positions, before_values, after_values, strict=True):
            axis.plot([start, end], [position, position], color="#bdbdbd", linewidth=1.2)
        axis.scatter(
            before_values,
            positions,
            color="#d95f0e",
            label="Before",
            s=38,
            zorder=3,
        )
        axis.scatter(
            after_values,
            positions,
            color="#2c7fb8",
            label="After",
            s=38,
            zorder=3,
        )
        axis.axvline(0.10, color="#8c2d04", linestyle="--", linewidth=1.2)
        axis.set_yticks(positions, labels=features)
        axis.invert_yaxis()
        axis.set_title(name.capitalize())
        axis.grid(axis="x", alpha=0.2)
    axes[-1, 0].set_xlabel("Absolute standardized mean difference")
    axes[-1, 1].set_xlabel("Absolute standardized mean difference")
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Balance before and after the preferred propensity-score match")
    figure.tight_layout()
    _save_figure(figure, output)
    plt.close(figure)


def plot_ate_error_by_strength(
    errors: dict[str, dict[str, float]],
    output: Path,
) -> None:
    """Plot absolute conversion ATE error for naive and DML estimators."""
    strengths = list(next(iter(errors.values())))
    positions = np.arange(len(strengths))
    figure, axis = plt.subplots(figsize=(10, 6))
    colors = ["#d95f0e", "#2c7fb8", "#41ab5d"]
    markers = ["o", "s", "^"]
    for color, marker, (name, values) in zip(colors, markers, errors.items(), strict=True):
        axis.plot(
            positions,
            [values[strength] for strength in strengths],
            marker=marker,
            linewidth=2,
            markersize=7,
            color=color,
            label=name.replace("_", " ").title(),
        )
    axis.set_xticks(positions, labels=[strength.capitalize() for strength in strengths])
    axis.set_ylabel("Absolute error vs randomized ATE")
    axis.set_xlabel("Training-data confounding")
    axis.set_title("Cross-fitted LinearDML corrects measured selection bias")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    _save_figure(figure, output)
    plt.close(figure)


def plot_causal_forest_validation(
    deciles: list[dict[str, object]],
    subgroups: list[dict[str, object]],
    output: Path,
) -> None:
    """Plot predicted CATE against randomized effects by rank and customer segment."""
    figure, axes = plt.subplots(1, 2, figsize=(15, 6))

    decile_axis = axes[0]
    positions = np.arange(1, len(deciles) + 1)
    predicted = np.array([row["predicted_cate_mean"] for row in deciles], dtype=float)
    observed = np.array([row["observed_rct_uplift"] for row in deciles], dtype=float)
    lower = np.array([row["observed_ci_lower"] for row in deciles], dtype=float)
    upper = np.array([row["observed_ci_upper"] for row in deciles], dtype=float)
    decile_axis.plot(
        positions,
        predicted,
        marker="o",
        color="#d95f0e",
        linewidth=2,
        label="Predicted CATE",
    )
    decile_axis.errorbar(
        positions,
        observed,
        yerr=np.vstack([observed - lower, upper - observed]),
        marker="s",
        color="#2c7fb8",
        linewidth=1.5,
        capsize=3,
        label="Observed RCT uplift (95% CI)",
    )
    decile_axis.axhline(0, color="#555555", linewidth=1)
    decile_axis.set_xticks(positions)
    decile_axis.set_xlabel("Predicted CATE decile (low to high)")
    decile_axis.set_ylabel("Conversion effect")
    decile_axis.set_title("Does CATE ranking survive randomized evaluation?")
    decile_axis.grid(axis="y", alpha=0.2)
    decile_axis.legend(frameon=False)

    subgroup_axis = axes[1]
    labels = [str(row["group"]) for row in subgroups]
    predicted = np.array([row["predicted_cate_mean"] for row in subgroups], dtype=float)
    observed = np.array([row["observed_rct_uplift"] for row in subgroups], dtype=float)
    lower = np.array([row["observed_ci_lower"] for row in subgroups], dtype=float)
    upper = np.array([row["observed_ci_upper"] for row in subgroups], dtype=float)
    positions = np.arange(len(labels))
    subgroup_axis.bar(
        positions - 0.18,
        predicted,
        width=0.36,
        color="#fdae6b",
        label="Predicted CATE",
    )
    subgroup_axis.errorbar(
        positions + 0.18,
        observed,
        yerr=np.vstack([observed - lower, upper - observed]),
        fmt="s",
        color="#2c7fb8",
        capsize=3,
        label="Observed RCT uplift (95% CI)",
    )
    subgroup_axis.axhline(0, color="#555555", linewidth=1)
    subgroup_axis.set_xticks(positions, labels=labels, rotation=18, ha="right")
    subgroup_axis.set_ylabel("Conversion effect")
    subgroup_axis.set_title("Predefined customer segments")
    subgroup_axis.grid(axis="y", alpha=0.2)
    subgroup_axis.legend(frameon=False)

    figure.suptitle("Causal-forest heterogeneity checked against the untouched RCT")
    figure.tight_layout()
    _save_figure(figure, output)
    plt.close(figure)


def plot_uplift_model_validation(
    model_deciles: dict[str, list[dict[str, object]]],
    output: Path,
) -> None:
    """Compare uplift-tree and uplift-forest ranking on randomized deciles."""
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for axis, (name, deciles) in zip(axes, model_deciles.items(), strict=True):
        positions = np.arange(1, len(deciles) + 1)
        predicted = np.array([row["predicted_cate_mean"] for row in deciles], dtype=float)
        observed = np.array([row["observed_rct_uplift"] for row in deciles], dtype=float)
        lower = np.array([row["observed_ci_lower"] for row in deciles], dtype=float)
        upper = np.array([row["observed_ci_upper"] for row in deciles], dtype=float)
        axis.plot(
            positions,
            predicted,
            marker="o",
            linewidth=2,
            color="#d95f0e",
            label="Predicted uplift",
        )
        axis.errorbar(
            positions,
            observed,
            yerr=np.vstack([observed - lower, upper - observed]),
            marker="s",
            linewidth=1.5,
            capsize=3,
            color="#2c7fb8",
            label="Observed RCT uplift (95% CI)",
        )
        axis.axhline(0, color="#555555", linewidth=1)
        axis.set_xticks(positions)
        axis.set_xlabel("Predicted uplift decile (low to high)")
        axis.set_title(name.replace("_", " ").title())
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Conversion effect")
    axes[0].legend(frameon=False)
    figure.suptitle("Uplift models checked against the untouched randomized holdout")
    figure.tight_layout()
    _save_figure(figure, output)
    plt.close(figure)


def plot_uplift_qini_curves(models: dict[str, dict[str, object]], output: Path) -> None:
    """Plot randomized cumulative gain and Qini curves for every targeting score."""
    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    colors = ["#636363", "#756bb1", "#31a354", "#e6550d", "#3182bd"]
    labels = {
        "response_model": "Response model",
        "pseudo_uplift": "Treatment-as-feature",
        "causal_forest_dml": "CausalForestDML",
        "uplift_tree": "Uplift tree",
        "uplift_random_forest": "Uplift random forest",
    }
    first_curve = next(iter(models.values()))["curve"]
    fractions = np.asarray(first_curve["fraction"], dtype=float)
    axes[0].plot(
        fractions * 100,
        np.asarray(first_curve["random_gain"]) * 1000,
        color="#969696",
        linestyle="--",
        linewidth=1.5,
        label="Random targeting",
    )
    axes[1].axhline(0, color="#969696", linestyle="--", linewidth=1.5)
    for color, (name, result) in zip(colors, models.items(), strict=True):
        curve = result["curve"]
        axes[0].plot(
            np.asarray(curve["fraction"]) * 100,
            np.asarray(curve["gain"]) * 1000,
            color=color,
            linewidth=2,
            label=labels.get(name, name.replace("_", " ").title()),
        )
        axes[1].plot(
            np.asarray(curve["fraction"]) * 100,
            np.asarray(curve["qini_gain"]) * 1000,
            color=color,
            linewidth=2,
            label=labels.get(name, name.replace("_", " ").title()),
        )
    axes[0].set_title("Cumulative incremental conversions")
    axes[0].set_ylabel("Incremental conversions per 1,000 customers")
    axes[1].set_title("Qini gain over random targeting")
    axes[1].set_ylabel("Gain over random per 1,000 customers")
    for axis in axes:
        axis.set_xlabel("Customers targeted (%)")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("Targeting rankings evaluated only on the randomized holdout")
    figure.tight_layout()
    _save_figure(figure, output)
    plt.close(figure)


def plot_uplift_metric_intervals(models: dict[str, dict[str, object]], output: Path) -> None:
    """Plot treatment-stratified bootstrap intervals for AUUC and Qini."""
    display_names = {
        "response_model": "Response model",
        "pseudo_uplift": "Treatment-as-feature",
        "causal_forest_dml": "CausalForestDML",
        "uplift_tree": "Uplift tree",
        "uplift_random_forest": "Uplift random forest",
    }
    labels = [display_names.get(name, name.replace("_", " ").title()) for name in models]
    positions = np.arange(len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for axis, metric, title in zip(
        axes,
        ["qini", "auuc"],
        ["Qini coefficient", "Area under uplift curve"],
        strict=True,
    ):
        estimates = np.array([models[name][metric]["estimate"] for name in models]) * 1000
        lower = np.array([models[name][metric]["ci_lower"] for name in models]) * 1000
        upper = np.array([models[name][metric]["ci_upper"] for name in models]) * 1000
        axis.errorbar(
            estimates,
            positions,
            xerr=np.vstack([estimates - lower, upper - estimates]),
            fmt="o",
            color="#2c7fb8",
            capsize=4,
        )
        if metric == "qini":
            random_baseline = 0.0
        else:
            curve = next(iter(models.values()))["curve"]
            random_baseline = float(np.trapezoid(curve["random_gain"], curve["fraction"]) * 1000)
        axis.axvline(
            random_baseline,
            color="#555555",
            linestyle="--",
            linewidth=1,
            label="Random-targeting baseline",
        )
        axis.set_title(title)
        axis.set_xlabel("Metric × 1,000 (95% bootstrap CI)")
        axis.grid(axis="x", alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    axes[0].set_yticks(positions, labels=labels)
    axes[0].invert_yaxis()
    figure.suptitle("Ranking uncertainty on the untouched randomized holdout")
    figure.tight_layout()
    _save_figure(figure, output)
    plt.close(figure)


def plot_policy_profit_curves(
    top_k: dict[str, dict[str, object]],
    output: Path,
    *,
    primary_cost: float,
    primary_budget: float,
) -> None:
    """Plot profit across targeting budgets and email costs."""
    labels = {
        "random": "Random",
        "response_model": "Response model",
        "pseudo_uplift": "Treatment-as-feature",
        "psm_segments": "PSM segments",
        "linear_dml": "LinearDML (constant)",
        "causal_forest_dml": "CausalForestDML",
        "uplift_tree": "Uplift tree",
        "uplift_random_forest": "Uplift random forest",
    }
    colors = [
        "#969696",
        "#252525",
        "#756bb1",
        "#e7ba52",
        "#8c6d31",
        "#31a354",
        "#e6550d",
        "#3182bd",
    ]
    figure, axes = plt.subplots(1, 2, figsize=(16, 6))
    for color, (name, budgets) in zip(colors, top_k.items(), strict=True):
        ordered = sorted(budgets.values(), key=lambda row: row["targeted_fraction"])
        axes[0].plot(
            [row["targeted_fraction"] * 100 for row in ordered],
            [row["profit_by_email_cost"][f"{primary_cost:.2f}"]["estimate"] for row in ordered],
            marker="o",
            markersize=3,
            linewidth=1.7,
            color=color,
            label=labels[name],
        )
        primary = min(ordered, key=lambda row: abs(row["targeted_fraction"] - primary_budget))
        costs = sorted(float(value) for value in primary["profit_by_email_cost"])
        axes[1].plot(
            costs,
            [primary["profit_by_email_cost"][f"{cost:.2f}"]["estimate"] for cost in costs],
            marker="o",
            markersize=3,
            linewidth=1.7,
            color=color,
            label=labels[name],
        )
    axes[0].axhline(0, color="#555555", linewidth=1)
    axes[0].set_xlabel("Customers emailed (%)")
    axes[0].set_ylabel("Incremental profit per 1,000 eligible customers ($)")
    axes[0].set_title(f"Campaign-size curve at ${primary_cost:.2f} per email")
    axes[1].axhline(0, color="#555555", linewidth=1)
    axes[1].set_xlabel("Cost per email ($)")
    axes[1].set_ylabel("Incremental profit per 1,000 eligible customers ($)")
    axes[1].set_title(f"Cost sensitivity at {primary_budget:.0%} targeting")
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=7.5, ncol=2)
    figure.suptitle("Business policies evaluated on randomized conversion spend")
    figure.tight_layout()
    _save_figure(figure, output)
    plt.close(figure)


def plot_primary_policy_intervals(
    headline: dict[str, dict[str, object]],
    output: Path,
    *,
    budget: float,
    email_cost: float,
) -> None:
    """Plot primary-budget randomized profit intervals for every policy."""
    labels = [row["display_name"] for row in headline.values()]
    estimates = np.array(
        [row["incremental_profit_per_1000_eligible"]["estimate"] for row in headline.values()]
    )
    lower = np.array(
        [row["incremental_profit_per_1000_eligible"]["ci_lower"] for row in headline.values()]
    )
    upper = np.array(
        [row["incremental_profit_per_1000_eligible"]["ci_upper"] for row in headline.values()]
    )
    positions = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(11, 7))
    axis.errorbar(
        estimates,
        positions,
        xerr=np.vstack([estimates - lower, upper - estimates]),
        fmt="o",
        color="#2c7fb8",
        capsize=4,
    )
    axis.axvline(0, color="#555555", linewidth=1)
    axis.set_yticks(positions, labels=labels)
    axis.invert_yaxis()
    axis.set_xlabel("Incremental profit per 1,000 eligible customers ($), 95% CI")
    axis.set_title(f"Randomized policy value at {budget:.0%} targeting and ${email_cost:.2f}/email")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    _save_figure(figure, output)
    plt.close(figure)


def plot_prediction_vs_uplift_decision(
    response_score: np.ndarray,
    uplift_score: np.ndarray,
    response_selected: np.ndarray,
    uplift_selected: np.ndarray,
    response_result: dict[str, object],
    uplift_result: dict[str, object],
    paired_profit: dict[str, float],
    output: Path,
    *,
    budget: float,
) -> None:
    """Show where predictive and causal targeting disagree and the randomized consequence."""
    response_selected = np.asarray(response_selected, dtype=bool)
    uplift_selected = np.asarray(uplift_selected, dtype=bool)
    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    scatter_axis = axes[0]
    categories = [
        ("Neither", ~response_selected & ~uplift_selected, "#bdbdbd", 0.12, 5),
        ("Both", response_selected & uplift_selected, "#41ab5d", 0.55, 11),
        ("Response only", response_selected & ~uplift_selected, "#d7301f", 0.65, 11),
        ("Uplift only", uplift_selected & ~response_selected, "#2c7fb8", 0.65, 11),
    ]
    for label, selected, color, alpha, size in categories:
        scatter_axis.scatter(
            uplift_score[selected],
            response_score[selected],
            color=color,
            alpha=alpha,
            s=size,
            linewidths=0,
            label=f"{label} ({selected.sum():,})",
            rasterized=True,
        )
    scatter_axis.set_xlabel("Uplift-tree predicted conversion effect")
    scatter_axis.set_ylabel("Response-model predicted conversion probability")
    scatter_axis.set_title("Purchase likelihood and persuadability select different customers")
    scatter_axis.grid(alpha=0.15)
    scatter_axis.legend(frameon=False, fontsize=8)

    interval_axis = axes[1]
    results = [response_result, uplift_result]
    labels = ["Response model", "Uplift tree"]
    estimates = np.array(
        [row["incremental_profit_per_1000_eligible"]["estimate"] for row in results]
    )
    lower = np.array([row["incremental_profit_per_1000_eligible"]["ci_lower"] for row in results])
    upper = np.array([row["incremental_profit_per_1000_eligible"]["ci_upper"] for row in results])
    positions = np.arange(2)
    interval_axis.errorbar(
        estimates,
        positions,
        xerr=np.vstack([estimates - lower, upper - estimates]),
        fmt="o",
        color="#2c7fb8",
        capsize=4,
    )
    interval_axis.axvline(0, color="#555555", linewidth=1)
    interval_axis.set_yticks(positions, labels=labels)
    interval_axis.invert_yaxis()
    interval_axis.set_xlabel("Incremental profit per 1,000 eligible customers ($), 95% CI")
    interval_axis.set_title("Randomized business value")
    interval_axis.grid(axis="x", alpha=0.2)
    interval_axis.text(
        0.04,
        0.04,
        (
            f"Paired uplift-tree minus response:\n"
            f"${paired_profit['estimate']:.1f} "
            f"[${paired_profit['ci_lower']:.1f}, ${paired_profit['ci_upper']:.1f}]"
        ),
        transform=interval_axis.transAxes,
        fontsize=9,
        va="bottom",
    )
    figure.suptitle(f"Where predictive targeting makes the wrong decision at a {budget:.0%} budget")
    figure.tight_layout()
    _save_figure(figure, output)
    plt.close(figure)


def plot_confounding_ate_ablation(samples: dict[str, dict[str, object]], output: Path) -> None:
    """Plot like-for-like ATE errors and PSM's separately labeled ATT reference gap."""
    strengths = list(samples)
    positions = np.arange(len(strengths))
    methods = {
        "naive": ("Naive association", "#d95f0e", "o"),
        "linear_dml": ("LinearDML", "#2c7fb8", "s"),
        "causal_forest_dml": ("CausalForestDML", "#31a354", "^"),
        "uplift_random_forest": ("Uplift random forest", "#756bb1", "D"),
    }
    figure, axis = plt.subplots(figsize=(11, 6))
    for method, (label, color, marker) in methods.items():
        axis.plot(
            positions,
            [
                samples[strength]["ate_estimators"][method]["absolute_error_vs_rct_ate"]
                for strength in strengths
            ],
            marker=marker,
            color=color,
            linewidth=2,
            label=label,
        )
    axis.plot(
        positions,
        [
            abs(samples[strength]["ate_estimators"]["psm"]["reference_gap_vs_overall_rct_ate"])
            for strength in strengths
        ],
        marker="x",
        color="#636363",
        linestyle="--",
        linewidth=1.5,
        label="PSM ATT gap (different estimand)",
    )
    axis.set_xticks(positions, labels=[strength.capitalize() for strength in strengths])
    axis.set_xlabel("Training-data confounding")
    axis.set_ylabel("Absolute conversion-effect gap vs overall RCT ATE")
    axis.set_title("Estimator error under controlled treatment-selection bias")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    _save_figure(figure, output)
    plt.close(figure)


def plot_confounding_policy_ablation(
    samples: dict[str, dict[str, object]],
    output: Path,
    *,
    budget: float,
    email_cost: float,
) -> None:
    """Plot randomized policy value as training confounding increases."""
    strengths = list(samples)
    positions = np.arange(len(strengths))
    methods = {
        "random": ("Random", "#969696", "o"),
        "response_model": ("Response model", "#252525", "s"),
        "pseudo_uplift": ("Treatment-as-feature", "#756bb1", "^"),
        "psm_segments": ("PSM segments", "#e7ba52", "D"),
        "causal_forest_dml": ("CausalForestDML", "#31a354", "P"),
        "uplift_random_forest": ("Uplift random forest", "#3182bd", "X"),
    }
    figure, axes = plt.subplots(1, 2, figsize=(16, 6))
    for method, (label, color, marker) in methods.items():
        policy = [samples[strength]["policies"][method] for strength in strengths]
        profit = np.array(
            [row["incremental_profit_per_1000_eligible"]["estimate"] for row in policy]
        )
        profit_lower = np.array(
            [row["incremental_profit_per_1000_eligible"]["ci_lower"] for row in policy]
        )
        profit_upper = np.array(
            [row["incremental_profit_per_1000_eligible"]["ci_upper"] for row in policy]
        )
        conversion = np.array([row["conversion"]["effect_per_1000_emails"] for row in policy])
        conversion_lower = np.array([row["conversion"]["ci_lower"] * 1000 for row in policy])
        conversion_upper = np.array([row["conversion"]["ci_upper"] * 1000 for row in policy])
        axes[0].errorbar(
            positions,
            profit,
            yerr=np.vstack([profit - profit_lower, profit_upper - profit]),
            marker=marker,
            color=color,
            linewidth=1.5,
            capsize=2,
            label=label,
        )
        axes[1].errorbar(
            positions,
            conversion,
            yerr=np.vstack([conversion - conversion_lower, conversion_upper - conversion]),
            marker=marker,
            color=color,
            linewidth=1.5,
            capsize=2,
            label=label,
        )
    axes[0].axhline(0, color="#555555", linewidth=1)
    axes[0].set_ylabel("Profit per 1,000 eligible customers ($)")
    axes[0].set_title("Incremental profit")
    axes[1].axhline(0, color="#555555", linewidth=1)
    axes[1].set_ylabel("Incremental conversions per 1,000 emails")
    axes[1].set_title("Incremental conversion effect")
    for axis in axes:
        axis.set_xticks(positions, labels=[strength.capitalize() for strength in strengths])
        axis.set_xlabel("Training-data confounding")
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    figure.suptitle(f"RCT policy value at {budget:.0%} targeting and ${email_cost:.2f}/email")
    figure.tight_layout()
    _save_figure(figure, output)
    plt.close(figure)


def plot_uplift_tree_structure(
    root: object,
    feature_names: tuple[str, ...],
    output: Path,
    *,
    control_name: str,
    treatment_name: str,
) -> None:
    """Render an uplift tree without requiring a system Graphviz installation."""
    nodes: list[dict[str, object]] = []
    edges: list[tuple[int, int, str]] = []
    leaf_position = 0

    def visit(node: object, depth: int) -> tuple[int, float]:
        nonlocal leaf_position
        index = len(nodes)
        nodes.append({})
        if node.results is not None:
            x_position = float(leaf_position)
            leaf_position += 1
            rates = {
                str(group): float(rate)
                for group, rate in zip(node.classes_, node.results, strict=True)
            }
            uplift = rates[treatment_name] - rates[control_name]
            label = (
                f"T: {rates[treatment_name]:.3f}\n"
                f"C: {rates[control_name]:.3f}\n"
                f"uplift: {uplift:+.3f}\n"
                f"n: {int(node.summary['samples']):,}"
            )
            nodes[index] = {
                "x": x_position,
                "y": -depth,
                "label": label,
                "leaf": True,
                "uplift": uplift,
            }
            return index, x_position

        true_index, true_x = visit(node.trueBranch, depth + 1)
        false_index, false_x = visit(node.falseBranch, depth + 1)
        x_position = (true_x + false_x) / 2
        feature = feature_names[int(node.col)]
        operator = ">=" if isinstance(node.value, (int, float)) else "=="
        label = f"{feature} {operator} {node.value}\nn: {int(node.summary['samples']):,}"
        nodes[index] = {
            "x": x_position,
            "y": -depth,
            "label": label,
            "leaf": False,
            "uplift": 0.0,
        }
        edges.extend([(index, true_index, "yes"), (index, false_index, "no")])
        return index, x_position

    visit(root, 0)
    max_depth = int(max(-float(node["y"]) for node in nodes))
    figure_width = max(14.0, leaf_position * 2.0)
    figure, axis = plt.subplots(figsize=(figure_width, max(7.0, (max_depth + 1) * 2.0)))
    for parent, child, label in edges:
        parent_node = nodes[parent]
        child_node = nodes[child]
        axis.plot(
            [parent_node["x"], child_node["x"]],
            [parent_node["y"], child_node["y"]],
            color="#969696",
            linewidth=1.2,
            zorder=1,
        )
        axis.text(
            (float(parent_node["x"]) + float(child_node["x"])) / 2,
            (float(parent_node["y"]) + float(child_node["y"])) / 2,
            label,
            fontsize=7,
            color="#555555",
        )
    leaf_uplifts = [float(node["uplift"]) for node in nodes if node["leaf"]]
    scale = max(max(abs(value) for value in leaf_uplifts), 1e-6)
    for node in nodes:
        if node["leaf"]:
            normalized = (float(node["uplift"]) / scale + 1) / 2
            color = plt.colormaps["RdYlBu"](normalized)
        else:
            color = "#f0f0f0"
        axis.text(
            node["x"],
            node["y"],
            node["label"],
            ha="center",
            va="center",
            fontsize=7.5,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": color, "edgecolor": "#555555"},
            zorder=2,
        )
    axis.set_xlim(-1, max(leaf_position, 1))
    axis.set_ylim(-max_depth - 0.7, 0.7)
    axis.axis("off")
    axis.set_title("Honest uplift tree trained on observational marketing data")
    figure.tight_layout()
    _save_figure(figure, output)
    plt.close(figure)
