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
