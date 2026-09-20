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
