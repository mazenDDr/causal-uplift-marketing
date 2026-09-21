from __future__ import annotations

import numpy as np
import pandas as pd


def _validated_arrays(
    outcome: np.ndarray,
    treatment: np.ndarray,
    score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outcome = np.asarray(outcome, dtype=float)
    treatment = np.asarray(treatment, dtype=int)
    score = np.asarray(score, dtype=float)
    if outcome.ndim != 1 or treatment.shape != outcome.shape or score.shape != outcome.shape:
        raise ValueError("outcome, treatment, and score must be aligned one-dimensional arrays")
    if len(outcome) == 0 or not np.isfinite(outcome).all() or not np.isfinite(score).all():
        raise ValueError("outcome and score must be finite and non-empty")
    if set(np.unique(treatment)) != {0, 1}:
        raise ValueError("both binary treatment groups are required")
    return outcome, treatment, score


def tie_aware_gain_curve(
    outcome: np.ndarray,
    treatment: np.ndarray,
    score: np.ndarray,
    fractions: np.ndarray,
) -> dict[str, np.ndarray]:
    """Estimate cumulative policy gain without arbitrary ordering inside score ties.

    Gain at fraction ``f`` is ``f * (mean(Y|T=1,targeted)-mean(Y|T=0,targeted))``.
    If a score tie crosses the targeting boundary, every row in that tie receives the same
    fractional selection weight.
    """
    outcome, treatment, score = _validated_arrays(outcome, treatment, score)
    fractions = np.asarray(fractions, dtype=float)
    if (
        fractions.ndim != 1
        or len(fractions) < 2
        or not np.isclose(fractions[0], 0.0)
        or not np.isclose(fractions[-1], 1.0)
        or np.any(np.diff(fractions) <= 0)
    ):
        raise ValueError("fractions must increase from 0 to 1")

    order = np.argsort(-score, kind="stable")
    sorted_score = score[order]
    sorted_treatment = treatment[order]
    sorted_outcome = outcome[order]
    block_start = np.r_[0, np.flatnonzero(np.diff(sorted_score) != 0) + 1]
    block_end = np.r_[block_start[1:], len(score)]
    block_rows = block_end - block_start
    block_treated = np.add.reduceat((sorted_treatment == 1).astype(float), block_start)
    block_control = np.add.reduceat((sorted_treatment == 0).astype(float), block_start)
    block_treated_outcomes = np.add.reduceat(sorted_outcome * (sorted_treatment == 1), block_start)
    block_control_outcomes = np.add.reduceat(sorted_outcome * (sorted_treatment == 0), block_start)
    cumulative_rows = np.cumsum(block_rows)
    cumulative_treated = np.cumsum(block_treated)
    cumulative_control = np.cumsum(block_control)
    cumulative_treated_outcomes = np.cumsum(block_treated_outcomes)
    cumulative_control_outcomes = np.cumsum(block_control_outcomes)

    uplift = np.zeros(len(fractions), dtype=float)
    gain = np.zeros(len(fractions), dtype=float)
    for position, fraction in enumerate(fractions[1:], start=1):
        target_rows = fraction * len(score)
        block = int(np.searchsorted(cumulative_rows, target_rows, side="left"))
        previous_rows = 0 if block == 0 else cumulative_rows[block - 1]
        weight = (target_rows - previous_rows) / block_rows[block]

        treated_previous = 0.0 if block == 0 else cumulative_treated[block - 1]
        control_previous = 0.0 if block == 0 else cumulative_control[block - 1]
        treated_count = float(treated_previous + weight * block_treated[block])
        control_count = float(control_previous + weight * block_control[block])
        if treated_count <= 0 or control_count <= 0:
            raise ValueError("each targeted fraction must contain treated and control support")
        treated_outcome_previous = 0.0 if block == 0 else cumulative_treated_outcomes[block - 1]
        control_outcome_previous = 0.0 if block == 0 else cumulative_control_outcomes[block - 1]
        treated_sum = float(treated_outcome_previous + weight * block_treated_outcomes[block])
        control_sum = float(control_outcome_previous + weight * block_control_outcomes[block])
        uplift[position] = treated_sum / treated_count - control_sum / control_count
        gain[position] = fraction * uplift[position]

    random_gain = fractions * gain[-1]
    qini_gain = gain - random_gain
    return {
        "fraction": fractions,
        "uplift": uplift,
        "gain": gain,
        "random_gain": random_gain,
        "qini_gain": qini_gain,
    }


def ranking_metrics(
    outcome: np.ndarray,
    treatment: np.ndarray,
    score: np.ndarray,
    fractions: np.ndarray,
) -> dict[str, object]:
    curve = tie_aware_gain_curve(outcome, treatment, score, fractions)
    auuc = float(np.trapezoid(curve["gain"], curve["fraction"]))
    qini = float(np.trapezoid(curve["qini_gain"], curve["fraction"]))
    return {
        "curve": {name: values.tolist() for name, values in curve.items()},
        "auuc": auuc,
        "qini": qini,
    }


def bootstrap_ranking_metrics(
    frame: pd.DataFrame,
    scores: dict[str, np.ndarray],
    fractions: np.ndarray,
    *,
    outcome: str = "conversion",
    samples: int = 1000,
    seed: int = 42,
    reference: str = "response_model",
    alpha: float = 0.05,
) -> dict[str, object]:
    """Return treatment-stratified bootstrap intervals and paired Qini differences."""
    if reference not in scores:
        raise ValueError("reference score is required")
    treatment = frame["treatment"].to_numpy(dtype=int)
    values = frame[outcome].to_numpy(dtype=float)
    validated_scores = {
        name: _validated_arrays(values, treatment, score)[2] for name, score in scores.items()
    }
    point = {
        name: ranking_metrics(values, treatment, score, fractions)
        for name, score in validated_scores.items()
    }
    metric_draws = {
        name: {"auuc": np.empty(samples), "qini": np.empty(samples)} for name in validated_scores
    }
    curve_draws = {
        name: {
            "gain": np.empty((samples, len(fractions))),
            "qini_gain": np.empty((samples, len(fractions))),
        }
        for name in validated_scores
    }
    treated_indices = np.flatnonzero(treatment == 1)
    control_indices = np.flatnonzero(treatment == 0)
    rng = np.random.default_rng(seed)
    for draw in range(samples):
        indices = np.concatenate(
            [
                rng.choice(treated_indices, len(treated_indices), replace=True),
                rng.choice(control_indices, len(control_indices), replace=True),
            ]
        )
        for name, score in validated_scores.items():
            result = ranking_metrics(values[indices], treatment[indices], score[indices], fractions)
            metric_draws[name]["auuc"][draw] = result["auuc"]
            metric_draws[name]["qini"][draw] = result["qini"]
            curve_draws[name]["gain"][draw] = result["curve"]["gain"]
            curve_draws[name]["qini_gain"][draw] = result["curve"]["qini_gain"]

    lower_quantile, upper_quantile = alpha / 2, 1 - alpha / 2
    models: dict[str, object] = {}
    for name, result in point.items():
        curve = result["curve"]
        models[name] = {
            "curve": {
                **curve,
                "gain_ci_lower": np.quantile(
                    curve_draws[name]["gain"], lower_quantile, axis=0
                ).tolist(),
                "gain_ci_upper": np.quantile(
                    curve_draws[name]["gain"], upper_quantile, axis=0
                ).tolist(),
                "qini_ci_lower": np.quantile(
                    curve_draws[name]["qini_gain"], lower_quantile, axis=0
                ).tolist(),
                "qini_ci_upper": np.quantile(
                    curve_draws[name]["qini_gain"], upper_quantile, axis=0
                ).tolist(),
            },
            "auuc": {
                "estimate": result["auuc"],
                "ci_lower": float(np.quantile(metric_draws[name]["auuc"], lower_quantile)),
                "ci_upper": float(np.quantile(metric_draws[name]["auuc"], upper_quantile)),
            },
            "qini": {
                "estimate": result["qini"],
                "ci_lower": float(np.quantile(metric_draws[name]["qini"], lower_quantile)),
                "ci_upper": float(np.quantile(metric_draws[name]["qini"], upper_quantile)),
            },
        }

    paired = {}
    reference_draws = metric_draws[reference]["qini"]
    for name in validated_scores:
        if name == reference:
            continue
        difference = metric_draws[name]["qini"] - reference_draws
        paired[name] = {
            "estimate": point[name]["qini"] - point[reference]["qini"],
            "ci_lower": float(np.quantile(difference, lower_quantile)),
            "ci_upper": float(np.quantile(difference, upper_quantile)),
        }
    return {
        "models": models,
        "paired_qini_difference_vs_reference": {"reference": reference, "models": paired},
    }
