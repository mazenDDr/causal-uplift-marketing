from __future__ import annotations

import numpy as np
import pandas as pd

from causal_uplift.evaluation.ate import bootstrap_difference_in_means


def ranked_policy_mask(
    scores: np.ndarray,
    fraction: float,
    *,
    tie_breaker: np.ndarray,
) -> np.ndarray:
    """Select an exact top fraction with an outcome-independent tie breaker."""
    scores = np.asarray(scores, dtype=float)
    tie_breaker = np.asarray(tie_breaker, dtype=float)
    if scores.ndim != 1 or tie_breaker.shape != scores.shape:
        raise ValueError("scores and tie_breaker must be aligned one-dimensional arrays")
    if not np.isfinite(scores).all() or not np.isfinite(tie_breaker).all():
        raise ValueError("scores and tie_breaker must be finite")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    selected_count = max(1, int(round(len(scores) * fraction)))
    order = np.lexsort((-tie_breaker, -scores))
    selected = np.zeros(len(scores), dtype=bool)
    selected[order[:selected_count]] = True
    return selected


def cost_aware_mask(
    predicted_conversion_effect: np.ndarray,
    *,
    conversion_value: float,
    email_cost: float,
) -> np.ndarray:
    """Treat when predicted incremental conversion value exceeds intervention cost."""
    effect = np.asarray(predicted_conversion_effect, dtype=float)
    if effect.ndim != 1 or not np.isfinite(effect).all():
        raise ValueError("predicted effects must be a finite one-dimensional array")
    if conversion_value <= 0 or email_cost < 0:
        raise ValueError("conversion value must be positive and email cost non-negative")
    return conversion_value * effect > email_cost


def evaluate_business_policy(
    frame: pd.DataFrame,
    selected: np.ndarray,
    *,
    email_cost: float,
    bootstrap_samples: int = 1000,
    seed: int = 42,
) -> dict[str, object]:
    """Evaluate conversion, visit, spend, profit, and ROI from randomized outcomes."""
    selected = np.asarray(selected, dtype=bool)
    if selected.shape != (len(frame),):
        raise ValueError("selected mask must have one entry per row")
    fraction = float(selected.mean())
    targeted = frame.loc[selected]
    if len(targeted) == 0:
        return {
            "targeted_customers": 0,
            "targeted_fraction": 0.0,
            "conversion": None,
            "visit": None,
            "spend": None,
            "incremental_revenue_per_1000_eligible": 0.0,
            "campaign_cost_per_1000_eligible": 0.0,
            "incremental_profit_per_1000_eligible": {
                "estimate": 0.0,
                "ci_lower": 0.0,
                "ci_upper": 0.0,
            },
            "roi": None,
        }
    if set(targeted["treatment"].unique()) != {0, 1}:
        raise ValueError("selected policy must have randomized treatment and control support")
    effects = {
        outcome: bootstrap_difference_in_means(
            targeted,
            outcome,
            samples=bootstrap_samples,
            seed=seed,
        )
        for outcome in ("conversion", "visit", "spend")
    }
    for result in effects.values():
        result["effect_per_1000_emails"] = result["estimate"] * 1000
        result["incremental_per_1000_eligible"] = result["estimate"] * fraction * 1000
        result["incremental_total_at_holdout_scale"] = result["estimate"] * len(targeted)

    revenue = effects["spend"]["estimate"] * fraction * 1000
    revenue_lower = effects["spend"]["ci_lower"] * fraction * 1000
    revenue_upper = effects["spend"]["ci_upper"] * fraction * 1000
    cost = fraction * email_cost * 1000
    profit = revenue - cost
    profit_lower = revenue_lower - cost
    profit_upper = revenue_upper - cost
    roi = None
    if cost > 0:
        roi = {
            "estimate": profit / cost,
            "ci_lower": profit_lower / cost,
            "ci_upper": profit_upper / cost,
        }
    return {
        "targeted_customers": int(selected.sum()),
        "targeted_fraction": fraction,
        **effects,
        "incremental_revenue_per_1000_eligible": revenue,
        "campaign_cost_per_1000_eligible": cost,
        "incremental_profit_per_1000_eligible": {
            "estimate": profit,
            "ci_lower": profit_lower,
            "ci_upper": profit_upper,
        },
        "roi": roi,
    }


def bootstrap_profit_difference(
    frame: pd.DataFrame,
    selected_a: np.ndarray,
    selected_b: np.ndarray,
    *,
    email_cost: float,
    samples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Paired treatment-stratified bootstrap of policy profit per 1,000 eligible customers."""
    treatment = frame["treatment"].to_numpy(dtype=int)
    spend = frame["spend"].to_numpy(dtype=float)
    selected_a = np.asarray(selected_a, dtype=bool)
    selected_b = np.asarray(selected_b, dtype=bool)
    if selected_a.shape != (len(frame),) or selected_b.shape != (len(frame),):
        raise ValueError("policy masks must align with the frame")

    def value(indices: np.ndarray, selected: np.ndarray) -> float:
        chosen = selected[indices]
        fraction = float(chosen.mean())
        if fraction == 0:
            return 0.0
        chosen_treatment = treatment[indices][chosen]
        chosen_spend = spend[indices][chosen]
        if set(np.unique(chosen_treatment)) != {0, 1}:
            raise ValueError("selected policy must have treatment and control support")
        uplift = (
            chosen_spend[chosen_treatment == 1].mean() - chosen_spend[chosen_treatment == 0].mean()
        )
        return float(fraction * (uplift - email_cost) * 1000)

    all_indices = np.arange(len(frame))
    estimate = value(all_indices, selected_a) - value(all_indices, selected_b)
    treated_indices = np.flatnonzero(treatment == 1)
    control_indices = np.flatnonzero(treatment == 0)
    rng = np.random.default_rng(seed)
    draws = np.empty(samples)
    for draw in range(samples):
        indices = np.concatenate(
            [
                rng.choice(treated_indices, len(treated_indices), replace=True),
                rng.choice(control_indices, len(control_indices), replace=True),
            ]
        )
        draws[draw] = value(indices, selected_a) - value(indices, selected_b)
    lower, upper = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {"estimate": estimate, "ci_lower": float(lower), "ci_upper": float(upper)}


def psm_segment_scores(
    training: pd.DataFrame,
    pairs: pd.DataFrame,
    scoring: pd.DataFrame,
    *,
    history_threshold: float,
    recency_threshold: float,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Estimate four matched-pair segment effects and map them onto scoring rows."""
    treated = training.iloc[pairs["treated_position"].to_numpy(dtype=int)].copy()
    control = training.iloc[pairs["control_position"].to_numpy(dtype=int)].copy()

    def labels(frame: pd.DataFrame) -> np.ndarray:
        history = np.where(frame["history"] >= history_threshold, "high history", "low history")
        recency = np.where(frame["recency"] <= recency_threshold, "recent", "inactive")
        return np.array(
            [
                f"{recency_value} / {history_value}"
                for recency_value, history_value in zip(recency, history, strict=True)
            ]
        )

    treated_labels = labels(treated)
    differences = treated["conversion"].to_numpy(float) - control["conversion"].to_numpy(float)
    segment_rows = []
    score_map = {}
    for label in sorted(np.unique(treated_labels)):
        selected = treated_labels == label
        score = float(differences[selected].mean())
        score_map[label] = score
        segment_rows.append(
            {"segment": label, "matched_pairs": int(selected.sum()), "conversion_att": score}
        )
    scoring_labels = labels(scoring)
    return np.array([score_map[label] for label in scoring_labels]), segment_rows
