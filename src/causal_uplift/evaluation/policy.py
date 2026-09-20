from __future__ import annotations

import numpy as np
import pandas as pd

from causal_uplift.evaluation.ate import bootstrap_difference_in_means


def top_fraction_mask(scores: np.ndarray, fraction: float) -> np.ndarray:
    """Select an exact, deterministic top fraction using stable tie handling."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or np.isnan(scores).any():
        raise ValueError("scores must be a one-dimensional array without missing values")
    selected_count = max(1, int(round(len(scores) * fraction)))
    ranking = np.argsort(-scores, kind="stable")
    mask = np.zeros(len(scores), dtype=bool)
    mask[ranking[:selected_count]] = True
    return mask


def random_policy_scores(frame: pd.DataFrame, *, seed: int) -> np.ndarray:
    """Return a fixed random ranking independent of row order."""
    if "source_row_id" not in frame:
        raise ValueError("source_row_id is required for a repeatable random policy")
    identifiers = frame["source_row_id"].to_numpy(dtype=np.uint64)
    mixed = identifiers ^ np.uint64(seed)
    mixed ^= mixed >> np.uint64(30)
    mixed *= np.uint64(0xBF58476D1CE4E5B9)
    mixed ^= mixed >> np.uint64(27)
    mixed *= np.uint64(0x94D049BB133111EB)
    mixed ^= mixed >> np.uint64(31)
    return mixed.astype(np.float64) / np.float64(np.iinfo(np.uint64).max)


def evaluate_binary_policy(
    frame: pd.DataFrame,
    selected: np.ndarray,
    *,
    outcome: str = "conversion",
    bootstrap_samples: int = 2000,
    seed: int = 42,
) -> dict[str, float | int]:
    selected = np.asarray(selected, dtype=bool)
    if selected.shape != (len(frame),):
        raise ValueError("selected mask must have one entry per row")
    targeted = frame.loc[selected]
    effect = bootstrap_difference_in_means(
        targeted,
        outcome,
        samples=bootstrap_samples,
        seed=seed,
    )
    return {
        "targeted_customers": int(selected.sum()),
        "targeted_fraction": float(selected.mean()),
        "treated_evaluation_rows": int((targeted["treatment"] == 1).sum()),
        "control_evaluation_rows": int((targeted["treatment"] == 0).sum()),
        "observed_uplift": effect["estimate"],
        "ci_lower": effect["ci_lower"],
        "ci_upper": effect["ci_upper"],
        "incremental_conversions_per_1000": effect["estimate"] * 1000,
    }


def _effect_for_mask(
    outcome: np.ndarray,
    treatment: np.ndarray,
    selected: np.ndarray,
) -> float:
    treated = outcome[(treatment == 1) & selected]
    control = outcome[(treatment == 0) & selected]
    if len(treated) == 0 or len(control) == 0:
        raise ValueError("each policy must select treated and control evaluation rows")
    return float(treated.mean() - control.mean())


def bootstrap_policy_difference(
    frame: pd.DataFrame,
    selected_a: np.ndarray,
    selected_b: np.ndarray,
    *,
    outcome: str = "conversion",
    samples: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Paired, treatment-stratified bootstrap of policy A minus policy B uplift."""
    treatment = frame["treatment"].to_numpy(dtype=int)
    values = frame[outcome].to_numpy(dtype=float)
    selected_a = np.asarray(selected_a, dtype=bool)
    selected_b = np.asarray(selected_b, dtype=bool)
    expected_shape = (len(frame),)
    if selected_a.shape != expected_shape or selected_b.shape != expected_shape:
        raise ValueError("policy masks must have one entry per row")

    estimate = _effect_for_mask(values, treatment, selected_a) - _effect_for_mask(
        values, treatment, selected_b
    )
    treated_indices = np.flatnonzero(treatment == 1)
    control_indices = np.flatnonzero(treatment == 0)
    rng = np.random.default_rng(seed)
    draws = np.empty(samples)
    for draw in range(samples):
        indices = np.concatenate(
            [
                rng.choice(treated_indices, size=len(treated_indices), replace=True),
                rng.choice(control_indices, size=len(control_indices), replace=True),
            ]
        )
        draws[draw] = _effect_for_mask(
            values[indices], treatment[indices], selected_a[indices]
        ) - _effect_for_mask(values[indices], treatment[indices], selected_b[indices])
    lower, upper = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {
        "uplift_difference": estimate,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "incremental_conversions_per_1000_difference": estimate * 1000,
    }
