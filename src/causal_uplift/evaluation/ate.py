from __future__ import annotations

import numpy as np
import pandas as pd


def difference_in_means(frame: pd.DataFrame, outcome: str) -> float:
    treated = frame.loc[frame["treatment"] == 1, outcome]
    control = frame.loc[frame["treatment"] == 0, outcome]
    if treated.empty or control.empty:
        raise ValueError("both treatment groups are required")
    return float(treated.mean() - control.mean())


def bootstrap_difference_in_means(
    frame: pd.DataFrame,
    outcome: str,
    *,
    samples: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Stratified percentile bootstrap CI for a treatment-control contrast."""
    if samples < 1:
        raise ValueError("samples must be positive")
    rng = np.random.default_rng(seed)
    treated = frame.loc[frame["treatment"] == 1, outcome].to_numpy(dtype=float)
    control = frame.loc[frame["treatment"] == 0, outcome].to_numpy(dtype=float)
    if len(treated) == 0 or len(control) == 0:
        raise ValueError("both treatment groups are required")
    draws = np.empty(samples)
    for index in range(samples):
        treated_draw = rng.choice(treated, size=len(treated), replace=True)
        control_draw = rng.choice(control, size=len(control), replace=True)
        draws[index] = treated_draw.mean() - control_draw.mean()
    lower, upper = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {
        "estimate": difference_in_means(frame, outcome),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
    }
