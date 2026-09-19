from __future__ import annotations

import numpy as np
import pandas as pd


def _numeric_smd(values: pd.Series, treatment: pd.Series) -> float:
    treated = values.loc[treatment == 1].astype(float)
    control = values.loc[treatment == 0].astype(float)
    pooled_sd = np.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2)
    mean_difference = treated.mean() - control.mean()
    if pooled_sd == 0:
        return 0.0 if mean_difference == 0 else float(np.sign(mean_difference) * np.inf)
    return float(mean_difference / pooled_sd)


def covariate_smds(
    frame: pd.DataFrame,
    columns: list[str] | tuple[str, ...],
) -> dict[str, float]:
    """Return signed SMDs, expanding each categorical level to an indicator."""
    if set(frame["treatment"].unique()) != {0, 1}:
        raise ValueError("balance requires both binary treatment groups")
    treatment = frame["treatment"]
    result: dict[str, float] = {}
    for column in columns:
        values = frame[column]
        if pd.api.types.is_numeric_dtype(values):
            result[column] = _numeric_smd(values, treatment)
            continue
        for level in sorted(values.astype(str).unique()):
            indicator = values.astype(str).eq(level).astype(float)
            result[f"{column}={level}"] = _numeric_smd(indicator, treatment)
    return result
