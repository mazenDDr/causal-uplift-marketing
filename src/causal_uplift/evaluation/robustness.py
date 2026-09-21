from __future__ import annotations

import numpy as np
import pandas as pd


def joint_stratified_subsample(
    frame: pd.DataFrame,
    fraction: float,
    *,
    seed: int,
) -> pd.DataFrame:
    """Subsample within joint treatment/outcome strata without altering source rows."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    required = {"source_row_id", "treatment", "conversion"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"subsample columns are missing: {sorted(missing)}")
    if fraction == 1:
        return frame.copy().sort_values("source_row_id").reset_index(drop=True)

    rng = np.random.default_rng(seed)
    selected = []
    for _, group in frame.groupby(["treatment", "conversion"], sort=True):
        rows = max(1, int(round(len(group) * fraction)))
        selected.extend(rng.choice(group.index.to_numpy(), size=rows, replace=False).tolist())
    return frame.loc[selected].copy().sort_values("source_row_id").reset_index(drop=True)


def add_noise_covariates(
    frame: pd.DataFrame,
    *,
    columns: int,
    seed: int,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Add independent numeric pre-treatment noise columns for a stability check."""
    if columns < 1:
        raise ValueError("columns must be positive")
    result = frame.copy()
    names = tuple(f"noise_{index}" for index in range(columns))
    values = np.random.default_rng(seed).normal(size=(len(frame), columns))
    for index, name in enumerate(names):
        result[name] = values[:, index]
    return result, names


def shuffle_treatment(frame: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """Create a placebo assignment while leaving covariates and outcomes intact."""
    if "treatment" not in frame:
        raise ValueError("treatment column is required")
    result = frame.copy()
    result["treatment"] = np.random.default_rng(seed).permutation(
        frame["treatment"].to_numpy(dtype=int)
    )
    return result
