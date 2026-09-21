from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit

FORBIDDEN_FEATURES = frozenset({"visit", "conversion", "spend", "treatment", "segment"})


@dataclass(frozen=True)
class ConfoundingConfig:
    strength: float
    clip_min: float = 0.05
    clip_max: float = 0.95
    affinity_column: str = "mens"
    history: float = 0.55
    recency: float = -0.45
    mens: float = 0.65
    multichannel: float = 0.45
    newbie: float = -0.35

    def __post_init__(self) -> None:
        if self.strength < 0:
            raise ValueError("strength must be non-negative")
        if not 0 < self.clip_min < self.clip_max < 1:
            raise ValueError("propensity clips must satisfy 0 < min < max < 1")
        if self.affinity_column not in {"mens", "womens"}:
            raise ValueError("affinity_column must be 'mens' or 'womens'")


def _standardize(series: pd.Series) -> np.ndarray:
    values = series.to_numpy(dtype=float)
    scale = values.std(ddof=0)
    return np.zeros_like(values) if scale == 0 else (values - values.mean()) / scale


def desired_propensity(frame: pd.DataFrame, config: ConfoundingConfig) -> np.ndarray:
    """Construct the pre-specified historical targeting propensity e(X)."""
    required = {"history", "recency", config.affinity_column, "channel", "newbie"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"confounding features are missing: {sorted(missing)}")
    if required & FORBIDDEN_FEATURES:
        raise AssertionError("post-treatment data entered the confounding feature set")

    score = (
        config.history * _standardize(frame["history"])
        + config.recency * _standardize(frame["recency"])
        + config.mens * frame[config.affinity_column].to_numpy(dtype=float)
        + config.multichannel
        * frame["channel"].astype(str).str.lower().eq("multichannel").to_numpy(dtype=float)
        + config.newbie * frame["newbie"].to_numpy(dtype=float)
    )
    score -= score.mean()
    propensity = expit(config.strength * score)
    return np.clip(propensity, config.clip_min, config.clip_max)


def sample_observational(
    randomized_training_pool: pd.DataFrame,
    config: ConfoundingConfig,
    *,
    seed: int,
) -> pd.DataFrame:
    """Select intact RCT rows so treatment depends on X without fabricating outcomes."""
    if "treatment" not in randomized_training_pool:
        raise ValueError("training pool must include binary treatment")
    treatment = randomized_training_pool["treatment"].to_numpy(dtype=int)
    if not np.isin(treatment, [0, 1]).all():
        raise ValueError("treatment must be binary")

    propensity = desired_propensity(randomized_training_pool, config)
    if config.strength == 0:
        selected = randomized_training_pool.copy()
        selected["desired_propensity"] = propensity
        selected["selection_probability"] = 1.0
        selected["confounding_strength"] = config.strength
        return selected.sort_values("source_row_id").reset_index(drop=True)

    keep_probability = np.where(treatment == 1, propensity, 1.0 - propensity)
    keep = np.random.default_rng(seed).random(len(randomized_training_pool)) < keep_probability

    selected = randomized_training_pool.loc[keep].copy()
    selected["desired_propensity"] = propensity[keep]
    selected["selection_probability"] = keep_probability[keep]
    selected["confounding_strength"] = config.strength
    return selected.sort_values("source_row_id").reset_index(drop=True)
