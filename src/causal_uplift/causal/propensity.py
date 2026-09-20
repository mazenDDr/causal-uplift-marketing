from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from causal_uplift.causal.baselines import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from causal_uplift.data.load import PRE_TREATMENT_COLUMNS


@dataclass
class LogisticPropensityModel:
    """Estimate treatment assignment from pre-treatment covariates only."""

    seed: int = 42
    max_iter: int = 2000

    def __post_init__(self) -> None:
        preprocess = ColumnTransformer(
            [
                ("numeric", StandardScaler(), NUMERIC_FEATURES),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", drop="first"),
                    CATEGORICAL_FEATURES,
                ),
            ],
            remainder="drop",
        )
        estimator = LogisticRegression(
            max_iter=self.max_iter,
            random_state=self.seed,
        )
        self.pipeline = Pipeline([("preprocess", preprocess), ("model", estimator)])

    def fit(self, frame: pd.DataFrame) -> LogisticPropensityModel:
        missing = set(PRE_TREATMENT_COLUMNS).difference(frame.columns)
        if missing:
            raise ValueError(f"propensity features are missing: {sorted(missing)}")
        self.pipeline.fit(frame, frame["treatment"])
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        propensity = self.pipeline.predict_proba(frame)[:, 1]
        if not np.isfinite(propensity).all() or not ((propensity >= 0) & (propensity <= 1)).all():
            raise AssertionError("propensity predictions must be finite probabilities")
        return propensity


def propensity_diagnostics(
    frame: pd.DataFrame,
    propensity: np.ndarray,
    *,
    overlap_min: float = 0.05,
    overlap_max: float = 0.95,
) -> dict[str, object]:
    """Summarize treatment prediction and empirical common support."""
    propensity = np.asarray(propensity, dtype=float)
    if propensity.shape != (len(frame),):
        raise ValueError("propensity must have one value per row")
    if not 0 < overlap_min < overlap_max < 1:
        raise ValueError("overlap thresholds must satisfy 0 < min < max < 1")
    treatment = frame["treatment"].to_numpy(dtype=int)
    if set(np.unique(treatment)) != {0, 1}:
        raise ValueError("both binary treatment groups are required")

    treated_scores = propensity[treatment == 1]
    control_scores = propensity[treatment == 0]
    common_lower = max(float(treated_scores.min()), float(control_scores.min()))
    common_upper = min(float(treated_scores.max()), float(control_scores.max()))
    quantiles = [0.01, 0.05, 0.50, 0.95, 0.99]

    diagnostics: dict[str, object] = {
        "roc_auc": float(roc_auc_score(treatment, propensity)),
        "brier_score": float(brier_score_loss(treatment, propensity)),
        "log_loss": float(log_loss(treatment, propensity)),
        "minimum": float(propensity.min()),
        "maximum": float(propensity.max()),
        "outside_overlap_threshold_fraction": float(
            ((propensity < overlap_min) | (propensity > overlap_max)).mean()
        ),
        "common_support_lower": common_lower,
        "common_support_upper": common_upper,
        "outside_empirical_common_support_fraction": float(
            ((propensity < common_lower) | (propensity > common_upper)).mean()
        ),
        "treated_quantiles": {
            str(quantile): float(value)
            for quantile, value in zip(
                quantiles, np.quantile(treated_scores, quantiles), strict=True
            )
        },
        "control_quantiles": {
            str(quantile): float(value)
            for quantile, value in zip(
                quantiles, np.quantile(control_scores, quantiles), strict=True
            )
        },
    }
    if "desired_propensity" in frame and frame["desired_propensity"].std(ddof=0) > 0:
        desired = frame["desired_propensity"].to_numpy(dtype=float)
        diagnostics["correlation_with_desired_propensity"] = float(
            np.corrcoef(propensity, desired)[0, 1]
        )
        diagnostics["rmse_vs_desired_propensity"] = float(
            np.sqrt(np.mean((propensity - desired) ** 2))
        )
    else:
        diagnostics["correlation_with_desired_propensity"] = None
        diagnostics["rmse_vs_desired_propensity"] = None
    return diagnostics
