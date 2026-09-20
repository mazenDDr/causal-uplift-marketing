from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML

from causal_uplift.causal.baselines import CATEGORICAL_FEATURES
from causal_uplift.causal.dml import (
    make_confounder_encoder,
    make_nuisance_models,
)
from causal_uplift.data.load import PRE_TREATMENT_COLUMNS


@dataclass
class FittedCausalForest:
    estimator: CausalForestDML
    encoder: object
    feature_names: tuple[str, ...]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        features = self.encoder.transform(frame.loc[:, PRE_TREATMENT_COLUMNS])
        return np.asarray(self.estimator.effect(features), dtype=float)

    def predict_interval(
        self,
        frame: pd.DataFrame,
        *,
        alpha: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray]:
        features = self.encoder.transform(frame.loc[:, PRE_TREATMENT_COLUMNS])
        lower, upper = self.estimator.effect_interval(features, alpha=alpha)
        return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)

    def ate_interval(
        self,
        frame: pd.DataFrame,
        *,
        alpha: float = 0.05,
    ) -> tuple[float, float, float]:
        features = self.encoder.transform(frame.loc[:, PRE_TREATMENT_COLUMNS])
        estimate = float(np.asarray(self.estimator.ate(features)).squeeze())
        lower, upper = self.estimator.ate_interval(features, alpha=alpha)
        return (
            estimate,
            float(np.asarray(lower).squeeze()),
            float(np.asarray(upper).squeeze()),
        )

    def aggregated_feature_importance(self) -> dict[str, float]:
        importance = np.asarray(self.estimator.feature_importances_, dtype=float)
        aggregated = {feature: 0.0 for feature in PRE_TREATMENT_COLUMNS}
        for encoded_name, value in zip(self.feature_names, importance, strict=True):
            matches = [
                feature
                for feature in PRE_TREATMENT_COLUMNS
                if encoded_name == feature
                or (feature in CATEGORICAL_FEATURES and encoded_name.startswith(f"{feature}_"))
            ]
            if len(matches) != 1:
                raise AssertionError(f"cannot map encoded feature {encoded_name!r}")
            aggregated[matches[0]] += float(value)
        total = sum(aggregated.values())
        return {feature: value / total for feature, value in aggregated.items()}


def fit_causal_forest(
    frame: pd.DataFrame,
    outcome: str,
    *,
    nuisance_candidate: str,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    n_estimators: int = 400,
    min_samples_leaf: int = 100,
    max_depth: int | None = 12,
    max_samples: float = 0.45,
    max_features: float = 0.7,
    subforest_size: int = 4,
    honest: bool = True,
    nuisance_random_forest_trees: int = 200,
    nuisance_max_iter: int = 200,
) -> FittedCausalForest:
    if n_estimators % subforest_size:
        raise ValueError("n_estimators must be divisible by subforest_size for inference")
    encoder = make_confounder_encoder()
    features = encoder.fit_transform(frame.loc[:, PRE_TREATMENT_COLUMNS])
    model_y, model_t = make_nuisance_models(
        nuisance_candidate,
        seed=seed,
        random_forest_trees=nuisance_random_forest_trees,
        min_samples_leaf=min_samples_leaf,
        max_iter=nuisance_max_iter,
    )
    estimator = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=True,
        cv=splits,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        max_samples=max_samples,
        max_features=max_features,
        subforest_size=subforest_size,
        honest=honest,
        inference=True,
        n_jobs=-1,
        random_state=seed,
    )
    estimator.fit(
        frame[outcome].to_numpy(dtype=float),
        frame["treatment"].to_numpy(dtype=int),
        X=features,
        inference="auto",
    )
    return FittedCausalForest(
        estimator=estimator,
        encoder=encoder,
        feature_names=tuple(encoder.get_feature_names_out()),
    )


def effect_bins(scores: np.ndarray, *, bins: int = 10) -> np.ndarray:
    """Assign exact, stable, equal-frequency bins from lowest to highest CATE."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("scores must be a finite one-dimensional array")
    if not 2 <= bins <= len(values):
        raise ValueError("bins must be between two and the number of scores")
    order = np.argsort(values, kind="stable")
    labels = np.empty(len(values), dtype=int)
    for label, indices in enumerate(np.array_split(order, bins), start=1):
        labels[indices] = label
    return labels


def customer_segments(
    frame: pd.DataFrame,
    *,
    history_threshold: float,
    recency_threshold: float,
) -> pd.Series:
    history = np.where(frame["history"] >= history_threshold, "high history", "low history")
    recency = np.where(frame["recency"] <= recency_threshold, "recent", "inactive")
    labels = [
        f"{recency_value} / {history_value}"
        for recency_value, history_value in zip(recency, history, strict=True)
    ]
    return pd.Series(
        labels,
        index=frame.index,
        name="customer_segment",
    )
