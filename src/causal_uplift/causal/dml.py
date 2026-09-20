from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from econml.dml import LinearDML
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from causal_uplift.causal.baselines import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from causal_uplift.data.load import PRE_TREATMENT_COLUMNS

NUISANCE_CANDIDATES = ("random_forest_logistic", "hist_gradient_boosting")


@dataclass(frozen=True)
class LinearDMLResult:
    estimate: float
    ci_lower: float
    ci_upper: float
    cross_fit_folds: int


def make_confounder_encoder() -> ColumnTransformer:
    """Encode only the frozen pre-treatment feature set."""
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_cross_fit_splits(
    outcome: pd.Series | np.ndarray,
    treatment: pd.Series | np.ndarray,
    *,
    folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create shuffled folds stratified by joint outcome/treatment state."""
    y = np.asarray(outcome, dtype=int)
    t = np.asarray(treatment, dtype=int)
    if y.shape != t.shape or y.ndim != 1:
        raise ValueError("outcome and treatment must be aligned one-dimensional arrays")
    if folds < 2:
        raise ValueError("cross-fitting requires at least two folds")
    strata = t * 2 + y
    counts = np.bincount(strata, minlength=4)
    if counts.min() < folds:
        raise ValueError("every joint outcome/treatment stratum must cover every fold")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    placeholder = np.zeros(len(y))
    return list(splitter.split(placeholder, strata))


def make_nuisance_models(
    name: str,
    *,
    seed: int,
    random_forest_trees: int = 200,
    min_samples_leaf: int = 100,
    max_iter: int = 200,
) -> tuple[RegressorMixin, object]:
    if name == "random_forest_logistic":
        return (
            RandomForestRegressor(
                n_estimators=random_forest_trees,
                min_samples_leaf=min_samples_leaf,
                n_jobs=-1,
                random_state=seed,
            ),
            LogisticRegression(max_iter=2000, random_state=seed),
        )
    if name == "hist_gradient_boosting":
        return (
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=max_iter,
                max_leaf_nodes=15,
                min_samples_leaf=min_samples_leaf,
                l2_regularization=1.0,
                random_state=seed,
            ),
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=max_iter,
                max_leaf_nodes=15,
                min_samples_leaf=min_samples_leaf,
                l2_regularization=1.0,
                random_state=seed,
            ),
        )
    raise ValueError(f"unknown nuisance candidate: {name}")


def nuisance_cross_fit_metrics(
    frame: pd.DataFrame,
    outcome: str,
    *,
    candidate: str,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    random_forest_trees: int = 200,
    min_samples_leaf: int = 100,
    max_iter: int = 200,
) -> dict[str, float]:
    encoder = make_confounder_encoder()
    confounders = encoder.fit_transform(frame.loc[:, PRE_TREATMENT_COLUMNS])
    y = frame[outcome].to_numpy(dtype=float)
    t = frame["treatment"].to_numpy(dtype=int)
    model_y, model_t = make_nuisance_models(
        candidate,
        seed=seed,
        random_forest_trees=random_forest_trees,
        min_samples_leaf=min_samples_leaf,
        max_iter=max_iter,
    )
    outcome_prediction = cross_val_predict(model_y, confounders, y, cv=splits, method="predict")
    treatment_probability = cross_val_predict(
        model_t,
        confounders,
        t,
        cv=splits,
        method="predict_proba",
    )[:, 1]
    clipped_outcome = np.clip(outcome_prediction, 0, 1)
    outcome_brier = brier_score_loss(y, clipped_outcome)
    treatment_brier = brier_score_loss(t, treatment_probability)
    outcome_null_brier = brier_score_loss(y, np.full(len(y), y.mean()))
    treatment_null_brier = brier_score_loss(t, np.full(len(t), t.mean()))
    return {
        "outcome_brier": float(outcome_brier),
        "outcome_brier_vs_null": float(outcome_brier / outcome_null_brier),
        "treatment_roc_auc": float(roc_auc_score(t, treatment_probability)),
        "treatment_brier": float(treatment_brier),
        "treatment_brier_vs_null": float(treatment_brier / treatment_null_brier),
        "treatment_log_loss": float(log_loss(t, treatment_probability)),
        "selection_loss": float(
            outcome_brier / outcome_null_brier + treatment_brier / treatment_null_brier
        ),
    }


def fit_linear_dml(
    frame: pd.DataFrame,
    outcome: str,
    *,
    candidate: str,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    random_forest_trees: int = 200,
    min_samples_leaf: int = 100,
    max_iter: int = 200,
) -> LinearDMLResult:
    """Fit a constant-effect LinearDML with explicit cross-fitting."""
    encoder = make_confounder_encoder()
    confounders = encoder.fit_transform(frame.loc[:, PRE_TREATMENT_COLUMNS])
    model_y, model_t = make_nuisance_models(
        candidate,
        seed=seed,
        random_forest_trees=random_forest_trees,
        min_samples_leaf=min_samples_leaf,
        max_iter=max_iter,
    )
    estimator = LinearDML(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=True,
        cv=splits,
        random_state=seed,
    )
    estimator.fit(
        frame[outcome].to_numpy(dtype=float),
        frame["treatment"].to_numpy(dtype=int),
        W=confounders,
        inference="statsmodels",
    )
    estimate = float(np.asarray(estimator.ate()).squeeze())
    lower, upper = estimator.ate_interval(alpha=0.05)
    return LinearDMLResult(
        estimate=estimate,
        ci_lower=float(np.asarray(lower).squeeze()),
        ci_upper=float(np.asarray(upper).squeeze()),
        cross_fit_folds=len(splits),
    )
