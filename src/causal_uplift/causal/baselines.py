from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from causal_uplift.data.load import OUTCOME_COLUMNS, PRE_TREATMENT_COLUMNS

CATEGORICAL_FEATURES = ("history_segment", "zip_code", "channel")
NUMERIC_FEATURES = tuple(
    column for column in PRE_TREATMENT_COLUMNS if column not in CATEGORICAL_FEATURES
)
FORBIDDEN_FEATURES = frozenset((*OUTCOME_COLUMNS, "segment", "desired_propensity"))


def validate_model_features(features: tuple[str, ...]) -> None:
    """Reject outcome, treatment-assignment, and unknown feature leakage."""
    leaking = set(features) & FORBIDDEN_FEATURES
    unknown = set(features) - set(PRE_TREATMENT_COLUMNS) - {"treatment"}
    if leaking or unknown:
        raise ValueError(
            f"invalid model features; leaking={sorted(leaking)}, unknown={sorted(unknown)}"
        )


def _pipeline(*, include_treatment: bool, seed: int) -> Pipeline:
    numeric = (*NUMERIC_FEATURES, "treatment") if include_treatment else NUMERIC_FEATURES
    features = (*PRE_TREATMENT_COLUMNS, "treatment") if include_treatment else PRE_TREATMENT_COLUMNS
    validate_model_features(features)
    preprocess = ColumnTransformer(
        [
            ("numeric", "passthrough", numeric),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=100,
        l2_regularization=1.0,
        random_state=seed,
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


@dataclass
class ResponseModel:
    """Predict conversion probability without representing the intervention."""

    seed: int = 42

    def __post_init__(self) -> None:
        self.pipeline = _pipeline(include_treatment=False, seed=self.seed)

    def fit(self, frame: pd.DataFrame, outcome: str = "conversion") -> ResponseModel:
        self.pipeline.fit(frame, frame[outcome])
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(frame)[:, 1]


@dataclass
class TreatmentFeatureModel:
    """A predictive model whose treatment toggle is not a causal adjustment."""

    seed: int = 42

    def __post_init__(self) -> None:
        self.pipeline = _pipeline(include_treatment=True, seed=self.seed)

    def fit(self, frame: pd.DataFrame, outcome: str = "conversion") -> TreatmentFeatureModel:
        self.pipeline.fit(frame, frame[outcome])
        return self

    def potential_predictions(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        treated = frame.copy()
        control = frame.copy()
        treated["treatment"] = 1
        control["treatment"] = 0
        return (
            self.pipeline.predict_proba(treated)[:, 1],
            self.pipeline.predict_proba(control)[:, 1],
        )

    def predict_uplift(self, frame: pd.DataFrame) -> np.ndarray:
        treated, control = self.potential_predictions(frame)
        return treated - control


def classification_metrics(y_true: pd.Series, probability: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "brier_score": float(brier_score_loss(y_true, probability)),
    }
