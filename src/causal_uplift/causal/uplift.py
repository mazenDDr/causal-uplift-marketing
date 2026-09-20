from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from causalml.inference.tree import (
    UpliftRandomForestClassifier,
    UpliftTreeClassifier,
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from causal_uplift.causal.baselines import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from causal_uplift.data.load import PRE_TREATMENT_COLUMNS


def make_uplift_encoder() -> ColumnTransformer:
    """Create a dense, tree-friendly encoder from pre-treatment features only."""
    return ColumnTransformer(
        [
            ("numeric", "passthrough", NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def treatment_labels(
    treatment: pd.Series | np.ndarray,
    *,
    control_name: str,
    treatment_name: str,
) -> np.ndarray:
    values = np.asarray(treatment, dtype=int)
    if set(np.unique(values)) != {0, 1}:
        raise ValueError("both binary treatment groups are required")
    return np.where(values == 1, treatment_name, control_name)


@dataclass
class FittedUpliftModel:
    model: object
    encoder: ColumnTransformer
    feature_names: tuple[str, ...]
    control_name: str
    treatment_name: str

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        features = self.encoder.transform(frame.loc[:, PRE_TREATMENT_COLUMNS])
        prediction = np.asarray(self.model.predict(features), dtype=float)
        classes = [str(group) for group in self.model.classes_]
        if prediction.ndim == 2 and prediction.shape[1] == len(classes) and len(classes) > 1:
            prediction = (
                prediction[:, classes.index(self.treatment_name)]
                - prediction[:, classes.index(self.control_name)]
            )
        if prediction.ndim == 2 and prediction.shape[1] == 1:
            prediction = prediction[:, 0]
        if prediction.shape != (len(frame),) or not np.isfinite(prediction).all():
            raise AssertionError("uplift model must return one finite score per row")
        return prediction


def _training_arrays(
    frame: pd.DataFrame,
    *,
    control_name: str,
    treatment_name: str,
) -> tuple[ColumnTransformer, np.ndarray, np.ndarray, np.ndarray]:
    encoder = make_uplift_encoder()
    features = encoder.fit_transform(frame.loc[:, PRE_TREATMENT_COLUMNS])
    labels = treatment_labels(
        frame["treatment"],
        control_name=control_name,
        treatment_name=treatment_name,
    )
    outcome = frame["conversion"].to_numpy(dtype=int)
    return encoder, features, labels, outcome


def fit_uplift_tree(
    frame: pd.DataFrame,
    *,
    control_name: str,
    treatment_name: str,
    evaluation_function: str,
    honesty: bool,
    estimation_sample_size: float,
    max_depth: int,
    min_samples_leaf: int,
    min_samples_treatment: int,
    n_reg: int,
    seed: int,
) -> FittedUpliftModel:
    encoder, features, labels, outcome = _training_arrays(
        frame,
        control_name=control_name,
        treatment_name=treatment_name,
    )
    model = UpliftTreeClassifier(
        control_name=control_name,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_treatment=min_samples_treatment,
        n_reg=n_reg,
        evaluationFunction=evaluation_function,
        normalization=True,
        honesty=honesty,
        estimation_sample_size=estimation_sample_size,
        random_state=seed,
    )
    model.fit(features, labels, outcome)
    return FittedUpliftModel(
        model,
        encoder,
        tuple(encoder.get_feature_names_out()),
        control_name,
        treatment_name,
    )


def fit_uplift_forest(
    frame: pd.DataFrame,
    *,
    control_name: str,
    treatment_name: str,
    evaluation_function: str,
    honesty: bool,
    estimation_sample_size: float,
    n_estimators: int,
    max_depth: int,
    max_features: int,
    min_samples_leaf: int,
    min_samples_treatment: int,
    n_reg: int,
    n_jobs: int,
    seed: int,
) -> FittedUpliftModel:
    encoder, features, labels, outcome = _training_arrays(
        frame,
        control_name=control_name,
        treatment_name=treatment_name,
    )
    model = UpliftRandomForestClassifier(
        control_name=control_name,
        n_estimators=n_estimators,
        max_features=max_features,
        random_state=seed,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_treatment=min_samples_treatment,
        n_reg=n_reg,
        evaluationFunction=evaluation_function,
        normalization=True,
        honesty=honesty,
        estimation_sample_size=estimation_sample_size,
        n_jobs=n_jobs,
    )
    model.fit(features, labels, outcome)
    return FittedUpliftModel(
        model,
        encoder,
        tuple(encoder.get_feature_names_out()),
        control_name,
        treatment_name,
    )


def uplift_tree_leaves(
    fitted: FittedUpliftModel,
    *,
    control_name: str,
    treatment_name: str,
) -> list[dict[str, object]]:
    """Extract paths, group rates, uplift, and sample counts from a fitted tree."""
    leaves: list[dict[str, object]] = []

    def visit(node: object, path: list[str]) -> None:
        if node.results is not None:
            rates = {
                str(group): float(rate)
                for group, rate in zip(node.classes_, node.results, strict=True)
            }
            group_samples = {
                group: int(count)
                for group, count in re.findall(r"([^\s:]+):\s*(\d+)", node.summary["group_size"])
            }
            leaves.append(
                {
                    "path": " and ".join(path) if path else "all customers",
                    "samples": int(node.summary["samples"]),
                    "group_samples": group_samples,
                    "conversion_rates": rates,
                    "estimated_uplift": rates[treatment_name] - rates[control_name],
                }
            )
            return
        feature = fitted.feature_names[int(node.col)]
        operator = ">=" if isinstance(node.value, (int, float)) else "=="
        condition = f"{feature} {operator} {node.value}"
        visit(node.trueBranch, [*path, condition])
        visit(node.falseBranch, [*path, f"not ({condition})"])

    visit(fitted.model.fitted_uplift_tree, [])
    return leaves
