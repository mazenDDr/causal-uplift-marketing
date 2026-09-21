from __future__ import annotations

import numpy as np


def inverse_probability_weight_diagnostics(
    treatment: np.ndarray,
    propensity: np.ndarray,
    *,
    overlap_min: float = 0.05,
    overlap_max: float = 0.95,
    epsilon: float = 1e-6,
) -> dict[str, float]:
    """Summarize how limited treatment overlap concentrates inverse-probability weights."""
    treatment = np.asarray(treatment, dtype=int)
    propensity = np.asarray(propensity, dtype=float)
    if treatment.ndim != 1 or propensity.shape != treatment.shape:
        raise ValueError("treatment and propensity must be aligned one-dimensional arrays")
    if set(np.unique(treatment)) != {0, 1}:
        raise ValueError("both binary treatment groups are required")
    if not np.isfinite(propensity).all() or ((propensity < 0) | (propensity > 1)).any():
        raise ValueError("propensity must contain finite probabilities between zero and one")
    if not 0 < overlap_min < overlap_max < 1:
        raise ValueError("overlap thresholds must satisfy 0 < min < max < 1")
    if not 0 < epsilon < 0.5:
        raise ValueError("epsilon must be between zero and one half")

    numerically_clipped = (propensity < epsilon) | (propensity > 1 - epsilon)
    safe_propensity = np.clip(propensity, epsilon, 1 - epsilon)
    weights = np.where(treatment == 1, 1 / safe_propensity, 1 / (1 - safe_propensity))

    def effective_sample_size(values: np.ndarray) -> float:
        return float(values.sum() ** 2 / np.square(values).sum())

    treated_weights = weights[treatment == 1]
    control_weights = weights[treatment == 0]
    effective_rows = effective_sample_size(weights)
    treated_effective_rows = effective_sample_size(treated_weights)
    control_effective_rows = effective_sample_size(control_weights)
    return {
        "numerically_clipped_fraction": float(numerically_clipped.mean()),
        "outside_overlap_fraction": float(
            ((propensity < overlap_min) | (propensity > overlap_max)).mean()
        ),
        "weight_p99": float(np.quantile(weights, 0.99)),
        "weight_max": float(weights.max()),
        "effective_sample_size": effective_rows,
        "effective_sample_fraction": effective_rows / len(weights),
        "treated_effective_sample_size": treated_effective_rows,
        "treated_effective_sample_fraction": treated_effective_rows / len(treated_weights),
        "control_effective_sample_size": control_effective_rows,
        "control_effective_sample_fraction": control_effective_rows / len(control_weights),
    }
