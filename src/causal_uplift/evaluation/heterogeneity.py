from __future__ import annotations

import numpy as np
import pandas as pd

from causal_uplift.evaluation.ate import bootstrap_difference_in_means


def randomized_group_effects(
    frame: pd.DataFrame,
    groups: pd.Series | np.ndarray,
    predicted_effect: np.ndarray,
    *,
    outcome: str = "conversion",
    bootstrap_samples: int = 2000,
    seed: int = 42,
) -> list[dict[str, float | int | str]]:
    """Validate predicted CATE groups using treatment contrasts in randomized data."""
    labels = np.asarray(groups)
    scores = np.asarray(predicted_effect, dtype=float)
    if labels.shape != (len(frame),) or scores.shape != (len(frame),):
        raise ValueError("groups and predicted effects must align with the frame")
    results = []
    for label in pd.unique(labels):
        selected = labels == label
        group = frame.loc[selected]
        effect = bootstrap_difference_in_means(
            group,
            outcome,
            samples=bootstrap_samples,
            seed=seed,
        )
        results.append(
            {
                "group": str(label),
                "rows": int(selected.sum()),
                "treated_rows": int((group["treatment"] == 1).sum()),
                "control_rows": int((group["treatment"] == 0).sum()),
                "predicted_cate_mean": float(scores[selected].mean()),
                "observed_rct_uplift": effect["estimate"],
                "observed_ci_lower": effect["ci_lower"],
                "observed_ci_upper": effect["ci_upper"],
            }
        )
    return results
