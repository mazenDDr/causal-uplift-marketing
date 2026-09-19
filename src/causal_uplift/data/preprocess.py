from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def randomized_train_evaluation_split(
    frame: pd.DataFrame,
    *,
    evaluation_fraction: float = 0.40,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split before confounding injection, preserving randomized treatment proportions."""
    if not 0 < evaluation_fraction < 1:
        raise ValueError("evaluation_fraction must be strictly between zero and one")
    if "source_row_id" not in frame or "treatment" not in frame:
        raise ValueError("frame must include source_row_id and treatment")

    train, evaluation = train_test_split(
        frame,
        test_size=evaluation_fraction,
        random_state=seed,
        stratify=frame["treatment"],
    )
    train_ids = set(train["source_row_id"])
    evaluation_ids = set(evaluation["source_row_id"])
    if train_ids & evaluation_ids:
        raise AssertionError("training and randomized evaluation IDs overlap")
    return train.sort_values("source_row_id").reset_index(drop=True), evaluation.sort_values(
        "source_row_id"
    ).reset_index(drop=True)
