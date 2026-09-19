from __future__ import annotations

import pandas as pd

from causal_uplift.data.preprocess import randomized_train_evaluation_split


def test_randomized_holdout_is_disjoint_and_repeatable() -> None:
    frame = pd.DataFrame(
        {
            "source_row_id": range(1000),
            "treatment": [0, 1] * 500,
            "conversion": [0] * 1000,
        }
    )
    train_a, evaluation_a = randomized_train_evaluation_split(frame, seed=42)
    train_b, evaluation_b = randomized_train_evaluation_split(frame, seed=42)
    assert set(train_a["source_row_id"]).isdisjoint(evaluation_a["source_row_id"])
    assert len(train_a) == 600
    assert len(evaluation_a) == 400
    pd.testing.assert_frame_equal(train_a, train_b)
    pd.testing.assert_frame_equal(evaluation_a, evaluation_b)
