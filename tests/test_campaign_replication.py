from __future__ import annotations

import pandas as pd

from causal_uplift.data.load import campaign_frame
from causal_uplift.data.preprocess import randomized_train_evaluation_split


def hillstrom_groups() -> pd.DataFrame:
    rows = []
    for source_id, segment in enumerate(["Mens E-Mail", "Womens E-Mail", "No E-Mail"] * 40):
        rows.append(
            {
                "source_row_id": source_id,
                "segment": segment,
                "treatment": -1,
                "conversion": source_id % 2,
            }
        )
    return pd.DataFrame(rows)


def test_womens_campaign_excludes_mens_email_rows() -> None:
    frame = campaign_frame(hillstrom_groups(), treatment_label="Womens E-Mail")
    assert set(frame["segment"]) == {"Womens E-Mail", "No E-Mail"}
    assert set(frame.loc[frame["treatment"] == 1, "segment"]) == {"Womens E-Mail"}
    assert set(frame.loc[frame["treatment"] == 0, "segment"]) == {"No E-Mail"}


def test_womens_split_is_disjoint_and_campaign_specific() -> None:
    frame = campaign_frame(hillstrom_groups(), treatment_label="Womens E-Mail")
    training, evaluation = randomized_train_evaluation_split(frame, seed=42)
    assert set(training["source_row_id"]).isdisjoint(evaluation["source_row_id"])
    assert len(training) == 48
    assert len(evaluation) == 32
