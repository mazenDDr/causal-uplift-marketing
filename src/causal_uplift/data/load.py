from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "recency",
    "history",
    "history_segment",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
    "segment",
    "visit",
    "conversion",
    "spend",
}

PRE_TREATMENT_COLUMNS = (
    "recency",
    "history",
    "history_segment",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
)
OUTCOME_COLUMNS = ("visit", "conversion", "spend")


def load_hillstrom(path: str | Path) -> pd.DataFrame:
    """Load Hillstrom and add an immutable source-row identifier."""
    frame = pd.read_csv(path)
    frame.columns = [column.strip().lower() for column in frame.columns]
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Hillstrom data is missing required columns: {sorted(missing)}")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Hillstrom required columns contain missing values")
    frame = frame.reset_index(drop=True)
    frame.insert(0, "source_row_id", frame.index.astype("int64"))
    return frame


def campaign_frame(
    frame: pd.DataFrame,
    *,
    treatment_label: str = "Mens E-Mail",
    control_label: str = "No E-Mail",
) -> pd.DataFrame:
    """Restrict to one campaign versus control and encode binary treatment."""
    selected = frame.loc[frame["segment"].isin([treatment_label, control_label])].copy()
    if selected.empty or selected["segment"].nunique() != 2:
        observed = sorted(frame["segment"].astype(str).unique())
        raise ValueError(
            f"Could not find both {treatment_label!r} and {control_label!r}; observed {observed}"
        )
    selected["treatment"] = (selected["segment"] == treatment_label).astype("int8")
    return selected.reset_index(drop=True)
