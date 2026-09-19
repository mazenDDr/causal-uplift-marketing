from __future__ import annotations

import math

import pandas as pd
import pytest

from causal_uplift.evaluation.balance import covariate_smds


def test_smd_expands_categorical_levels() -> None:
    frame = pd.DataFrame(
        {
            "treatment": [0, 0, 1, 1],
            "history": [0.0, 2.0, 2.0, 4.0],
            "channel": ["Phone", "Web", "Phone", "Phone"],
        }
    )
    smds = covariate_smds(frame, ("history", "channel"))
    assert smds["history"] == pytest.approx(math.sqrt(2.0))
    assert set(smds) == {"history", "channel=Phone", "channel=Web"}
