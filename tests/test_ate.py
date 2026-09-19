from __future__ import annotations

import pandas as pd

from causal_uplift.evaluation.ate import (
    bootstrap_difference_in_means,
    difference_in_means,
)


def test_difference_in_means_known_effect() -> None:
    frame = pd.DataFrame({"treatment": [0, 0, 1, 1], "conversion": [0, 0, 0, 1]})
    assert difference_in_means(frame, "conversion") == 0.5


def test_bootstrap_is_reproducible() -> None:
    frame = pd.DataFrame(
        {"treatment": [0] * 50 + [1] * 50, "conversion": [0, 1] * 25 + [0, 1] * 25}
    )
    first = bootstrap_difference_in_means(frame, "conversion", samples=100, seed=9)
    second = bootstrap_difference_in_means(frame, "conversion", samples=100, seed=9)
    assert first == second
