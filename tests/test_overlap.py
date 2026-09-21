from __future__ import annotations

import numpy as np
import pytest

from causal_uplift.evaluation.overlap import inverse_probability_weight_diagnostics


def test_constant_propensity_has_full_effective_sample_size() -> None:
    treatment = np.array([0, 1] * 50)
    result = inverse_probability_weight_diagnostics(treatment, np.full(100, 0.5))
    assert result["effective_sample_size"] == pytest.approx(100)
    assert result["effective_sample_fraction"] == pytest.approx(1)
    assert result["outside_overlap_fraction"] == 0


def test_rare_counter_assignment_creates_extreme_weight_and_lower_ess() -> None:
    treatment = np.array([1] * 49 + [0] + [0] * 49 + [1])
    propensity = np.array([0.99] * 50 + [0.01] * 50)
    result = inverse_probability_weight_diagnostics(treatment, propensity)
    assert result["outside_overlap_fraction"] == 1
    assert result["weight_max"] == pytest.approx(100)
    assert result["effective_sample_fraction"] < 0.1


def test_weight_diagnostics_caps_and_reports_boundary_probabilities() -> None:
    result = inverse_probability_weight_diagnostics(np.array([1, 0]), np.array([0.0, 1.0]))
    assert result["numerically_clipped_fraction"] == 1
    assert result["weight_max"] == pytest.approx(1_000_000)


def test_weight_diagnostics_rejects_values_outside_probability_range() -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        inverse_probability_weight_diagnostics(np.array([0, 1]), np.array([-0.1, 1.1]))
