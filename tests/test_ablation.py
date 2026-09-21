import pytest

from causal_uplift.evaluation.ablation import summarize_ate_estimators


def test_ate_ablation_does_not_mislabel_psm_att_as_ate_error() -> None:
    result = summarize_ate_estimators(
        rct_ate=0.02,
        naive=0.08,
        linear_dml=0.03,
        causal_forest=0.025,
        uplift_forest=0.04,
        psm_att=0.015,
    )
    assert result["naive"]["absolute_error_vs_rct_ate"] == pytest.approx(0.06)
    assert result["causal_forest_dml"]["absolute_error_vs_rct_ate"] == pytest.approx(0.005)
    assert result["psm"]["absolute_error_vs_rct_ate"] is None
    assert result["psm"]["estimand"].startswith("ATT")
    assert result["psm"]["reference_gap_vs_overall_rct_ate"] == pytest.approx(-0.005)
