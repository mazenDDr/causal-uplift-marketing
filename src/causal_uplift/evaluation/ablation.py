from __future__ import annotations


def summarize_ate_estimators(
    *,
    rct_ate: float,
    naive: float,
    linear_dml: float,
    causal_forest: float,
    uplift_forest: float,
    psm_att: float,
) -> dict[str, dict[str, float | str | None]]:
    """Separate like-for-like ATE errors from PSM's different ATT estimand."""
    ate = {
        "naive": naive,
        "linear_dml": linear_dml,
        "causal_forest_dml": causal_forest,
        "uplift_random_forest": uplift_forest,
    }
    result = {
        name: {
            "estimand": "ATE on the RCT covariate distribution",
            "estimate": estimate,
            "absolute_error_vs_rct_ate": abs(estimate - rct_ate),
        }
        for name, estimate in ate.items()
    }
    result["psm"] = {
        "estimand": "ATT in matched observational treated customers",
        "estimate": psm_att,
        "absolute_error_vs_rct_ate": None,
        "reference_gap_vs_overall_rct_ate": psm_att - rct_ate,
    }
    return result
