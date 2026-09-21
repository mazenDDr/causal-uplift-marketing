from __future__ import annotations

import pytest

from causal_uplift.evaluation.failures import ratio, validate_failure_cases


def valid_cases() -> list[dict]:
    return [
        {
            "id": failure_id,
            "method": "method",
            "failure": "failure",
            "symptom": "symptom",
            "cause": "cause",
            "diagnostic": "diagnostic",
            "fix": "fix",
            "estimand": (
                "ATT in retained matched treated customers"
                if failure_id == "psm_overlap"
                else "ATE"
            ),
            "evidence": {"measured": 1},
            "source_artifacts": ["result.json"],
        }
        for failure_id in (
            "naive_confounding",
            "psm_overlap",
            "dml_overlap",
            "uplift_forest_ranking",
        )
    ]


def test_complete_failure_taxonomy_is_accepted() -> None:
    validate_failure_cases(valid_cases())


def test_psm_att_cannot_be_mislabeled_as_ate_error() -> None:
    cases = valid_cases()
    psm = next(case for case in cases if case["id"] == "psm_overlap")
    psm["evidence"]["ate_error"] = 0.01
    with pytest.raises(ValueError, match="ATT"):
        validate_failure_cases(cases)


def test_incomplete_taxonomy_is_rejected() -> None:
    cases = valid_cases()
    cases.pop()
    with pytest.raises(ValueError, match="failure IDs"):
        validate_failure_cases(cases)


def test_ratio_rejects_undefined_comparison() -> None:
    assert ratio(2, 4) == 0.5
    with pytest.raises(ValueError, match="denominator"):
        ratio(1, 0)
