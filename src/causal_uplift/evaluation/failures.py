from __future__ import annotations

from typing import Any

REQUIRED_FAILURE_IDS = {
    "naive_confounding",
    "psm_overlap",
    "dml_overlap",
    "uplift_forest_ranking",
}


def validate_failure_cases(cases: list[dict[str, Any]]) -> None:
    """Reject incomplete or causally misleading failure-analysis records."""
    ids = {case.get("id") for case in cases}
    if ids != REQUIRED_FAILURE_IDS:
        raise ValueError(f"failure IDs must be exactly {sorted(REQUIRED_FAILURE_IDS)}")

    required_text = ("method", "failure", "symptom", "cause", "diagnostic", "fix")
    for case in cases:
        missing = [field for field in required_text if not case.get(field)]
        if missing:
            raise ValueError(f"{case['id']} is missing required fields: {missing}")
        if not case.get("evidence"):
            raise ValueError(f"{case['id']} must include measured evidence")
        if not case.get("source_artifacts"):
            raise ValueError(f"{case['id']} must cite source artifacts")

    psm = next(case for case in cases if case["id"] == "psm_overlap")
    if psm.get("estimand") != "ATT in retained matched treated customers":
        raise ValueError("PSM failure must retain its ATT estimand label")
    if "ate_error" in psm["evidence"]:
        raise ValueError("PSM ATT must not be reported as an ATE error")


def ratio(numerator: float, denominator: float) -> float:
    """Return a finite diagnostic ratio and reject an undefined comparison."""
    if denominator == 0:
        raise ValueError("diagnostic ratio denominator cannot be zero")
    return float(numerator / denominator)
