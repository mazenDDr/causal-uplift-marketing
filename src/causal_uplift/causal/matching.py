from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MatchResult:
    pairs: pd.DataFrame
    caliper: float
    matched_treated: int
    matching_ratio: int
    eligible_treated: int
    eligible_controls: int
    discarded_outside_overlap: int


def propensity_logit(propensity: np.ndarray, *, epsilon: float = 1e-6) -> np.ndarray:
    """Convert probabilities to finite log odds."""
    values = np.asarray(propensity, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("propensity must be a finite one-dimensional array")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("propensity must be between zero and one")
    clipped = np.clip(values, epsilon, 1 - epsilon)
    return np.log(clipped / (1 - clipped))


def match_on_propensity(
    frame: pd.DataFrame,
    propensity: np.ndarray,
    *,
    replacement: bool,
    caliper_sd: float = 0.2,
    matching_ratio: int = 1,
    overlap_min: float = 0.05,
    overlap_max: float = 0.95,
    seed: int = 42,
) -> MatchResult:
    """Greedy 1:k nearest-neighbor matching on logit propensity."""
    scores = np.asarray(propensity, dtype=float)
    if scores.shape != (len(frame),):
        raise ValueError("propensity must have one value per row")
    if not 0 < overlap_min < overlap_max < 1:
        raise ValueError("overlap thresholds must satisfy 0 < min < max < 1")
    if caliper_sd <= 0:
        raise ValueError("caliper_sd must be positive")
    if matching_ratio < 1:
        raise ValueError("matching_ratio must be positive")
    if "source_row_id" not in frame or not frame["source_row_id"].is_unique:
        raise ValueError("source_row_id must exist and be unique")
    treatment = frame["treatment"].to_numpy(dtype=int)
    if set(np.unique(treatment)) != {0, 1}:
        raise ValueError("both binary treatment groups are required")

    logits = propensity_logit(scores)
    eligible = (scores >= overlap_min) & (scores <= overlap_max)
    eligible_positions = np.flatnonzero(eligible)
    treated_positions = eligible_positions[treatment[eligible_positions] == 1]
    control_positions = eligible_positions[treatment[eligible_positions] == 0]
    if len(treated_positions) == 0 or len(control_positions) == 0:
        raise ValueError("overlap trimming removed a treatment group")
    caliper = float(caliper_sd * np.std(logits[eligible_positions], ddof=1))
    if not np.isfinite(caliper) or caliper <= 0:
        raise ValueError("matching caliper is not positive and finite")

    order = np.argsort(logits[control_positions], kind="stable")
    available_scores = logits[control_positions][order].tolist()
    available_positions = control_positions[order].tolist()
    rng = np.random.default_rng(seed)
    treated_order = rng.permutation(treated_positions)
    rows: list[dict[str, int | float]] = []
    matched_treated = 0

    for treated_position in treated_order:
        if len(available_scores) < matching_ratio:
            break
        treated_score = logits[treated_position]
        insertion = bisect_left(available_scores, treated_score)
        left = insertion - 1
        right = insertion
        selected_indices = []
        while len(selected_indices) < matching_ratio and (
            left >= 0 or right < len(available_scores)
        ):
            candidates = []
            if left >= 0:
                candidates.append(left)
            if right < len(available_scores):
                candidates.append(right)
            selected = min(
                candidates,
                key=lambda index: (
                    abs(available_scores[index] - treated_score),
                    available_positions[index],
                ),
            )
            selected_indices.append(selected)
            if selected == left:
                left -= 1
            else:
                right += 1
        distances = [
            abs(available_scores[selected] - treated_score) for selected in selected_indices
        ]
        if len(selected_indices) < matching_ratio or max(distances) > caliper:
            continue
        for match_number, (selected, distance) in enumerate(
            zip(selected_indices, distances, strict=True), start=1
        ):
            control_position = available_positions[selected]
            rows.append(
                {
                    "pair_id": matched_treated,
                    "match_number": match_number,
                    "treated_position": int(treated_position),
                    "control_position": int(control_position),
                    "treated_source_row_id": int(frame.iloc[treated_position]["source_row_id"]),
                    "control_source_row_id": int(frame.iloc[control_position]["source_row_id"]),
                    "logit_distance": float(distance),
                }
            )
        if not replacement:
            for selected in sorted(selected_indices, reverse=True):
                available_scores.pop(selected)
                available_positions.pop(selected)
        matched_treated += 1

    pairs = pd.DataFrame(rows)
    if pairs.empty:
        raise ValueError("no matches satisfy the caliper")
    return MatchResult(
        pairs=pairs,
        caliper=caliper,
        matched_treated=matched_treated,
        matching_ratio=matching_ratio,
        eligible_treated=len(treated_positions),
        eligible_controls=len(control_positions),
        discarded_outside_overlap=int((~eligible).sum()),
    )


def matched_frame(frame: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Materialize one treated and one control row per matched pair."""
    required = {"pair_id", "treated_position", "control_position"}
    if not required.issubset(pairs):
        raise ValueError(f"pairs are missing columns: {sorted(required.difference(pairs))}")
    treated = frame.iloc[pairs["treated_position"].to_numpy(dtype=int)].copy()
    control = frame.iloc[pairs["control_position"].to_numpy(dtype=int)].copy()
    treated["pair_id"] = pairs["pair_id"].to_numpy(dtype=int)
    control["pair_id"] = pairs["pair_id"].to_numpy(dtype=int)
    return pd.concat([treated, control], ignore_index=True)


def paired_att(
    frame: pd.DataFrame,
    pairs: pd.DataFrame,
    outcome: str,
    *,
    bootstrap_samples: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Estimate ATT and a paired percentile-bootstrap confidence interval."""
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    treated = frame.iloc[pairs["treated_position"].to_numpy(dtype=int)][outcome].to_numpy(float)
    control = frame.iloc[pairs["control_position"].to_numpy(dtype=int)][outcome].to_numpy(float)
    pair_outcomes = pd.DataFrame(
        {"pair_id": pairs["pair_id"].to_numpy(dtype=int), "treated": treated, "control": control}
    )
    grouped = pair_outcomes.groupby("pair_id", sort=True)
    differences = grouped["treated"].first().to_numpy() - grouped["control"].mean().to_numpy()
    if len(differences) == 0:
        raise ValueError("at least one matched pair is required")
    rng = np.random.default_rng(seed)
    draws = np.empty(bootstrap_samples)
    for index in range(bootstrap_samples):
        draws[index] = rng.choice(differences, size=len(differences), replace=True).mean()
    lower, upper = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {
        "estimate": float(differences.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
    }
