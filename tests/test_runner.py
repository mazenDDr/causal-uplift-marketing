from __future__ import annotations

import json
from pathlib import Path

import pytest

from causal_uplift.evaluation.runner import (
    ExperimentStep,
    atomic_write_json,
    can_resume,
    completed_step_fingerprint,
    hash_files,
    step_fingerprint,
    topological_steps,
    validate_steps,
)


def steps() -> tuple[ExperimentStep, ...]:
    return (
        ExperimentStep("foundation", "foundation.py", (), ("foundation.json",)),
        ExperimentStep("model", "model.py", ("foundation",), ("model.json",)),
        ExperimentStep("policy", "policy.py", ("model",), ("policy.json",)),
    )


def test_requested_step_includes_dependencies_once() -> None:
    ordered = topological_steps(steps(), ["policy", "model"])
    assert [step.name for step in ordered] == ["foundation", "model", "policy"]


def test_unknown_dependency_and_cycle_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown dependencies"):
        validate_steps((ExperimentStep("model", "model.py", ("missing",), ("out",)),))
    cyclic = (
        ExperimentStep("a", "a.py", ("b",), ("a.json",)),
        ExperimentStep("b", "b.py", ("a",), ("b.json",)),
    )
    with pytest.raises(ValueError, match="cycle"):
        validate_steps(cyclic)


def test_content_hash_changes_with_file_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("first\n")
    first = hash_files([source], root=tmp_path)
    source.write_text("second\n")
    assert hash_files([source], root=tmp_path) != first


def test_step_fingerprint_includes_upstream_fingerprint() -> None:
    step = steps()[1]
    first = step_fingerprint(
        step,
        code_fingerprint="code",
        config_fingerprint="config",
        dependency_fingerprints={"foundation": "one"},
    )
    second = step_fingerprint(
        step,
        code_fingerprint="code",
        config_fingerprint="config",
        dependency_fingerprints={"foundation": "two"},
    )
    assert first != second


def test_completed_step_fingerprint_includes_output_bytes() -> None:
    first = completed_step_fingerprint("inputs-and-code", "output-one")
    second = completed_step_fingerprint("inputs-and-code", "output-two")
    assert first != second


def test_resume_requires_success_matching_fingerprint_and_outputs(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    record = {"status": "succeeded", "fingerprint": "same"}
    assert not can_resume(record, fingerprint="same", outputs=[output], root=tmp_path)
    output.write_text("{}\n")
    record["output_fingerprint"] = hash_files([output], root=tmp_path)
    assert can_resume(record, fingerprint="same", outputs=[output], root=tmp_path)
    assert not can_resume(record, fingerprint="changed", outputs=[output], root=tmp_path)
    assert not can_resume(
        {**record, "status": "failed"}, fingerprint="same", outputs=[output], root=tmp_path
    )
    output.write_text('{"corrupted": true}\n')
    assert not can_resume(record, fingerprint="same", outputs=[output], root=tmp_path)


def test_atomic_manifest_write_replaces_complete_json(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    atomic_write_json(path, {"status": "first"})
    atomic_write_json(path, {"status": "second", "steps": [1, 2]})
    assert json.loads(path.read_text()) == {"status": "second", "steps": [1, 2]}
    assert list(tmp_path.glob("*.tmp-*")) == []
