from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentStep:
    name: str
    script: str
    dependencies: tuple[str, ...]
    outputs: tuple[str, ...]


def validate_steps(steps: tuple[ExperimentStep, ...]) -> None:
    names = [step.name for step in steps]
    if len(names) != len(set(names)):
        raise ValueError("experiment step names must be unique")
    known = set(names)
    for step in steps:
        missing = set(step.dependencies).difference(known)
        if missing:
            raise ValueError(f"{step.name} has unknown dependencies: {sorted(missing)}")
        if step.name in step.dependencies:
            raise ValueError(f"{step.name} cannot depend on itself")
        if not step.outputs:
            raise ValueError(f"{step.name} must declare at least one output")
    topological_steps(steps, names)


def topological_steps(
    steps: tuple[ExperimentStep, ...],
    requested: Iterable[str],
) -> list[ExperimentStep]:
    """Return requested steps and dependencies once, in dependency order."""
    by_name = {step.name: step for step in steps}
    requested_names = list(requested)
    unknown = set(requested_names).difference(by_name)
    if unknown:
        raise ValueError(f"unknown experiment steps: {sorted(unknown)}")

    ordered: list[ExperimentStep] = []
    permanent: set[str] = set()
    temporary: set[str] = set()

    def visit(name: str) -> None:
        if name in permanent:
            return
        if name in temporary:
            raise ValueError(f"experiment dependency cycle includes {name}")
        temporary.add(name)
        for dependency in by_name[name].dependencies:
            visit(dependency)
        temporary.remove(name)
        permanent.add(name)
        ordered.append(by_name[name])

    for name in requested_names:
        visit(name)
    return ordered


def hash_files(paths: Iterable[Path], *, root: Path) -> str:
    """Hash file paths and bytes in stable repository-relative order."""
    digest = hashlib.sha256()
    resolved = sorted({path.resolve() for path in paths})
    for path in resolved:
        if not path.is_file():
            raise FileNotFoundError(path)
        relative = path.relative_to(root.resolve())
        digest.update(str(relative).encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def step_fingerprint(
    step: ExperimentStep,
    *,
    code_fingerprint: str,
    config_fingerprint: str,
    dependency_fingerprints: dict[str, str],
) -> str:
    payload = {
        "name": step.name,
        "script": step.script,
        "outputs": step.outputs,
        "code_fingerprint": code_fingerprint,
        "config_fingerprint": config_fingerprint,
        "dependencies": {name: dependency_fingerprints[name] for name in step.dependencies},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def completed_step_fingerprint(fingerprint: str, output_fingerprint: str) -> str:
    """Bind a step's inputs and implementation to the exact bytes it produced."""
    payload = {"fingerprint": fingerprint, "output_fingerprint": output_fingerprint}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def can_resume(
    record: dict | None,
    *,
    fingerprint: str,
    outputs: Iterable[Path],
    root: Path,
) -> bool:
    output_paths = list(outputs)
    if not (
        record
        and record.get("status") == "succeeded"
        and record.get("fingerprint") == fingerprint
        and all(path.is_file() for path in output_paths)
    ):
        return False
    return record.get("output_fingerprint") == hash_files(output_paths, root=root)


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)
