from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import pytest

from scripts import download_hillstrom


def test_pinned_hash_matches_committed_experiment() -> None:
    repository = Path(__file__).resolve().parents[1]
    result = repository / "experiments/results/foundation_summary.json"
    foundation = json.loads(result.read_text())
    assert foundation["dataset"]["rows"] == download_hillstrom.EXPECTED_ROWS
    assert foundation["dataset"]["sha256"] == download_hillstrom.EXPECTED_SHA256


def test_dataset_requires_exact_checksum_even_with_expected_row_count(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = b"customer,outcome\n1,0\n2,1\n"
    path = tmp_path / "hillstrom.csv"
    path.write_bytes(expected)
    monkeypatch.setattr(download_hillstrom, "EXPECTED_ROWS", 2)
    monkeypatch.setattr(download_hillstrom, "EXPECTED_SHA256", hashlib.sha256(expected).hexdigest())

    assert download_hillstrom.matches_expected_dataset(path)
    path.write_bytes(b"customer,outcome\n1,1\n2,1\n")
    assert not download_hillstrom.matches_expected_dataset(path)


def test_mismatched_download_keeps_existing_data(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    expected = b"customer,outcome\n1,0\n2,1\n"
    path = tmp_path / "hillstrom.csv"
    path.write_bytes(expected)
    monkeypatch.setattr(download_hillstrom, "EXPECTED_ROWS", 2)
    monkeypatch.setattr(download_hillstrom, "EXPECTED_SHA256", hashlib.sha256(expected).hexdigest())
    monkeypatch.setattr(
        download_hillstrom.urllib.request,
        "urlopen",
        lambda *_, **__: io.BytesIO(b"customer,outcome\n1,1\n2,1\n"),
    )
    monkeypatch.setattr(sys, "argv", ["download_hillstrom.py", "--output", str(path), "--force"])

    with pytest.raises(ValueError, match="SHA-256"):
        download_hillstrom.main()

    assert path.read_bytes() == expected
    assert not path.with_suffix(".download").exists()
