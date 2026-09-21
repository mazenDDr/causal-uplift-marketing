from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

ORIGINAL_URL = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
DEFAULT_URL = (
    "https://raw.githubusercontent.com/AdityaDabrase/ab-testing-email-marketing/"
    "main/data/raw/hillstrom.csv"
)
EXPECTED_ROWS = 64_000
EXPECTED_SHA256 = "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def data_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def matches_expected_dataset(path: Path) -> bool:
    return path.is_file() and data_rows(path) == EXPECTED_ROWS and sha256(path) == EXPECTED_SHA256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, default=Path("data/raw/hillstrom.csv"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    needs_download = args.force or not matches_expected_dataset(args.output)
    if needs_download:
        temporary = args.output.with_suffix(".download")
        with urllib.request.urlopen(args.url, timeout=30) as response, temporary.open("wb") as sink:
            shutil.copyfileobj(response, sink)
        downloaded_rows = data_rows(temporary)
        downloaded_sha256 = sha256(temporary)
        if downloaded_rows != EXPECTED_ROWS or downloaded_sha256 != EXPECTED_SHA256:
            temporary.unlink()
            raise ValueError(
                f"downloaded {downloaded_rows:,} rows and SHA-256 {downloaded_sha256}; "
                f"expected {EXPECTED_ROWS:,} rows and {EXPECTED_SHA256}; "
                "existing data kept"
            )
        temporary.replace(args.output)
    metadata = {
        "path": str(args.output),
        "rows": data_rows(args.output),
        "bytes": args.output.stat().st_size,
        "sha256": sha256(args.output),
        "download_url": args.url,
        "original_publisher_url": ORIGINAL_URL,
    }
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
