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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def data_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, default=Path("data/raw/hillstrom.csv"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    needs_download = (
        args.force or not args.output.exists() or data_rows(args.output) != EXPECTED_ROWS
    )
    if needs_download:
        temporary = args.output.with_suffix(".download")
        with urllib.request.urlopen(args.url, timeout=30) as response, temporary.open("wb") as sink:
            shutil.copyfileobj(response, sink)
        downloaded_rows = data_rows(temporary)
        if downloaded_rows != EXPECTED_ROWS:
            temporary.unlink()
            raise ValueError(
                f"downloaded {downloaded_rows:,} rows; expected {EXPECTED_ROWS:,}; "
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
