from __future__ import annotations

from pathlib import Path

from causal_uplift.visualization.portfolio import build_portfolio


def main() -> None:
    written = build_portfolio(Path.cwd())
    for path in written:
        print(f"wrote {path.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
