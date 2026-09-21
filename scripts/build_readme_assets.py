from __future__ import annotations

import argparse
from pathlib import Path

from causal_uplift.visualization.readme import (
    comparison_markdown,
    load_readme_results,
    plot_headline_summary,
    replace_generated_comparison,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build generated README evidence")
    parser.add_argument(
        "--table-only",
        action="store_true",
        help="refresh the Markdown block without rendering the figure",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = Path.cwd()
    results = load_readme_results(repository)
    if not args.table_only:
        plot_headline_summary(results, repository / "experiments/figures/headline_summary.svg")
    replace_generated_comparison(repository / "README.md", comparison_markdown(results))
    if args.table_only:
        print("refreshed README comparison")
    else:
        print("wrote experiments/figures/headline_summary.svg and refreshed README comparison")


if __name__ == "__main__":
    main()
