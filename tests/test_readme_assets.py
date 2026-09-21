from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from causal_uplift.visualization.readme import (
    comparison_markdown,
    load_readme_results,
    replace_generated_comparison,
)


@pytest.fixture(scope="module")
def results() -> dict:
    return load_readme_results(Path(__file__).resolve().parents[1])


def test_comparison_is_derived_from_measured_results(results: dict) -> None:
    markdown = comparison_markdown(results)
    assert "| Naive association | Raw association | 1.98 | — | — |" in markdown
    assert "| LinearDML | Constant ATE | 1.37 | — | $184 [-$3, $374] |" in markdown
    assert "0.034 [-0.874, 0.873]" in markdown
    assert markdown.count("<!-- BEGIN GENERATED FINAL COMPARISON -->") == 1
    assert markdown.count("<!-- END GENERATED FINAL COMPARISON -->") == 1


def test_headline_svg_is_valid() -> None:
    repository = Path(__file__).resolve().parents[1]
    root = ET.parse(repository / "experiments" / "figures" / "headline_summary.svg").getroot()
    assert root.tag.endswith("svg")


def test_readme_local_links_exist() -> None:
    repository = Path(__file__).resolve().parents[1]
    readme = (repository / "README.md").read_text()
    targets = re.findall(r"!?\[[^]]*\]\(([^)]+)\)", readme)
    local_targets = [target.split("#", maxsplit=1)[0] for target in targets if "://" not in target]
    assert local_targets
    assert all((repository / target).exists() for target in local_targets)


def test_readme_block_replacement_preserves_surrounding_text(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "before\n<!-- BEGIN GENERATED FINAL COMPARISON -->\nold\n"
        "<!-- END GENERATED FINAL COMPARISON -->\nafter\n"
    )
    replacement = (
        "<!-- BEGIN GENERATED FINAL COMPARISON -->\nnew\n<!-- END GENERATED FINAL COMPARISON -->"
    )
    replace_generated_comparison(readme, replacement)
    assert readme.read_text() == "before\n" + replacement + "\nafter\n"


def test_readme_requires_exactly_one_generated_block(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("no markers\n")
    with pytest.raises(ValueError, match="exactly one"):
        replace_generated_comparison(readme, "replacement")
