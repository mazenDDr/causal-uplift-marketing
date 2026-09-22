from __future__ import annotations

import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from causal_uplift.visualization.portfolio import (
    circle_distance_for_jaccard,
    design_svg,
    estimators_svg,
    hero_svg,
    load_portfolio_results,
)

REPOSITORY = Path(__file__).resolve().parents[1]
ASSETS = REPOSITORY / "docs" / "assets"
BUILDERS = (("hero.svg", hero_svg), ("design.svg", design_svg), ("estimators.svg", estimators_svg))
RESULT_SOURCES = {
    "ablation": "experiments/results/confounding_ablation_summary.json",
    "business": "experiments/results/business_policy_summary.json",
    "foundation": "experiments/results/foundation_summary.json",
    "uplift": "experiments/results/uplift_metrics_summary.json",
}


def _committed(relative_path: str) -> str | None:
    """The file as HEAD has it, or None outside a checkout (gpu-box keeps its own results)."""
    completed = subprocess.run(
        ["git", "-C", str(REPOSITORY), "show", f"HEAD:{relative_path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout if completed.returncode == 0 else None


@pytest.fixture(scope="module")
def results() -> dict:
    return load_portfolio_results(REPOSITORY)


@pytest.fixture(scope="module")
def rendered(results: dict) -> dict:
    return {
        "hero.svg": hero_svg(results),
        "design.svg": design_svg(results),
        "estimators.svg": estimators_svg(results),
    }


def test_committed_assets_match_the_committed_results() -> None:
    """The published SVGs must be the ones the committed result files produce.

    The working tree is not consulted: experiments/results is excluded from the gpu-box sync,
    so that machine carries its own rebuilt summaries.
    """
    texts = {name: _committed(path) for name, path in RESULT_SOURCES.items()}
    if any(text is None for text in texts.values()):
        pytest.skip("no git checkout here, so the committed result files are unavailable")
    results = {name: json.loads(text) for name, text in texts.items()}
    for name, builder in BUILDERS:
        committed = _committed(f"docs/assets/{name}")
        assert committed is not None, name
        assert committed == builder(results), f"{name} is stale; run make portfolio-assets"


def test_every_committed_asset_parses_as_svg() -> None:
    for name, _ in BUILDERS:
        root = ET.parse(ASSETS / name).getroot()
        assert root.tag.endswith("svg"), name


def test_every_asset_is_valid_svg(rendered: dict) -> None:
    for name, markup in rendered.items():
        root = ET.fromstring(markup)
        assert root.tag.endswith("svg"), name
        assert root.get("role") == "img", name
        assert root.get("aria-label"), name


def test_hero_reports_the_measured_decision_failure(rendered: dict) -> None:
    hero = rendered["hero.svg"]
    assert "Two ways to choose 852 customers." in hero
    assert "They agree on 76." in hero
    assert ">$162<" in hero
    assert "95% CI $28 to $312 · excludes zero" in hero
    assert "-$63 [-$188, $30]" in hero


def test_hero_states_the_multiplicity_caveat(rendered: dict) -> None:
    assert "not adjusted for that choice" in rendered["hero.svg"]


def test_design_reports_the_measured_split(rendered: dict) -> None:
    design = rendered["design.svg"]
    assert "42,613 customers" in design
    assert "25,567 rows in" in design
    assert "12,781 rows kept" in design
    assert "17,046 rows" in design
    assert "SMD 0.58" in design


def test_estimators_reports_measured_errors_and_intervals(rendered: dict) -> None:
    estimators = rendered["estimators.svg"]
    for value in (">1.98<", ">1.37<", ">1.29<", ">1.23<"):
        assert value in estimators


def test_venn_geometry_encodes_the_measured_overlap(results: dict) -> None:
    jaccard = results["business"]["decision_failure_comparison"][
        "response_vs_uplift_tree_decisions"
    ]["jaccard"]
    radius = 74.0
    distance = circle_distance_for_jaccard(jaccard, radius)
    assert 0 < distance < 2 * radius
    # a larger overlap must push the circles closer together
    assert circle_distance_for_jaccard(jaccard * 4, radius) < distance


def test_assets_rest_on_the_finished_picture(rendered: dict) -> None:
    for name, markup in rendered.items():
        assert "infinite" not in markup, f"{name} must not loop back to its hidden state"
        assert "prefers-reduced-motion" in markup, name
        assert markup.count("11s both") >= 1, name


def test_readme_embeds_every_portfolio_asset() -> None:
    readme = (REPOSITORY / "README.md").read_text()
    sources = re.findall(r'<img src="([^"]+)"', readme)
    embedded = {source for source in sources if source.startswith("docs/assets/")}
    assert embedded == {
        "docs/assets/hero.svg",
        "docs/assets/design.svg",
        "docs/assets/estimators.svg",
    }
    for source in sources:
        assert (REPOSITORY / source).exists(), source
