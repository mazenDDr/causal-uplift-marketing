from __future__ import annotations

import json
import math
from pathlib import Path

BACKGROUND = "#EFEDE6"
CARD = "#FFFFFF"
EDGE = "#DAD7CE"
INK = "#1A2233"
MUTED = "#6E7684"
RULE = "#E6E3DA"
BLUE = "#3A6EA5"
RED = "#C0272D"
GREEN = "#2E7D4F"
GOLD = "#FFE36A"

SERIF = "Georgia,'Times New Roman',serif"
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"
SANS = "-apple-system,'Segoe UI',Helvetica,Arial,sans-serif"

REDUCED = (
    "@media(prefers-reduced-motion:reduce)"
    "{*{animation:none!important;opacity:1!important;transform:none!important}}"
)

ATE_ESTIMATORS = (
    ("naive", "Naive association"),
    ("linear_dml", "LinearDML"),
    ("uplift_random_forest", "Uplift random forest"),
    ("causal_forest_dml", "CausalForestDML"),
)

QINI_RANKINGS = (
    ("response_model", "Response model"),
    ("pseudo_uplift", "Treatment-as-feature"),
    ("causal_forest_dml", "CausalForestDML"),
    ("uplift_tree", "Uplift tree"),
    ("uplift_random_forest", "Uplift random forest"),
)


def load_portfolio_results(repository: Path) -> dict:
    result_dir = repository / "experiments" / "results"
    filenames = {
        "ablation": "confounding_ablation_summary.json",
        "business": "business_policy_summary.json",
        "foundation": "foundation_summary.json",
        "uplift": "uplift_metrics_summary.json",
    }
    return {
        name: json.loads((result_dir / filename).read_text())
        for name, filename in filenames.items()
    }


def _money(value: float) -> str:
    rounded = int(round(value))
    return f"-${abs(rounded):,}" if rounded < 0 else f"${rounded:,}"


def _count(value: int) -> str:
    return f"{value:,}"


def _svg(width: int, height: int, label: str, title: str, style: str, body: str) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}" role="img" aria-label="{label}">',
            f"  <title>{title}</title>",
            "  <style>",
            style,
            f"    {REDUCED}",
            "  </style>",
            f'  <rect width="{width}" height="{height}" rx="14" fill="{BACKGROUND}"/>',
            body,
            "</svg>",
            "",
        ]
    )


def _lens_area(distance: float, radius: float) -> float:
    if distance >= 2 * radius:
        return 0.0
    half = distance / 2
    return 2 * radius**2 * math.acos(half / radius) - half * math.sqrt(
        max(4 * radius**2 - distance**2, 0.0)
    )


def circle_distance_for_jaccard(jaccard: float, radius: float) -> float:
    """Centre distance whose lens-to-union ratio equals the measured overlap."""
    low, high = 0.0, 2 * radius
    for _ in range(80):
        middle = (low + high) / 2
        lens = _lens_area(middle, radius)
        union = 2 * math.pi * radius**2 - lens
        if lens / union > jaccard:
            low = middle
        else:
            high = middle
    return round((low + high) / 2, 2)


def hero_svg(results: dict) -> str:
    failure = results["business"]["decision_failure_comparison"]
    decisions = failure["response_vs_uplift_tree_decisions"]
    response = failure["response_policy"]
    switch = failure["alternatives_vs_response"]["uplift_tree"][
        "profit_difference_vs_response_per_1000_eligible"
    ]
    holdout = _count(results["business"]["evaluation_rows"])
    picked = _count(response["targeted_customers"])
    shared = _count(decisions["both"])
    budget = f"{failure['budget'] * 100:.0f}%"

    zero_x = 534
    profit_scale = 294 / (switch["ci_upper"] * 1.06)
    lower_x = round(zero_x + switch["ci_lower"] * profit_scale, 1)
    upper_x = round(zero_x + switch["ci_upper"] * profit_scale, 1)
    point_x = round(zero_x + switch["estimate"] * profit_scale, 1)

    radius = 74.0
    distance = circle_distance_for_jaccard(decisions["jaccard"], radius)
    centre_y = 214
    left_x = round(266 - distance / 2, 2)
    right_x = round(266 + distance / 2, 2)
    lens_x = round((left_x + right_x) / 2, 2)

    style = "\n".join(
        [
            f"    .lab{{font:600 10px {MONO};fill:{MUTED};letter-spacing:.11em}}",
            f"    .head{{font:400 25px {SERIF};fill:{INK}}}",
            f"    .body{{font:400 13px {SANS};fill:{MUTED}}}",
            f"    .tiny{{font:400 11px {SANS};fill:{MUTED}}}",
            f"    .mono{{font:500 12px {MONO};fill:{INK}}}",
            f"    .big{{font:600 44px {SANS};fill:{GREEN}}}",
            f"    .setname{{font:600 12px {MONO};letter-spacing:.06em}}",
            f"    .card{{fill:{CARD};stroke:{EDGE}}}",
            "    .ring{fill:none;stroke-width:2.2}",
            f"    .fillA{{fill:{RED};opacity:.10}}",
            f"    .fillB{{fill:{BLUE};opacity:.12}}",
            f"    .lens{{fill:{GOLD};opacity:.85}}",
            f"    .leader{{fill:none;stroke:{MUTED};stroke-width:1}}",
            f"    .ci{{fill:{GREEN};transform-box:fill-box;transform-origin:left center}}",
            f"    .zero{{stroke:{MUTED};stroke-width:1;stroke-dasharray:3 3}}",
            "    .setA{animation:pop 11s both}",
            "    .setB{animation:popb 11s both}",
            "    .overlap{animation:late 11s both}",
            "    .money{animation:later 11s both}",
            "    .interval{animation:last 11s both}",
            "    @keyframes pop{0%,4%{opacity:0}12%,100%{opacity:1}}",
            "    @keyframes popb{0%,14%{opacity:0}22%,100%{opacity:1}}",
            "    @keyframes late{0%,28%{opacity:0}36%,100%{opacity:1}}",
            "    @keyframes later{0%,44%{opacity:0}52%,100%{opacity:1}}",
            "    @keyframes last{0%,58%{opacity:0}66%,100%{opacity:1}}",
        ]
    )

    body = "\n".join(
        [
            f'  <text class="lab" x="30" y="34">UNTOUCHED RANDOMIZED HOLDOUT · {holdout} CUSTOMERS '
            f"· {budget} CAMPAIGN BUDGET</text>",
            f'  <text class="head" x="30" y="76">Two ways to choose {picked} customers.</text>',
            f'  <text class="head" x="30" y="108">They agree on {shared}.</text>',
            '  <g class="setA">',
            f'    <circle class="fillA" cx="{left_x}" cy="{centre_y}" r="{radius}"/>',
            f'    <circle class="ring" cx="{left_x}" cy="{centre_y}" r="{radius}" stroke="{RED}"/>',
            f'    <text class="setname" x="{left_x}" y="{centre_y - 8}" fill="{RED}" '
            f'text-anchor="middle">RESPONSE</text>',
            f'    <text class="setname" x="{left_x}" y="{centre_y + 10}" fill="{RED}" '
            f'text-anchor="middle">MODEL</text>',
            f'    <text class="tiny" x="{left_x}" y="{centre_y + 30}" '
            f'text-anchor="middle">{picked} picked</text>',
            "  </g>",
            '  <g class="setB">',
            f'    <circle class="fillB" cx="{right_x}" cy="{centre_y}" r="{radius}"/>',
            f'    <circle class="ring" cx="{right_x}" cy="{centre_y}" r="{radius}" '
            f'stroke="{BLUE}"/>',
            f'    <text class="setname" x="{right_x}" y="{centre_y - 8}" fill="{BLUE}" '
            f'text-anchor="middle">UPLIFT</text>',
            f'    <text class="setname" x="{right_x}" y="{centre_y + 10}" fill="{BLUE}" '
            f'text-anchor="middle">TREE</text>',
            f'    <text class="tiny" x="{right_x}" y="{centre_y + 30}" '
            f'text-anchor="middle">{picked} picked</text>',
            "  </g>",
            '  <g class="overlap">',
            f'    <clipPath id="lensclip"><circle cx="{left_x}" cy="{centre_y}" '
            f'r="{radius}"/></clipPath>',
            f'    <circle class="lens" cx="{right_x}" cy="{centre_y}" r="{radius}" '
            f'clip-path="url(#lensclip)"/>',
            f'    <path class="leader" d="M{lens_x} {centre_y + radius - 4} L{lens_x} 332"/>',
            f'    <text class="mono" x="{lens_x}" y="348" text-anchor="middle">{shared} in '
            f"both</text>",
            "  </g>",
            '  <rect class="card" x="512" y="120" width="338" height="216" rx="9"/>',
            '  <g class="money">',
            '    <text class="lab" x="534" y="152">SWITCHING THE LIST IS WORTH</text>',
            f'    <text class="big" x="534" y="200">{_money(switch["estimate"])}</text>',
            '    <text class="tiny" x="534" y="222">incremental profit per 1,000 eligible '
            "customers</text>",
            "  </g>",
            '  <g class="interval">',
            f'    <line class="zero" x1="{zero_x}" y1="240" x2="{zero_x}" y2="274"/>',
            f'    <text class="tiny" x="{zero_x}" y="288">no effect</text>',
            f'    <rect class="ci" x="{lower_x}" y="254" width="{round(upper_x - lower_x, 1)}" '
            f'height="3.4" rx="1.7"/>',
            f'    <circle cx="{point_x}" cy="255.7" r="4.2" fill="{GREEN}"/>',
            f'    <text class="tiny" x="534" y="308">95% CI {_money(switch["ci_lower"])} to '
            f"{_money(switch['ci_upper'])} · excludes zero</text>",
            f'    <text class="tiny" x="534" y="326">Response model on its own: '
            f"{_money(response['incremental_profit_per_1000_eligible']['estimate'])} "
            f"[{_money(response['incremental_profit_per_1000_eligible']['ci_lower'])}, "
            f"{_money(response['incremental_profit_per_1000_eligible']['ci_upper'])}]</text>",
            "  </g>",
            '  <text class="body" x="30" y="376">The customers most likely to buy are not the '
            "customers the email changes.</text>",
            f'  <text class="tiny" x="30" y="400">{budget} came from the prespecified budget grid '
            f"but was chosen for this diagnostic after "
            "inspecting the policy curves, so the interval is not adjusted for that choice.</text>",
        ]
    )

    label = (
        f"Two overlapping circles: the response model and the uplift tree each pick {picked} "
        f"customers and share only {shared}. Switching to the uplift tree is worth "
        f"{_money(switch['estimate'])} per 1,000 eligible customers, 95% confidence interval "
        f"{_money(switch['ci_lower'])} to {_money(switch['ci_upper'])}."
    )
    return _svg(880, 420, label, "Two target lists that barely overlap", style, body)


def design_svg(results: dict) -> str:
    foundation = results["foundation"]
    rows = foundation["rows"]
    randomized = foundation["observational_samples"]["randomized"]
    medium = foundation["observational_samples"]["medium"]
    campaign = _count(rows["campaign_total"])
    pool = _count(rows["randomized_training_pool"])
    holdout = _count(rows["rct_evaluation"])
    kept = _count(medium["rows"])
    balance_scale = 250 / 0.9

    style = "\n".join(
        [
            f"    .lab{{font:600 10px {MONO};fill:{MUTED};letter-spacing:.11em}}",
            f"    .head{{font:400 22px {SERIF};fill:{INK}}}",
            f"    .body{{font:400 12.5px {SANS};fill:{MUTED}}}",
            f"    .tiny{{font:400 10.5px {SANS};fill:{MUTED}}}",
            f"    .mono{{font:500 13px {MONO};fill:{INK}}}",
            f"    .card{{fill:{CARD};stroke:{EDGE}}}",
            f"    .flow{{fill:none;stroke:{MUTED};stroke-width:1.4;stroke-dasharray:320;"
            "stroke-dashoffset:320;animation:draw 11s both}",
            "    .bar{transform-box:fill-box;transform-origin:left center;animation:grow 11s both}",
            f"    .seal rect{{fill:none;stroke:{GREEN};stroke-width:2.4}}",
            f"    .seal text{{font:500 11px {MONO};fill:{GREEN};letter-spacing:.14em}}",
            "    .seal{transform-box:fill-box;transform-origin:center;"
            "transform:rotate(-7deg);animation:stamp 11s both}",
            "    .step{animation:show 11s both}",
            "    @keyframes draw{0%,16%{stroke-dashoffset:320}34%,100%{stroke-dashoffset:0}}",
            "    @keyframes show{0%,34%{opacity:0}44%,100%{opacity:1}}",
            "    @keyframes grow{0%,46%{transform:scaleX(0)}64%,100%{transform:scaleX(1)}}",
            "    @keyframes stamp{0%,62%{opacity:0;transform:rotate(-7deg) scale(1.6)}"
            "70%,100%{opacity:1;transform:rotate(-7deg) scale(1)}}",
        ]
    )

    body = "\n".join(
        [
            '  <text class="lab" x="30" y="32">THE DESIGN · SPLIT ONCE, SEEDED</text>',
            '  <text class="head" x="30" y="66">One randomized experiment, split before anything '
            "else happens.</text>",
            '  <rect class="card" x="284" y="92" width="312" height="66" rx="9"/>',
            '  <text class="lab" x="304" y="116">HILLSTROM RANDOMIZED EXPERIMENT</text>',
            f'  <text class="mono" x="304" y="140">{campaign} customers · Men’s Email vs No '
            f"Email</text>",
            '  <path class="flow" d="M440 158 L440 184 L196 184 L196 208"/>',
            '  <path class="flow" d="M440 158 L440 184 L684 184 L684 208"/>',
            '  <g class="step">',
            '    <rect class="card" x="30" y="208" width="332" height="164" rx="9"/>',
            '    <text class="lab" x="52" y="234">TRAINING POOL, MADE OBSERVATIONAL</text>',
            f'    <text class="mono" x="52" y="260">{pool} rows in</text>',
            '    <text class="body" x="52" y="282">Confounding applied on purpose, then</text>',
            f'    <text class="mono" x="52" y="304">{kept} rows kept</text>',
            '    <text class="lab" x="52" y="330">WORST COVARIATE IMBALANCE</text>',
            "  </g>",
            f'  <rect class="bar" x="52" y="340" '
            f'width="{round(medium["max_absolute_smd"] * balance_scale, 1)}" height="11" rx="3" '
            f'fill="{RED}"/>',
            f'  <text class="tiny" '
            f'x="{round(52 + medium["max_absolute_smd"] * balance_scale + 10, 1)}" y="350">SMD '
            f"{medium['max_absolute_smd']:.2f}</text>",
            '  <g class="step">',
            '    <rect class="card" x="518" y="208" width="332" height="164" rx="9"/>',
            '    <text class="lab" x="540" y="234">RANDOMIZED HOLDOUT</text>',
            f'    <text class="mono" x="540" y="260">{holdout} rows</text>',
            '    <text class="body" x="540" y="282">Never used to select features, models</text>',
            '    <text class="body" x="540" y="300">or hyperparameters. Read once, at the '
            "end.</text>",
            f'    <text class="tiny" x="540" y="330">Imbalance by construction: SMD '
            f"{randomized['max_absolute_smd']:.2f}</text>",
            "  </g>",
            '  <g class="seal">',
            '    <rect x="700" y="332" width="122" height="28" rx="4"/>',
            '    <text x="716" y="351">UNTOUCHED</text>',
            "  </g>",
            '  <text class="body" x="30" y="398">Every number this project reports about targeting '
            "was measured on the right-hand side.</text>",
        ]
    )

    label = (
        f"The {campaign}-customer randomized campaign splits once into a {pool}-row training "
        f"pool, deliberately confounded down to {kept} rows with worst imbalance "
        f"{medium['max_absolute_smd']:.2f}, and a {holdout}-row randomized holdout that is never "
        "used for selection."
    )
    return _svg(880, 420, label, "Split once: confounded training, untouched holdout", style, body)


def estimators_svg(results: dict) -> str:
    medium = results["ablation"]["samples"]["medium"]["ate_estimators"]
    rankings = results["uplift"]["models"]
    errors = [
        (label, medium[key]["absolute_error_vs_rct_ate"] * 1000) for key, label in ATE_ESTIMATORS
    ]
    qinis = [(label, rankings[key]["qini"]) for key, label in QINI_RANKINGS]

    error_scale = 232 / max(value for _, value in errors)
    bound = max(max(abs(q["ci_lower_per_1000"]), abs(q["ci_upper_per_1000"])) for _, q in qinis)
    qini_scale = 124 / bound
    zero_x = 700

    style = "\n".join(
        [
            f"    .lab{{font:600 10px {MONO};fill:{MUTED};letter-spacing:.11em}}",
            f"    .head{{font:400 22px {SERIF};fill:{INK}}}",
            f"    .body{{font:400 12.5px {SANS};fill:{MUTED}}}",
            f"    .tiny{{font:400 10.5px {SANS};fill:{MUTED}}}",
            f"    .name{{font:400 11.5px {SANS};fill:{INK}}}",
            f"    .mono{{font:500 11px {MONO};fill:{INK}}}",
            f"    .card{{fill:{CARD};stroke:{EDGE}}}",
            "    .bar{transform-box:fill-box;transform-origin:left center;animation:grow 11s both}",
            f"    .zero{{stroke:{INK};stroke-width:1.2}}",
            f"    .ci{{stroke:{BLUE};stroke-width:2.2;transform-box:fill-box;"
            "transform-origin:center;animation:widen 11s both}",
            f"    .dot{{fill:{BLUE}}}",
            "    .late{animation:late 11s both}",
            "    @keyframes grow{0%,8%{transform:scaleX(0)}30%,100%{transform:scaleX(1)}}",
            "    @keyframes widen{0%,34%{transform:scaleX(0)}56%,100%{transform:scaleX(1)}}",
            "    @keyframes late{0%,58%{opacity:0}68%,100%{opacity:1}}",
        ]
    )

    parts = [
        '  <text class="lab" x="30" y="32">'
        "MEDIUM CONFOUNDING · UNTOUCHED RANDOMIZED HOLDOUT</text>",
        '  <text class="head" x="30" y="66">A useful average effect and a useful targeting '
        "order</text>",
        '  <text class="head" x="30" y="94">are different claims.</text>',
        '  <rect class="card" x="30" y="116" width="396" height="226" rx="9"/>',
        '  <text class="lab" x="52" y="142">ERROR IN THE AVERAGE EFFECT · PER 1,000</text>',
        '  <rect class="card" x="454" y="116" width="396" height="226" rx="9"/>',
        '  <text class="lab" x="476" y="142">QINI · PER 1,000 · 95% INTERVAL</text>',
    ]

    for index, (label, value) in enumerate(errors):
        top = 168 + index * 44
        colour = RED if index == 0 else BLUE
        parts.extend(
            [
                f'  <text class="name" x="52" y="{top}">{label}</text>',
                f'  <rect class="bar" x="52" y="{top + 8}" width="{round(value * error_scale, 1)}" '
                f'height="12" rx="3" fill="{colour}"/>',
                f'  <text class="mono" x="{round(52 + value * error_scale + 9, 1)}" '
                f'y="{top + 18}">{value:.2f}</text>',
            ]
        )

    for index, (label, qini) in enumerate(qinis):
        top = 172 + index * 34
        lower = round(zero_x + qini["ci_lower_per_1000"] * qini_scale, 1)
        upper = round(zero_x + qini["ci_upper_per_1000"] * qini_scale, 1)
        point = round(zero_x + qini["estimate_per_1000"] * qini_scale, 1)
        parts.extend(
            [
                f'  <text class="name" x="476" y="{top + 4}">{label}</text>',
                f'  <line class="ci" x1="{lower}" y1="{top}" x2="{upper}" y2="{top}"/>',
                f'  <circle class="dot" cx="{point}" cy="{top}" r="3.2"/>',
            ]
        )

    parts.extend(
        [
            f'  <line class="zero" x1="{zero_x}" y1="156" x2="{zero_x}" y2="322"/>',
            f'  <text class="tiny" x="{zero_x}" y="336" text-anchor="middle">no effect</text>',
            '  <g class="late">',
            '  <text class="body" x="30" y="376">Causal adjustment cut the average-effect error. '
            "Every ranking interval still crosses zero, "
            "so no targeting winner is claimed.</text>",
            "  </g>",
        ]
    )

    label = (
        "Left: absolute error in the average treatment effect per 1,000, worst for naive "
        "association and lower for every causal estimator. Right: Qini intervals per 1,000 for "
        "five customer rankings, all crossing zero."
    )
    title = "Average effects recovered, rankings still uncertain"
    return _svg(880, 400, label, title, style, "\n".join(parts))


def build_portfolio(repository: Path) -> list[Path]:
    results = load_portfolio_results(repository)
    assets = repository / "docs" / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    written = []
    for name, markup in (
        ("hero.svg", hero_svg(results)),
        ("design.svg", design_svg(results)),
        ("estimators.svg", estimators_svg(results)),
    ):
        path = assets / name
        path.write_text(markup)
        written.append(path)
    return written
