from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from causal_uplift.dashboard import (
    MODEL_SPECS,
    RESULT_FILES,
    balance_rows,
    effect_summary,
    heterogeneity_rows,
    load_dashboard_results,
    policy_curve,
    policy_summary,
    propensity_quantiles,
    qini_curve,
)

REPOSITORY = Path(__file__).resolve().parents[1]
STRENGTH_LABELS = {
    "None — randomized": "randomized",
    "Weak": "weak",
    "Medium": "medium",
    "Strong": "strong",
}
COMPACT_MODEL_NAMES = {
    "Naive pseudo-uplift": "Naive",
    "Response model": "Response",
    "PSM segments": "PSM",
    "LinearDML": "LinearDML",
    "CausalForestDML": "Causal forest",
    "Uplift random forest": "Uplift forest",
}


@st.cache_data
def load_results() -> dict:
    return load_dashboard_results(REPOSITORY)


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def money(value: float) -> str:
    return f"${value:,.0f}"


def format_budget(value: float | str) -> str:
    return value if isinstance(value, str) and value.endswith("%") else f"{float(value):.0%}"


def format_cost(value: float | str) -> str:
    return value if isinstance(value, str) and value.startswith("$") else f"${float(value):.2f}"


def parse_budget(value: float | str) -> float:
    return float(value.removesuffix("%")) / 100 if isinstance(value, str) else float(value)


def parse_cost(value: float | str) -> float:
    return float(value.removeprefix("$")) if isinstance(value, str) else float(value)


st.set_page_config(
    page_title="Intervention Lab · Causal Uplift Marketing",
    page_icon="↗",
    layout="wide",
    initial_sidebar_state="auto",
)
st.markdown(
    """
    <style>
    :root { --ink:#172126; --muted:#5f6d70; --mint:#dff6ed; }
    .stApp { background: #f7f5ef; color: var(--ink); }
    [data-testid="stSidebar"] { background: #172126; }
    [data-testid="stSidebar"] * { color: #f7f5ef; }
    [data-testid="stMetric"] {
      background: rgba(255,255,255,.72); border: 1px solid #d9ddd8; border-radius: 14px;
      padding: 14px 16px;
    }
    .eyebrow { color:#a84e3c; font-size:.74rem; font-weight:800; letter-spacing:.12em; }
    .hero { font-size:clamp(2.3rem,6vw,5.2rem); line-height:.94; letter-spacing:-.055em;
      max-width:980px; margin:.25rem 0 1rem; }
    .lede { color:var(--muted); font-size:1.05rem; max-width:760px; }
    .evidence { background:var(--mint); border-left:5px solid #168263; border-radius:12px;
      padding:14px 18px; margin:1rem 0 1.5rem; }
    h2 { letter-spacing:-.025em; padding-top:1.5rem; }
    [data-testid="stDataFrame"] { border:1px solid #d9ddd8; border-radius:12px; overflow:hidden; }
    @media (max-width: 700px) {
      .hero { font-size:2.55rem; }
      .block-container { padding-top:1.5rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

results = load_results()
business = results["business"]

with st.sidebar:
    st.markdown("### Decision controls")
    strength_label = st.selectbox("Confounding", list(STRENGTH_LABELS), index=2)
    strength = STRENGTH_LABELS[strength_label]
    model_name = st.selectbox("Targeting model", list(MODEL_SPECS), index=4)
    if strength == "medium":
        budget = st.select_slider(
            "Campaign budget",
            options=business["settings"]["budgets"],
            value=0.20,
            format_func=format_budget,
        )
        budget = parse_budget(budget)
    else:
        st.text_input(
            "Campaign budget",
            value="20%",
            disabled=True,
            help=(
                "The complete budget grid was frozen for medium confounding. "
                "Other stress levels were evaluated at 20%."
            ),
        )
        budget = 0.20
    email_cost = st.select_slider(
        "Email cost",
        options=business["settings"]["email_costs"],
        value=0.05,
        format_func=format_cost,
    )
    email_cost = parse_cost(email_cost)
    st.caption("Every displayed outcome comes from the untouched randomized holdout.")

st.markdown('<div class="eyebrow">RCT-BENCHMARKED DECISION SYSTEM</div>', unsafe_allow_html=True)
st.markdown('<div class="hero">Prediction is not persuasion.</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="lede">Explore how treatment-selection bias changes the estimate, then test '
    "whether the resulting targeting policy creates incremental conversions and profit.</div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="evidence"><b>Ground truth:</b> Men\'s Email increased conversion by '
    f"{percent(results['foundation']['rct_effects']['conversion']['estimate'])} on the randomized "
    "holdout. A model is useful here only if its decisions survive that experiment.</div>",
    unsafe_allow_html=True,
)

top = st.columns(4)
observational_rows = results["ablation"]["samples"][strength]["training_rows"]
top[0].metric("Observational rows", f"{observational_rows:,}")
top[1].metric("RCT evaluation rows", f"{results['foundation']['rows']['rct_evaluation']:,}")
top[2].metric("RCT conversion lift", percent(results["ablation"]["rct_effect"]["estimate"]))
top[3].metric("Selected policy", COMPACT_MODEL_NAMES[model_name])

st.header("1 · Assignment bias and balance")
balance, propensity = st.columns([1.15, 0.85])
with balance:
    st.subheader("Absolute standardized mean differences")
    balance_frame = pd.DataFrame(balance_rows(results, strength)).set_index("covariate")
    st.bar_chart(balance_frame, horizontal=True, height=420)
    st.caption(
        f"Maximum |SMD| moves from {balance_frame['before'].max():.3f} to "
        f"{balance_frame['after'].max():.3f} after the preferred match. "
        "The design target is below 0.10."
    )
with propensity:
    st.subheader("Propensity-score quantiles")
    propensity_frame = pd.DataFrame(propensity_quantiles(results, strength)).set_index("quantile")
    st.line_chart(propensity_frame, height=320)
    propensity_sample = results["propensity"]["samples"][strength]
    st.metric(
        "Outside [0.05, 0.95]",
        percent(propensity_sample["outside_overlap_threshold_fraction"]),
    )
    st.caption("Quantiles summarize fitted treatment probabilities for treated and control rows.")

st.header("2 · Causal estimate against experimental truth")
effect = effect_summary(results, strength, model_name)
if effect is None:
    st.info(
        "The response model has no treatment-effect estimate. It predicts who will buy, which is "
        "precisely why it remains a separate targeting baseline."
    )
else:
    estimate_cols = st.columns(3)
    estimate_cols[0].metric("Model estimate", percent(effect["estimate"]))
    estimate_cols[1].metric("RCT benchmark", percent(effect["rct_estimate"]))
    if effect.get("absolute_error_vs_rct_ate") is not None:
        estimate_cols[2].metric("Absolute ATE error", percent(effect["absolute_error_vs_rct_ate"]))
    else:
        estimate_cols[2].metric(
            "ATT reference gap", percent(effect["reference_gap_vs_overall_rct_ate"])
        )
    effect_frame = pd.DataFrame(
        {"effect": [effect["estimate"], effect["rct_estimate"]]},
        index=[model_name, "RCT benchmark"],
    )
    st.bar_chart(effect_frame, height=260)
    st.caption(effect["model_estimand"])

st.header("3 · Where effects appear heterogeneous")
heterogeneity = heterogeneity_rows(results, strength, model_name)
if heterogeneity:
    heterogeneity_frame = pd.DataFrame(heterogeneity).set_index("segment")
    chart_columns = ["predicted"]
    if heterogeneity_frame["observed"].notna().any():
        chart_columns.append("observed")
    st.bar_chart(heterogeneity_frame[chart_columns], height=360)
    st.dataframe(
        heterogeneity_frame,
        width="stretch",
        column_config={
            "predicted": st.column_config.NumberColumn("Estimated effect", format="%.4f"),
            "observed": st.column_config.NumberColumn("RCT effect", format="%.4f"),
            "ci_lower": st.column_config.NumberColumn("RCT CI lower", format="%.4f"),
            "ci_upper": st.column_config.NumberColumn("RCT CI upper", format="%.4f"),
        },
    )
    if model_name == "PSM segments":
        st.caption(
            "PSM rows are segment ATT estimates; no segment-level randomized effect was stored."
        )
    else:
        st.caption("Observed effects and intervals are calculated only after ranking was frozen.")
        if strength != "medium":
            st.caption(
                "This heterogeneity model was trained on the frozen medium-confounding sample; "
                "the selector does not fabricate a refit at other strengths."
            )
else:
    st.info(
        "This model does not produce a defensible heterogeneous-effect view in the experiment. "
        "Choose PSM segments, CausalForestDML, or Uplift random forest to inspect heterogeneity."
    )

st.header("4 · Uplift ranking on the randomized holdout")
ranking = qini_curve(results, model_name)
if ranking is None:
    st.info("This estimator has no individual ranking, so Qini and AUUC are not applicable.")
else:
    qini_rows, qini = ranking
    ranking_cols = st.columns([1.4, 0.6])
    with ranking_cols[0]:
        ranking_frame = pd.DataFrame(qini_rows).set_index("fraction")
        st.line_chart(ranking_frame, height=330)
    with ranking_cols[1]:
        st.metric("Qini / 1,000", f"{qini['estimate_per_1000']:.3f}")
        st.write(
            f"95% bootstrap interval: **{qini['ci_lower_per_1000']:.3f} to "
            f"{qini['ci_upper_per_1000']:.3f}**"
        )
        st.caption(
            "The interval includes zero when the ranking is not distinguishable from random."
        )
    st.caption(
        "Qini uses scores trained on the frozen medium-confounding sample and the randomized "
        "holdout; it is not recomputed when the confounding selector changes."
    )

st.header("5 · Business decision")
policy, policy_scope = policy_summary(results, strength, model_name, budget, email_cost)
profit = policy["incremental_profit_per_1000_eligible"]
business_cols = st.columns(5)
business_cols[0].metric("Customers targeted", f"{policy['targeted_customers']:,}")
business_cols[1].metric(
    "Conversions / 1,000 emails", f"{policy['conversion']['effect_per_1000_emails']:.2f}"
)
business_cols[2].metric(
    "Incremental revenue / 1,000", money(policy["incremental_revenue_per_1000_eligible"])
)
business_cols[3].metric("Campaign cost / 1,000", money(policy["campaign_cost_per_1000_eligible"]))
business_cols[4].metric("Incremental profit / 1,000", money(profit["estimate"]))
st.caption(
    f"95% profit interval: {money(profit['ci_lower'])} to {money(profit['ci_upper'])}. "
    f"Evidence scope: {policy_scope}."
)

if strength == "medium":
    st.subheader("Profit across campaign sizes")
    policy_curve_frame = pd.DataFrame(policy_curve(results, model_name, email_cost)).set_index(
        "budget"
    )
    st.line_chart(policy_curve_frame, height=300)
else:
    st.info(
        "The confounding stress test froze the budget at 20%. Switch to Medium to explore the "
        "complete 5–100% policy curve without extrapolating unmeasured results."
    )

st.markdown("---")
st.caption(
    "All estimates are generated from committed JSON artifacts. Individual counterfactual effects "
    "are not observed; ranking and policy value are evaluated with treatment-control differences "
    "inside the untouched randomized holdout."
)
with st.expander("Evidence provenance"):
    st.json(
        {
            "training_sample": strength,
            "model": model_name,
            "budget": budget,
            "email_cost": email_cost,
            "evaluation": results["business"]["evaluation_data"],
            "seed": results["business"]["seed"],
            "source_files": sorted(f"experiments/results/{name}" for name in RESULT_FILES.values()),
        }
    )
