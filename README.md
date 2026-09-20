# Causal inference and uplift modeling for marketing

> Prediction estimates what will happen. Causal inference estimates what will happen because we intervene.

This project asks which customers should receive a marketing email. It uses the randomized
Hillstrom email experiment as an untouched benchmark, deliberately turns only the training split
into observational data, and measures whether causal estimators recover better targeting decisions
than ordinary predictive ML.

**Status:** the RCT split, controlled-confounding foundation, naive targeting baselines, propensity
diagnostics, propensity-score matching, and LinearDML are verified. Heterogeneous treatment-effect
modeling is next.

## Experimental design

```text
Original randomized Men's Email vs No Email experiment
                         |
                  split once (seeded)
                  /                 \
        60% training pool       40% RCT evaluation
                  |                  |
     selection on pre-treatment X   | untouched
                  |                  |
     observational training data    |
                  \                 /
             train on left, score policies on right
```

The holdout is not a model-selection set. Hyperparameters and nuisance models are chosen within the
observational training side; the randomized holdout is opened only for final estimator and policy
comparisons.

The first frozen run is stored in
[`experiments/results/foundation_summary.json`](experiments/results/foundation_summary.json). It
verifies the 60/40 split, dataset checksum, randomized benchmark, covariate imbalance, and raw
association error for each confounding strength. Later tasks will generate the headline figures and
full estimator comparison from structured results rather than hand-copying values.

The naive response-model and treatment-as-feature policy run is stored in
[`experiments/results/naive_baselines_summary.json`](experiments/results/naive_baselines_summary.json).
It includes development-set predictive metrics, held-out RCT policy effects, paired bootstrap policy
differences, score summaries, runtimes, and decision disagreement.

## Why association can be wrong

![Causal DAG showing customer characteristics confounding treatment and outcome](docs/causal_dag.svg)

Historical marketers may target customers with stronger purchase history. Those same customers are
already more likely to convert. The association between email and conversion therefore combines the
email's effect with pre-existing customer differences.

![Estimated treatment propensity by confounding strength](experiments/figures/propensity_overlap.svg)

![Pre-matching covariate balance by confounding strength](experiments/figures/pre_matching_love_plot.svg)

The generated propensity diagnostics are stored in
[`experiments/results/propensity_diagnostics_summary.json`](experiments/results/propensity_diagnostics_summary.json).
The figures show the fitted treatment-selection model and balance before matching; they are
diagnostics, not evidence that adjustment has succeeded.

## Propensity-score matching

The matching experiment compares 1:1 nearest neighbors with and without replacement on the logit
of the estimated propensity score. It trims scores outside `[0.05, 0.95]`, enforces a
`0.2 × SD(logit propensity)` caliper, and checks every pre-treatment covariate after matching. The
preferred variant is chosen without outcomes or RCT results: first pass maximum `|SMD| < 0.10`,
then retain the most treated customers.

![Covariate balance before and after preferred propensity-score matching](experiments/figures/matching_balance.svg)

| Training sample | Max \|SMD\| before | Max \|SMD\| after | Matched treated | Conversion ATT (95% paired bootstrap CI) |
|---|---:|---:|---:|---:|
| Randomized | 0.018 | 0.018 | 12,781 | 0.0086 [0.0066, 0.0109] |
| Weak | 0.325 | 0.035 | 6,407 | 0.0070 [0.0039, 0.0103] |
| Medium | 0.584 | 0.026 | 6,262 | 0.0065 [0.0029, 0.0102] |
| Strong | 0.837 | 0.062 | 5,466 | 0.0099 [0.0064, 0.0134] |

All preferred matches pass the balance gate. Matching without replacement produces still lower
maximum imbalance, but discards more treated customers—2,378 of 5,466 overlap-eligible treated
customers under strong confounding. The full comparison, secondary outcomes, caliper checks,
retention counts, and ATT intervals are generated in
[`experiments/results/matching_summary.json`](experiments/results/matching_summary.json). PSM
estimates ATT in the selected observational population, so its gap from the overall RCT ATE is a
reference rather than a like-for-like estimator error.

## Double Machine Learning

LinearDML separates two nuisance problems—predicting conversion from customer history and
predicting email assignment from the same pre-treatment history—then estimates the treatment
effect from their residuals. Every observation receives nuisance predictions from models that did
not train on that observation:

```text
five joint-stratified folds
        |
        +-- train nuisance models on folds 2–5 -> residualize fold 1
        +-- train nuisance models on folds 1,3–5 -> residualize fold 2
        +-- repeat until every row is scored out of fold
        |
        +-- estimate the treatment effect from residual-on-residual variation
```

Two nuisance configurations are compared: random-forest outcome plus logistic treatment, and
histogram-gradient-boosting outcome plus treatment. Selection uses only five-fold out-of-fold
Brier loss relative to constant baselines. The randomized outcomes are loaded after all selections
finish and are used only to measure error.

![ATE error for naive association and cross-fitted LinearDML](experiments/figures/dml_ate_error.svg)

As measured selection becomes stronger, naive ATE error grows while both DML configurations stay
closer to the randomized benchmark. The selected nuisance configuration is not always the one with
the smallest eventual RCT error, which is expected because the holdout is not a tuning set. Exact
ATE intervals, nuisance metrics, runtimes, selections, package versions, and errors are generated in
[`experiments/results/linear_dml_summary.json`](experiments/results/linear_dml_summary.json).
The conversion nuisance models are also roughly level with a constant-probability baseline on this
rare outcome; flexibility alone did not create useful outcome prediction.

| Feature | Timing | Allowed? | Reason |
|---|---|---:|---|
| `history` | pre-treatment | Yes | Prior spend |
| `recency` | pre-treatment | Yes | Prior activity |
| `channel` | pre-treatment | Yes | Prior purchase channel |
| `visit` | post-treatment | No | Secondary outcome |
| `conversion` | post-treatment | No | Primary outcome |
| `spend` | post-treatment | No | Secondary outcome |

The observational analyses assume conditional exchangeability, positivity, consistency, and no
interference. These assumptions are documented in [the causal design](docs/causal_design.md); the
stress tests are designed to show where they become implausible or uninformative.

## Measured comparison roadmap

The fixed comparison includes naive differences, a response model, treatment-as-feature pseudo-
uplift, propensity-score matching, LinearDML, CausalForestDML, an uplift tree, and an uplift random
forest. A method is useful only if it improves held-out RCT ATE error, uplift ranking, or business
policy value with uncertainty—not because it is more sophisticated.

## Reproduce the foundation

Data and heavy runs live on `gpu-box`:

```bash
./gpu push
./gpu exec python scripts/download_hillstrom.py
./gpu exec make foundation
./gpu exec make test
./gpu pull
```

The equivalent flow is `rsync` to `gpu-box`, then an SSH command inside the conda environment
`main`. The helper keeps `data/` and `outputs/` remote and excluded from source control.

## Repository map

- `src/causal_uplift/data`: schema validation, randomized split, and confounding injection
- `src/causal_uplift/causal`: causal estimators (added task by task)
- `src/causal_uplift/evaluation`: ATE, ranking, uncertainty, and policy metrics
- `scripts`: reproducible command-line experiment entry points
- `configs`: versioned experiment settings
- `tests`: causal invariants and metric checks
- `app`: the final Streamlit decision demo

## Limitations known in advance

- Randomization identifies group effects, not individual counterfactual outcomes.
- Selection on recorded covariates cannot test robustness to truly unmeasured confounding.
- Weak overlap can make effects unidentified for some customer types; no estimator fixes missing
  comparisons.
- Hillstrom is one retailer and one historical campaign. External validity requires new experiments.
