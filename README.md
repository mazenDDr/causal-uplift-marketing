<p align="center">
  <img src="docs/assets/hero.svg" width="100%" alt="The response model and the uplift tree each pick 852 customers from the randomized holdout and share only 76. Switching to the uplift tree is worth $162 per 1,000 eligible customers, 95% CI $28 to $312.">
</p>

<h1 align="center">Causal inference and uplift modeling for marketing</h1>

<p align="center">
  <b>Prediction estimates what will happen. Causal inference estimates what will happen <i>because we intervene</i>.</b><br>
  A randomized experiment held back untouched, training data made confounded on purpose,
  and eight methods measured against both.
</p>

<p align="center">
  <a href="#results-at-a-glance"><b>Results</b></a>
  &nbsp;·&nbsp;
  <a href="#experimental-design"><b>The design</b></a>
  &nbsp;·&nbsp;
  <a href="#failure-analysis-every-method-has-a-boundary"><b>Failure taxonomy</b></a>
  &nbsp;·&nbsp;
  <a href="#reproduce-from-raw-data"><b>Reproduce it</b></a>
</p>

---

This project asks which customers should receive a marketing email. It uses the randomized
Hillstrom email experiment as an untouched benchmark, deliberately turns only the training split
into observational data, and measures whether causal estimators recover better targeting decisions
than ordinary predictive ML.

**Status:** the complete 14-stage study is reproducible from one command. It includes matching,
LinearDML, CausalForestDML, uplift trees/forests, randomized ranking and policy evaluation,
confounding and overlap stress tests, robustness checks, a frozen Women's Email replication, and an
interactive decision dashboard.

## Results at a glance

![Average-effect recovery and randomized policy uncertainty](experiments/figures/headline_summary.svg)

<!-- BEGIN GENERATED FINAL COMPARISON -->
| Method | What it estimates | ATE error / 1,000 | Qini / 1,000 (95% CI) | Profit / 1,000 eligible (95% CI) |
|---|---|---:|---:|---:|
| Naive association | Raw association | 1.98 | — | — |
| Response model | Purchase probability | — | 0.106 [-0.748, 0.947] | $145 [-$76, $384] |
| Treatment-as-feature | Pseudo-uplift | — | 0.184 [-0.721, 1.079] | $114 [-$126, $337] |
| PSM segments | Matched ATT | ATT, not ATE | — | $276 [$70, $515] |
| LinearDML | Constant ATE | 1.37 | — | $184 [-$3, $374] |
| CausalForestDML | CATE | 1.23 | -0.200 [-1.039, 0.663] | -$4 [-$237, $244] |
| Uplift tree | CATE ranking | — | -0.190 [-1.051, 0.723] | $167 [-$35, $369] |
| Uplift random forest | CATE ranking | 1.29 | 0.034 [-0.874, 0.873] | $211 [$1, $435] |

*Setting: medium-confounding training data, 20% targeting, $0.05/email, and the untouched randomized holdout. PSM remains ATT rather than being mislabeled as ATE. Every Qini interval and every paired profit difference versus random and response includes zero, so the table does not declare a targeting winner.*
<!-- END GENERATED FINAL COMPARISON -->

The result is deliberately not a leaderboard. Causal adjustment sharply improves average-effect
recovery when selection bias is strong, but none of the individualized rankings achieves a Qini
interval excluding zero. A useful average treatment effect and a useful targeting order are
different claims, and this project tests them separately.

## Where predictive targeting made the wrong decision

![Predictive purchase ranking and uplift targeting make different decisions](experiments/figures/prediction_vs_uplift_policy.svg)

At a 5% campaign budget and $0.05 per email, the response model targets customers with an observed
randomized effect of -4.71 conversions per 1,000 emails and -$63.35 profit per 1,000 eligible
customers. The uplift tree improves on it by 23.82 conversions per 1,000 emails (paired 95% CI
0.99 to 48.34) and $161.53 profit per 1,000 eligible customers (paired 95% CI $27.99 to $311.91).
Only 76 customers appear in both 852-customer target lists.

This is the concrete failure the project was designed to find: customers most likely to purchase
are not necessarily those whose behavior changes because of the email. The 5% diagnostic was
selected from the configured budget grid after inspecting the policy curves, so its nominal
intervals are not multiplicity-adjusted and should be confirmed in a new experiment.

## Experimental design

<p align="center"><img src="docs/assets/design.svg" width="100%" alt="The 42,613-customer randomized campaign splits once into a 25,567-row training pool, deliberately confounded down to 12,781 rows with worst imbalance SMD 0.58, and a 17,046-row randomized holdout never used for selection."></p>

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
association error for each confounding strength. The headline figure and final comparison table are
rebuilt from structured results with `make readme-assets` rather than maintained by hand.

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

## Confounding-strength ablation

The complete estimator set is retrained on randomized, weak-, medium-, and strong-confounding
training samples. Nuisance configurations and the 20% policy budget are inherited from earlier
training-only choices; all policy scores are frozen before randomized outcomes are loaded.

![ATE error as training-data confounding increases](experiments/figures/confounding_ate_ablation.svg)

| Training data | Naive error | LinearDML error | CausalForestDML error | Uplift RF error |
|---|---:|---:|---:|---:|
| Randomized | 0.00039 | 0.00039 | 0.00051 | 0.00055 |
| Weak | 0.00150 | 0.00035 | 0.00016 | 0.00101 |
| Medium | 0.00198 | 0.00137 | 0.00123 | 0.00129 |
| Strong | 0.00371 | 0.00047 | 0.00038 | 0.00300 |

The naive association degrades steadily as measured selection grows. Under strong confounding,
LinearDML and CausalForestDML reduce absolute conversion-effect error by roughly 87% and 90%
relative to the naive estimate. The uplift random forest has no explicit propensity correction and
its mean-effect error also grows sharply. PSM is plotted separately as an ATT reference gap—not
renamed as ATE error—because it targets matched treated customers rather than the overall RCT
population. The randomized sample has 25,567 rows while each selected observational sample has
about 12,700; randomized-to-observational differences therefore include the retention-size change,
whereas weak-to-strong comparisons hold sample size approximately constant.

![Randomized policy value as training-data confounding increases](experiments/figures/confounding_policy_ablation.svg)

Average-effect recovery does not imply a validated customer ranking. At 20% targeting and $0.05
per email, policy point estimates move non-monotonically across training samples and every paired
profit difference versus random has a 95% interval containing zero. This is a useful failure result:
orthogonalization stabilized average effects here, but did not manufacture precise individualized
decisions from a low-rate outcome. The structured estimates, intervals, method warnings, and
runtime are generated in
[`experiments/results/confounding_ablation_summary.json`](experiments/results/confounding_ablation_summary.json).

## Positivity and overlap stress test

The overlap experiment keeps the selected sample near 12,700 rows while increasing treatment-logit
strength and relaxing propensity clips from `[0.10, 0.90]` to `[0.001, 0.999]`. Each level is
sampled five times. The DML nuisance model and matching rules are fixed before the run, and RCT
outcomes remain closed until all 20 observational fits finish.

![Overlap, weight, effective-sample, and matching diagnostics](experiments/figures/overlap_stress_diagnostics.svg)

| Overlap | Known / fitted IPW effective rows | Mean matched pairs | Match balance pass rate | Mean DML CI width | Mean DML ATE error |
|---|---:|---:|---:|---:|---:|
| Healthy | 80.9% / 62.0% | 6,256 | 100% | 0.00662 | 0.00180 |
| Limited | 47.6% / 10.2% | 5,473 | 100% | 0.00764 | 0.00157 |
| Poor | 11.0% / 0.21% | 3,636 | 100% | 0.00965 | 0.00306 |
| Severe | 1.02% / 0.013% | 1,987 | 40% | 0.01336 | 0.00363 |

At severe stress, about 69% of rows have known selection propensities outside `[0.05, 0.95]`.
Fitted propensities sometimes round numerically to zero or one; the diagnostic applies and records
an explicit `1e-6` cap, producing observed weights as large as one million instead of silently
dividing by zero. These weights diagnose support—they are not used to report an IPW effect.

![Repeated DML estimates and uncertainty as overlap deteriorates](experiments/figures/overlap_estimate_stability.svg)

DML's mean analytic interval width roughly doubles from healthy to severe overlap, its
between-resample standard deviation rises from 0.00153 to 0.00280, and only three of five severe
intervals cover the randomized point estimate. Matching preserves `|SMD| < 0.10` in only two of
five severe resamples after discarding an average of 8,569 rows. The result is not that one method
fails first: without comparable treated and control customers, weighting loses effective sample,
matching loses rows and balance, and DML becomes wider and less stable. Full per-seed diagnostics
and intervals are generated in
[`experiments/results/overlap_stress_summary.json`](experiments/results/overlap_stress_summary.json).

## Robustness suite

The robustness suite changes one design choice at a time on the medium-confounding problem: three
propensity models, twelve matching configurations, three DML nuisance specifications, five
observational-selection seeds, five placebo shuffles, ten irrelevant features, and four training
fractions across five fits. The grid is configured before execution, and the RCT benchmark remains
closed until every observational specification is frozen.

![DML nuisance, seed, placebo, sample-size, and noise checks](experiments/figures/robustness_dml.svg)

| Check | Measured result |
|---|---:|
| Nuisance-model ATE error range | 0.00133–0.00164 |
| Selection-seed estimate SD | 0.00161 |
| Placebo mean effect | -0.00037 |
| Placebo intervals covering zero | 5 / 5 |
| Estimate change after 10 noise features | 0.00017 |
| Mean DML CI width, 25% → 100% data | 0.01399 → 0.00672 |

The three nuisance specifications give nearly identical intervals. Irrelevant features barely move
the estimate, and the placebo behaves as expected. More data steadily tightens uncertainty, but
point-estimate error is not monotonic: the 75% subsamples have lower mean error than the 100% fits.
Precision improves with sample size; realized accuracy need not improve in a perfectly smooth line.

![Propensity-model and matching-specification sensitivity](experiments/figures/robustness_matching.svg)

Every matching specification passes the `|SMD| < 0.10` balance rule, yet retained treated customers
range from 2,902 to 6,261 and ATT ranges from 0.00411 to 0.00874. Changing the propensity model
alone moves ATT from 0.00708 to 0.00976. These are not interchangeable estimates of one fixed
population: caliper, replacement, ratio, and propensity model change which treated customers remain
represented. Balance is necessary, but it does not make design choices irrelevant. Complete
per-configuration intervals and runtime are generated in
[`experiments/results/robustness_summary.json`](experiments/results/robustness_summary.json).

## Failure analysis: every method has a boundary

![Measured estimator failure taxonomy](experiments/figures/failure_taxonomy.svg)

| Method | Measured failure | Diagnostic evidence | Required response |
|---|---|---|---|
| Naive association | Strong-confounding ATE is 52.8% above the RCT estimate | Error rises to 0.00371 as selection strengthens | Adjust for pre-treatment confounders and validate against an experiment |
| PSM | Comparable customers disappear | 1,987 mean pairs, 8,569 rows discarded, balance passes in 2/5 severe runs | Restrict the ATT to supported customers; do not extrapolate |
| LinearDML | Orthogonalization cannot create missing treatment variation | CI width is 2.02× healthy overlap; 3/5 severe intervals cover the RCT point estimate | Trim or redefine the population and gather overlapping data |
| Uplift forest | A detailed score fails to become a validated ranking | Predicted top–bottom spread is +12.68 conversions/1,000, observed RCT spread is -2.86 (95% CI -17.23 to +10.54) | Regularize and require held-out uplift/Qini validation before deployment |

The PSM row is an ATT failure in the retained matched treated population, not an ATE-error claim.
The uplift-forest case is labeled as failed generalization and overfit-like behavior: training Qini
was intentionally not used, so the project does not manufacture a training-versus-test overfit
gap. Each symptom, cause, diagnostic, fix, estimand, and source artifact is generated in
[`experiments/results/failure_analysis_summary.json`](experiments/results/failure_analysis_summary.json).

## Women's Email replication

The complete medium-confounding pipeline was frozen and rerun for `Womens E-Mail` versus
`No E-Mail`. The split seed, confounding strength and coefficients, nuisance model selected on the
Men's training data, estimator hyperparameters, bootstrap, budgets, and costs were unchanged. The
only semantic substitution was historical merchandise affinity: `womens` replaces `mens` in the
simulated targeting rule. Women's randomized outcomes were not used to select any model.

![Frozen Men's and Women's campaign comparison](experiments/figures/womens_replication.svg)

| Result | Men's Email | Women's Email |
|---|---:|---:|
| RCT conversion effect / 1,000 | 7.04 [3.99, 10.09] | 1.61 [-0.85, 4.07] |
| Naive ATE error / 1,000 | 1.98 | 4.15 |
| LinearDML ATE error / 1,000 | 1.37 | 1.11 |
| CausalForestDML ATE error / 1,000 | 1.23 | 0.96 |
| Uplift-forest ATE error / 1,000 | 1.29 | 3.28 |
| Best observed Qini / 1,000 | Pseudo-uplift: 0.18 | Response: 0.57 |

The average-effect result replicates: naive association is much farther from the randomized
benchmark than LinearDML or CausalForestDML. The targeting conclusion does **not** replicate. Every
Women's Qini interval contains zero, and every top-minus-bottom decile interval contains zero. At a
20% budget the response model has the highest profit point estimate, $158 per 1,000 eligible
customers, while the causal forest is -$63. Their paired causal-forest-minus-response interval is
-$447 to -$3, but this is one of several unadjusted model/budget comparisons and is evidence to
confirm—not a deployment winner declaration. The full design audit, effect intervals, matching
balance, ranking metrics, policy comparisons, and runtime are generated in
[`experiments/results/womens_replication_summary.json`](experiments/results/womens_replication_summary.json).

## Heterogeneous treatment effects

An honest `CausalForestDML` is trained on the medium-confounding sample using the nuisance
configuration selected in the previous training-only experiment. Its hyperparameters and the
history/recency subgroup rules are frozen before opening randomized outcomes. The forest scores the
RCT holdout, which is divided into exact equal-frequency CATE deciles.

![Causal-forest predictions checked against randomized deciles and customer segments](experiments/figures/causal_forest_validation.svg)

This evaluation does **not** validate the forest's individual ranking. Observed randomized uplift
does not rise with predicted CATE, and the top-minus-bottom decile interval includes zero. The
pointwise CATE intervals are also wide relative to the variation in predictions. Predefined
history/recency segments show some differences, but their intervals overlap and the ordering is not
the forest's predicted ordering.

That negative result is kept as a core project artifact: a sophisticated causal model can recover a
reasonable average effect without learning a reliable individual targeting order. Exact decile and
subgroup intervals, top-minus-bottom uncertainty, CATE dispersion, runtimes, and aggregated feature
importance are generated in
[`experiments/results/causal_forest_summary.json`](experiments/results/causal_forest_summary.json).
Feature importance describes where the forest split; it is not evidence of a causal mechanism.

## Uplift tree and uplift random forest

An honest shallow uplift tree and a 200-tree uplift random forest are trained on the same
medium-confounding observational sample. Their split objectives seek differences between treated
and control outcomes, rather than ordinary outcome accuracy. They do **not** explicitly adjust for
the observational treatment propensity, so randomized validation remains essential.

![Interpretable uplift tree trained on observational marketing data](experiments/figures/uplift_tree.svg)

The tree exposes seven customer leaves and their training-side treated rate, control rate, uplift,
and sample size. Its largest estimated uplift is not treated as a discovered mechanism: these are
descriptive segments learned from selected observational data.

![Uplift tree and forest predictions checked against randomized deciles](experiments/figures/uplift_model_validation.svg)

Neither model learned a validated ranking on the untouched RCT holdout. The tree's top-minus-bottom
uplift is +3.40 conversions per 1,000 emails (95% paired bootstrap interval -9.97 to +17.62); the
forest's is -2.86 (-17.23 to +10.54). The forest-minus-tree top-decile difference is also uncertain
at -3.79 per 1,000 (-19.03 to +10.47). The forest produces smoother, more granular scores, but that
did not translate into better randomized ranking evidence.

The tree has only seven distinct leaf scores, so exact deciles split tied customers arbitrarily.
Its leaf artifact and randomized uncertainty should be read alongside—not replaced by—the decile
chart. Parameters, every leaf, score distributions, model agreement, runtimes, intervals, and the
explicit propensity warning are generated in
[`experiments/results/uplift_models_summary.json`](experiments/results/uplift_models_summary.json).

## Uplift ranking metrics

All five targeting scores are evaluated on the same untouched randomized customers at campaign
sizes from 5% through 100%. Cumulative gain is the targeted fraction multiplied by its randomized
treatment-control conversion difference. Qini subtracts the random-targeting line; AUUC retains the
campaign's average effect. Tied scores receive equal fractional weight at a targeting boundary, so
the seven-leaf tree cannot gain from arbitrary row order.

![Cumulative uplift and Qini curves on the randomized holdout](experiments/figures/uplift_qini_curves.svg)

![Bootstrap intervals for Qini and AUUC](experiments/figures/uplift_metric_intervals.svg)

| Ranking | Qini × 1,000 (95% CI) | AUUC × 1,000 (95% CI) |
|---|---:|---:|
| Response model | 0.106 [-0.748, 0.947] | 3.626 [1.726, 5.526] |
| Treatment-as-feature | 0.184 [-0.721, 1.079] | 3.704 [1.817, 5.679] |
| CausalForestDML | -0.200 [-1.039, 0.663] | 3.320 [1.412, 5.216] |
| Uplift tree | -0.190 [-1.051, 0.723] | 3.330 [1.682, 5.057] |
| Uplift random forest | 0.034 [-0.874, 0.873] | 3.554 [1.683, 5.500] |

Every Qini interval includes zero. Paired Qini differences between each uplift/causal ranking and
the response model also include zero. Positive AUUC alone is not ranking evidence here: the overall
campaign ATE is positive, so random targeting has positive AUUC too. The bootstrap is paired and
stratified by randomized treatment arm; its uncertainty is conditional on the already-fitted model
scores. Definitions, curves, intervals, paired differences, and runtime are generated in
[`experiments/results/uplift_metrics_summary.json`](experiments/results/uplift_metrics_summary.json).

## Business targeting policies

The frozen rankings are converted into exact top-k policies at 5%, 10%, 20%, 30%, 40%, 50%, 75%,
and 100% campaign sizes. A seeded, outcome-independent tie breaker handles the shallow uplift tree,
PSM segments, and constant-effect LinearDML. Incremental revenue is the randomized difference in
Hillstrom `spend`; profit subtracts the configurable email cost.

![Profit across campaign sizes and email costs](experiments/figures/policy_profit_curves.svg)

![Policy profit intervals at the primary business setting](experiments/figures/policy_20pct_profit.svg)

At the fixed primary setting—20% targeted and $0.05 per email—the point estimates are:

| Policy | Conversions / 1,000 emails | Profit / 1,000 eligible (95% CI) |
|---|---:|---:|
| Random | 9.62 | $184 [-$3, $374] |
| Response model | 8.72 | $145 [-$76, $384] |
| Treatment-as-feature | 5.56 | $114 [-$126, $337] |
| PSM segments | 9.71 | $276 [$70, $515] |
| LinearDML (constant) | 9.62 | $184 [-$3, $374] |
| CausalForestDML | 4.84 | -$4 [-$237, $244] |
| Uplift tree | 8.96 | $167 [-$35, $369] |
| Uplift random forest | 10.15 | $211 [$1, $435] |

Despite some individually positive intervals, every paired 20% profit difference versus both the
response and random policies includes zero; no primary-setting winner is declared. LinearDML is a
constant-effect estimator here, so it correctly reduces to the same seeded ordering as random
rather than pretending to personalize treatment.

Cost-aware rules send when `$50 × predicted conversion uplift > email cost` and are evaluated at
$0.01, $0.05, $0.10, $0.25, and $0.50. The best point-estimate top-k budget is 100% for every model
and tested cost, reflecting positive average incremental spend and relatively cheap email—not proof
that ranking is unnecessary. Those maxima use RCT outcomes and are descriptive, not deployable
budget choices. Complete visit, spend, revenue, profit, ROI, cost-aware, paired, and policy-curve
results are generated in
[`experiments/results/business_policy_summary.json`](experiments/results/business_policy_summary.json).

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

## Methods compared

<p align="center"><img src="docs/assets/estimators.svg" width="100%" alt="Left: absolute error in the average treatment effect per 1,000, worst for naive association and lower for every causal estimator. Right: Qini intervals per 1,000 for five customer rankings, all crossing zero."></p>

The fixed comparison includes naive differences, a response model, treatment-as-feature pseudo-
uplift, propensity-score matching, LinearDML, CausalForestDML, an uplift tree, and an uplift random
forest. A method is useful only if it improves held-out RCT ATE error, uplift ranking, or business
policy value with uncertainty—not because it is more sophisticated.

## Reproduce from raw data

Use Python 3.11 on Linux or WSL and install the project with its causal, visualization, and test
extras. The audited run used Python 3.11.14, EconML 0.16.0, CausalML 0.16.0, NumPy 2.3.5,
pandas 2.3.3, and scikit-learn 1.6.1; these key versions and the machine platform are recorded in the
[experiment audit](experiments/results/experiment_runner_summary.json).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[causal,viz,dev]'
python scripts/download_hillstrom.py
make experiments
make readme-assets
make check
```

The downloader accepts only the 64,000-row Hillstrom file with SHA-256
`0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece`. A mirror that
returns different bytes is rejected before replacing an existing file. Raw data and run logs are
kept outside Git. Figures and JSON summaries already committed to the repository let readers inspect
the results without rerunning the heavy models.

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

## Run the complete experiment DAG

After downloading Hillstrom once, the complete measured pipeline runs with one command:

```bash
./gpu run make experiments
```

`scripts/run_experiments.py` resolves 14 stages in dependency order, writes each stage's stdout to
`outputs/experiment_runner/logs/`, and atomically checkpoints
`outputs/experiment_runner/manifest.json` after every stage. Repeating the command resumes only a
stage whose previous status succeeded, whose code/config/upstream fingerprint is unchanged, and
whose declared outputs still exist with the exact recorded byte hashes. A stale input, missing or
altered output, or failed run invalidates that stage; changed regenerated results also invalidate
its downstream dependents.

The verified full run completed all 14 stages in 429.14 seconds on `gpu-box`; an immediate repeat
resumed all 14 stages in 0.02 seconds without recomputing them.

Useful local equivalents inside the configured Python environment are:

```bash
# Inspect the resolved order without running anything.
make experiment-plan

# Run one terminal stage and all of its dependencies.
PYTHONPATH=src python scripts/run_experiments.py --steps business_policies

# Deliberately rebuild the complete grid.
PYTHONPATH=src python scripts/run_experiments.py --force
```

The latest machine, package versions, commands, fingerprints, required outputs, per-stage runtimes,
and resume actions are recorded in
[`experiments/results/experiment_runner_summary.json`](experiments/results/experiment_runner_summary.json).

## Explore the decision dashboard

The Streamlit dashboard reads the committed experiment summaries directly—no result is copied into
the interface by hand. It exposes confounding strength, targeting model, campaign budget, and email
cost while keeping the randomized holdout visible as the benchmark.

```bash
pip install -e '.[viz]'
make dashboard
```

The full campaign-budget grid is available for the medium-confounding experiment. The other
confounding levels deliberately remain fixed at the pre-registered 20% budget rather than
extrapolating results that were never measured.

## Repository map

- `src/causal_uplift/data`: schema validation, randomized split, and confounding injection
- `src/causal_uplift/causal`: causal estimators (added task by task)
- `src/causal_uplift/evaluation`: ATE, ranking, uncertainty, and policy metrics
- `scripts`: reproducible command-line experiment entry points
- `configs`: versioned experiment settings
- `tests`: causal invariants and metric checks
- `app`: the Streamlit decision demo

## Limitations known in advance

- Randomization identifies group effects, not individual counterfactual outcomes.
- Selection on recorded covariates cannot test robustness to truly unmeasured confounding.
- Weak overlap can make effects unidentified for some customer types; no estimator fixes missing
  comparisons.
- Small changes in causal-forest scores can reorder near-tied customers across reruns, so its
  targeting value should be interpreted with the randomized intervals, not as a byte-stable point
  estimate.
- Hillstrom is one retailer and one historical campaign. External validity requires new experiments.

## License

The project code and original documentation are available under the [MIT License](LICENSE).
The Hillstrom source dataset is not included in this repository.
