# Causal design

## Estimand

For the first campaign, `T=1` means Men's E-Mail and `T=0` means No E-Mail. The primary estimand is
the average treatment effect on conversion:

```text
ATE = E[Y(1) - Y(0)]
```

Visit and spend are secondary outcomes. The randomized evaluation split estimates the benchmark
effect by difference in means. Observational estimators train only on selected rows from the other
split.

## Assumed graph

```text
X ──▶ T ──▶ Y
└────────▶ Y
```

`X` contains pre-treatment customer characteristics. Historical selection creates `X -> T`, and
purchase propensity creates `X -> Y`, opening the backdoor path `T <- X -> Y`.

## Assumptions

- **Conditional exchangeability:** `Y(1), Y(0) ⟂ T | X`. All common causes used by the simulated
  assignment mechanism are observed. This cannot be guaranteed in ordinary historical data.
- **Positivity:** each modeled customer type has a nonzero probability of either treatment. We clip
  desired propensities and separately stress-test overlap; clipping does not create evidence where
  the realized sample has none.
- **Consistency:** each observed outcome is the potential outcome under the treatment received.
- **No interference:** one customer's email does not materially change another customer's outcome.
- **Pre-treatment adjustment only:** post-email visit, conversion, and spend never enter features.

## Leakage boundary

The single split happens while treatment remains randomized. Selection is applied only to the
training pool. Source row IDs make train/evaluation disjointness testable, and the confounding
generator selects complete rows without changing treatment or any outcome.

Model and hyperparameter selection occurs inside the observational training side. The RCT evaluation
set is reserved for final ATE error, uplift-decile checks, Qini/AUUC, and policy value.
