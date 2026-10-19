# hiring-experiments

A demo of the applied-statistics toolkit behind A/B testing and program
evaluation: design and analyze a hypothetical experiment — "new structured
interview rubric vs. old unstructured interviews", measured on a hypothetical
90-day new-hire performance score — with power analysis, CUPED variance
reduction, and a simple causal (IPW / doubly robust) estimate on
non-randomized data.

> **All data is synthetic** (generated at runtime by `src/data_gen.py`, fixed
> seeds). This repo demonstrates experiment **methodology** — it is not legal
> advice, not HR advice, and not evidence about any real hiring process.
> Nothing here should be used to make real hiring decisions.

## Why this exists

I'm a Data Science master's student (UMich) with 5+ years in HR and recruiting.
Hiring teams constantly ask causal questions — "did the new interview process
*cause* better hires?" — and usually answer them with before/after comparisons
that confuse selection effects with treatment effects. This repo shows the
right machinery on clean synthetic data where the ground truth is known, so
each method can be checked against it: randomization, power analysis, CUPED,
and propensity-score adjustment.

## Methodology

1. **`src/data_gen.py`** — generates two datasets with a **known true effect
   tau = +0.35 residual-noise SD** baked in (documented in the module docstring):
   - *Randomized experiment* (N=2000, 50/50): `perf_90d = 0.35*group +
     1.0*assessment_score + noise`. The pre-experiment covariate
     `assessment_score` is strongly predictive of the outcome (R² ≈ 0.52).
   - *Observational dataset* (N=2000): same outcome model, but the new rubric
     was piloted first with stronger-assessed candidates
     (`P(W=1|X) = sigmoid(X + 0.2)`) — documented confounding, so the naive
     comparison is biased upward.
2. **`src/power.py`** — MDE / sample-size calculator for the two-sample design
   (default sigma = sqrt(2) for unadjusted outcomes), plus a printable power table.
3. **`src/analysis.py`** — randomized-experiment analysis: Welch
   difference-in-means with 95% CI, CUPED using the pre-experiment covariate
   (pooled-theta, reports variance-reduction %), and ANCOVA (OLS-adjusted)
   estimate.
4. **`src/causal.py`** — observational causal estimates: naive diff-in-means,
   Hajek IPW with IRLS-estimated propensity scores, outcome regression, and a
   doubly robust estimator (bootstrap 95% CIs). States the unconfoundedness
   assumption and why it may fail.
5. **`src/make_report.py`** — regenerates `report/report.md` and
   `report/figures/power_curve.png` from scratch, so no number is ever
   hand-copied.

## Results (from a real run on this machine)

True effect tau = +0.35 everywhere. Randomized experiment (980 control /
1020 treatment):

| Estimator | ATE | 95% CI | p-value |
|---|---|---|---|
| Difference-in-means (Welch t) | 0.305 | [0.183, 0.428] | 1.2e-06 |
| CUPED | 0.389 | [0.304, 0.474] | 8.6e-19 |
| OLS / ANCOVA | 0.390 | [0.304, 0.475] | — |

- **CUPED cut the estimator's sampling variance by 51.8%** (theta = 0.997),
  visibly tightening the CI.
- Both the raw and CUPED 95% CIs cover the true 0.35. Note the honest detail:
  the raw diff-in-means *underestimated* this draw (0.305) while CUPED landed
  slightly above (0.389) — single draws wobble; the procedure is what's valid.
- Power: the unadjusted design uses outcome SD sqrt(2), while an ideal
  covariate-adjusted design uses residual SD 1. See the regenerated report for
  distinct sample-size calculations; 129/arm applies only to the latter.

Observational data (mean assessment: 0.381 treated vs −0.457 control):

| Estimator | ATE | 95% bootstrap CI |
|---|---|---|
| Naive difference-in-means | 1.142 | [1.023, 1.268] |
| IPW (Hajek) | 0.329 | [0.209, 0.447] |
| Outcome regression | 0.315 | [0.213, 0.415] |
| Doubly robust | 0.306 | [0.199, 0.412] |

The naive estimate is biased upward by +0.79 — it mistakes the treated group's
pre-existing assessment advantage for a rubric effect. IPW removes most of the
bias (|error| = 0.022); doubly robust has |error| = 0.044; IPW is closer in this draw. Full tables
and the power curve: [`report/report.md`](report/report.md).

## Assumptions & limitations

- **Synthetic data, known DGP.** Every result is conditional on the simulation
  being right — in real hiring data, effects, noise, and confounding are all
  unknown, and effect sizes are rarely this clean.
- **Randomized section** assumes proper randomization with no interference
  (one applicant's rubric doesn't affect another's performance) and no
  attrition/selection on the 90-day outcome.
- **Observational section** assumes *unconfoundedness*: conditional on
  `assessment_score`, rubric assignment is as good as random, plus overlap
  (propensities bounded away from 0/1). This is unverifiable from data alone —
  an unmeasured confounder (e.g. enthusiastic managers who both piloted the
  rubric and coached harder) would leave even the adjusted estimates biased.
  That's the point of the demo: adjustment helps, but it can't manufacture the
  credibility of a randomized experiment.
- **Not legal or HR advice.** A real evaluation of a hiring procedure would
  need employment-law review (selection-rate disparity, validation standards)
  and domain expertise this repo does not provide.

## Quickstart

```bash
bash run.sh          # venv + pytest + regenerate report/report.md
```

Manual steps:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -q
python -m src.make_report
```

Dependencies are pinned to the versions used for this revision; the setup no
longer assumes particular packages happen to be preinstalled by the OS.

## Layout

```
src/
  data_gen.py     synthetic experiment + observational datasets (seeds fixed)
  power.py        MDE / sample-size / power-table calculations
  analysis.py     diff-in-means, CUPED, OLS-adjusted ATE
  causal.py       naive / IPW / outcome-regression / doubly-robust + bootstrap CIs
  make_report.py  regenerates report/report.md + figures from real computations
tests/            pytest: determinism, CUPED variance reduction, CI coverage,
                  observational bias ordering, report generation
report/           report.md + figures/ (committed; regenerable via make_report.py)
run.sh            end-to-end: venv, install, test, report
```

