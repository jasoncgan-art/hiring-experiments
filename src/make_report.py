"""Generate report/report.md (and report/figures/power_curve.png) with REAL
computed numbers. Never hand-copy metrics into the report — run this script.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import analysis, causal, power
from src.data_gen import (
    TRUE_TAU, generate_experiment, generate_observational,
)

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "report"
FIG_DIR = REPORT_DIR / "figures"

DISCLAIMER = (
    "> **Disclaimer:** this is a *methodology demonstration* on fully synthetic, "
    "simulated data. It is not legal advice, not HR advice, and not evidence about "
    "any real hiring process. Nothing here should be used to make real hiring "
    "decisions."
)


def plot_power_curve(path: Path) -> None:
    ns = np.arange(100, 5001, 100)
    pw = [power.power_for(TRUE_TAU, n // 2) for n in ns]
    n_req = 2 * power.required_n_per_arm(TRUE_TAU)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ns, pw, lw=2, label=f"power at delta={TRUE_TAU}")
    ax.axhline(0.80, color="gray", ls="--", lw=1, label="80% power")
    ax.axvline(n_req, color="C1", ls=":", lw=1.5, label=f"required N={n_req}")
    ax.axvline(2000, color="C2", ls=":", lw=1.5, label="actual experiment N=2000")
    ax.set_xlabel("total applicants N")
    ax.set_ylabel("statistical power (alpha=0.05, two-sided)")
    ax.set_title(f"Power curve for detecting the true effect (delta={TRUE_TAU})")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fmt(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}"


def build_report() -> str:
    exp = generate_experiment()
    obs = generate_observational()

    raw = analysis.diff_in_means(exp)
    cu = analysis.cuped(exp)
    ols = analysis.ols_adjusted(exp)

    n_req_arm = power.required_n_per_arm(TRUE_TAU)
    mde_2000 = power.mde(1000)
    pwr_2000 = power.power_for(TRUE_TAU, 1000)

    caus = causal.summarize(obs)
    b0, b1 = causal.fit_propensity(obs["assessment_score"].to_numpy(),
                                   obs["group"].to_numpy())
    mean_x_t = obs.loc[obs.group == 1, "assessment_score"].mean()
    mean_x_c = obs.loc[obs.group == 0, "assessment_score"].mean()

    deltas = (0.20, 0.30, 0.35, 0.50)
    ptable = power.power_table(deltas=deltas)
    prow = "\n".join(
        f"| {r['total_n']} | {r['n_per_arm']} | {fmt(r['mde'], 3)} | "
        + " | ".join(fmt(r[f"power_d{d}"], 3) for d in deltas) + " |"
        for r in ptable
    )

    # Honest interpretation snippets, driven by the computed numbers.
    raw_covers = "covers" if raw["ci_lo"] <= TRUE_TAU <= raw["ci_hi"] else "MISSES"
    cu_covers = "covers" if cu["ci_lo"] <= TRUE_TAU <= cu["ci_hi"] else "MISSES"
    naive_bias = caus["naive"]["ate"] - TRUE_TAU
    ipw_err = abs(caus["ipw"]["ate"] - TRUE_TAU)
    dr_err = abs(caus["doubly_robust"]["ate"] - TRUE_TAU)

    return f"""# Hiring-experiment analysis report

{DISCLAIMER}

All numbers below are computed by `src/make_report.py` from synthetic data
generated with fixed seeds (`src/data_gen.py`). The **known true treatment
effect** baked into both datasets is **tau = +0.35** (standardized units:
+0.35 SD of the residual outcome noise). The experiment tests a hypothetical
"new structured interview rubric vs. old unstructured interviews" on a
hypothetical 90-day new-hire performance score.

## 1. Randomized experiment (N = {len(exp)})

- Randomization: 50/50, `{raw['n_control']}` control / `{raw['n_treatment']}` treatment.
- Observed means: control {fmt(raw['mean_control'], 3)}, treatment {fmt(raw['mean_treatment'], 3)}.
- Pre-experiment covariate `assessment_score` correlates with the outcome
  (outcome-model R^2 = {fmt(ols['r_squared'], 3)}), which is what makes CUPED work.

| Estimator | ATE | SE | 95% CI | p-value |
|---|---|---|---|---|
| Difference-in-means (Welch t) | {fmt(raw['ate'])} | {fmt(raw['se'])} | [{fmt(raw['ci_lo'])}, {fmt(raw['ci_hi'])}] | {raw['p_value']:.2e} |
| CUPED (covariate-adjusted) | {fmt(cu['ate'])} | {fmt(cu['se'])} | [{fmt(cu['ci_lo'])}, {fmt(cu['ci_hi'])}] | {cu['p_value']:.2e} |
| OLS / ANCOVA | {fmt(ols['ate'])} | {fmt(ols['se'])} | [{fmt(ols['ci_lo'])}, {fmt(ols['ci_hi'])}] | — |

**Variance reduction.** CUPED estimated theta = {fmt(cu['theta'])} and reduced
the sampling variance of the ATE estimator by **{cu['var_reduction_pct']:.1f}%**
relative to raw difference-in-means (CI width shrank accordingly).

**Coverage.** The raw 95% CI {raw_covers} the true tau = 0.35; the CUPED 95% CI
{cu_covers} it. (A 95% CI is *expected* to miss the truth ~5% of the time —
this run got lucky/unlucky accordingly; what matters is the procedure, not the
single draw.)

## 2. Power analysis

The unadjusted outcome has within-arm variance 2 (covariate variance 1 plus
noise variance 1). The default calculations below therefore use sigma=sqrt(2).
An ideal covariate-adjusted design uses residual sigma=1 and requires
{power.required_n_per_arm(TRUE_TAU, sigma=1)} per arm under this known DGP.
This is a planning approximation, not a universal CUPED sample-size guarantee.

Minimum detectable effect (MDE) and power at several effect sizes
(alpha = 0.05 two-sided, sigma = sqrt(2), unadjusted outcome SD):

| total N | per arm | MDE | power d=0.20 | power d=0.30 | power d=0.35 | power d=0.50 |
|---|---|---|---|---|---|---|
{prow}

- To detect the true effect delta = 0.35 at 80% power: **{n_req_arm} per arm
  ({2 * n_req_arm} total)**.
- With the actual N = {len(exp)} ({raw['n_control']} control / {raw['n_treatment']}
  treatment realized): MDE = {fmt(mde_2000, 3)},
  power at delta = 0.35 is {fmt(pwr_2000, 4)} — the experiment is heavily
  over-powered for the true effect, which is why all CIs above are tight.

![Power curve](figures/power_curve.png)

## 3. Observational (non-randomized) data — causal estimates (N = {len(obs)})

Confounding: the new rubric was piloted first with stronger-assessed
candidates — mean `assessment_score` is {fmt(mean_x_t, 3)} among treated vs
{fmt(mean_x_c, 3)} among controls. Estimated propensity model:
logit(e) = {fmt(b0, 3)} + {fmt(b1, 3)} * assessment_score.

| Estimator | ATE | 95% bootstrap CI |
|---|---|---|
| Naive difference-in-means | {fmt(caus['naive']['ate'])} | [{fmt(caus['naive']['ci_lo'])}, {fmt(caus['naive']['ci_hi'])}] |
| IPW (Hajek, estimated propensity) | {fmt(caus['ipw']['ate'])} | [{fmt(caus['ipw']['ci_lo'])}, {fmt(caus['ipw']['ci_hi'])}] |
| Outcome regression (OLS adj.) | {fmt(caus['outcome_regression']['ate'])} | [{fmt(caus['outcome_regression']['ci_lo'])}, {fmt(caus['outcome_regression']['ci_hi'])}] |
| Doubly robust | {fmt(caus['doubly_robust']['ate'])} | [{fmt(caus['doubly_robust']['ci_lo'])}, {fmt(caus['doubly_robust']['ci_hi'])}] |

True tau = 0.35. The naive estimate is biased upward by **+{fmt(naive_bias)}** —
it mistakes the treated group's pre-existing assessment advantage for a rubric
effect. IPW removes most of that bias (|error| = {fmt(ipw_err)}) and the doubly
robust estimator has |error| = {fmt(dr_err)}. In this particular draw,
the smaller absolute error identifies the closer estimate; no estimator is
guaranteed to be closest in every finite sample.

**The unconfoundedness assumption.** These corrections are only valid if,
conditional on `assessment_score`, rubric assignment was as good as random.
That can fail: if, say, enthusiastic managers both volunteered for the pilot
*and* coached their new hires harder, an unmeasured confounder remains and
even the adjusted estimates stay biased. With real (non-synthetic) data there
is no way to verify this from the data alone — which is why randomized
experiments (Section 1) are the gold standard.

## 4. Reproducibility

```
pip install -r requirements.txt
python -m src.make_report   # regenerates this file + the figure
pytest tests/               # determinism, variance-reduction, coverage checks
```

Seeds: experiment = 20260922, observational = 424242, bootstrap = 777.
"""


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_power_curve(FIG_DIR / "power_curve.png")
    report = build_report()
    out = REPORT_DIR / "report.md"
    out.write_text(report)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

