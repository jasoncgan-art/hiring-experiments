"""Synthetic hiring-experiment data generator.

Methodology demo only — NOT legal or HR advice. All data is synthetic and
deterministic (fixed seeds). No real people, no real employer data, no real
hiring decisions.

Data-generating process (randomized experiment)
----------------------------------------------
For each applicant i:

    X_i ~ N(0, 1)                    # pre-experiment covariate: prior assessment score
    W_i ~ Bernoulli(0.5)             # 0 = old unstructured interview, 1 = new structured rubric
    Y_i = tau * W_i + 1.0 * X_i + e_i,  e_i ~ N(0, 1)

where the TRUE treatment effect is

    TRUE_TAU = +0.35

i.e. the new structured interview rubric raises 90-day new-hire performance by
0.35 standard deviations of the residual noise (sigma = 1). Randomization makes
W independent of (X, potential outcomes), so the difference-in-means estimator
is unbiased for tau.

Observational (non-randomized) variant
--------------------------------------
Same outcome model, but rubric adoption was NOT randomized: hiring managers
piloted the new rubric first with the strongest-assessed candidates, so

    P(W_i = 1 | X_i) = sigmoid(1.0 * X_i + 0.2).

Hence E[X | W=1] > E[X | W=0] and the naive difference-in-means overstates tau
(documented confounding via the observed covariate X). This dataset exists to
demonstrate propensity-score (IPW) adjustment and to show the limits of
observational causal claims.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

#: Ground-truth causal effect of the structured rubric on 90-day performance
#: (standardized units; residual noise sigma = 1, so tau = +0.35 SD).
TRUE_TAU = 0.35

#: Effect of the assessment covariate on performance (signal-to-noise anchor).
BETA_X = 1.0

#: Residual noise standard deviation in the outcome model.
SIGMA = 1.0

EXPERIMENT_SEED = 20260922
OBSERVATIONAL_SEED = 424242


def generate_experiment(n: int = 2000, seed: int = EXPERIMENT_SEED) -> pd.DataFrame:
    """Generate the randomized experiment dataset (balanced 50/50 A/B)."""
    rng = np.random.default_rng(seed)
    assessment = rng.normal(0.0, 1.0, n)
    group = rng.binomial(1, 0.5, n)
    noise = rng.normal(0.0, SIGMA, n)
    perf_90d = TRUE_TAU * group + BETA_X * assessment + noise
    return pd.DataFrame(
        {
            "applicant_id": [f"EXP-{i:05d}" for i in range(n)],
            "assessment_score": assessment,
            "group": group,
            "group_label": np.where(group == 1, "treatment", "control"),
            "perf_90d": perf_90d,
        }
    )


def generate_observational(n: int = 2000, seed: int = OBSERVATIONAL_SEED) -> pd.DataFrame:
    """Generate the non-randomized dataset with documented confounding.

    Confounding mechanism: propensity of receiving the new rubric rises with
    the assessment score (sigmoid(1.0*X + 0.2)), so treated applicants would
    have performed better even without the treatment. Naive comparison of
    group means is therefore biased upward for tau.
    """
    rng = np.random.default_rng(seed)
    assessment = rng.normal(0.0, 1.0, n)
    propensity = expit(1.0 * assessment + 0.2)
    group = rng.binomial(1, propensity, n)
    noise = rng.normal(0.0, SIGMA, n)
    perf_90d = TRUE_TAU * group + BETA_X * assessment + noise
    return pd.DataFrame(
        {
            "applicant_id": [f"OBS-{i:05d}" for i in range(n)],
            "assessment_score": assessment,
            "propensity_true": propensity,
            "group": group,
            "group_label": np.where(group == 1, "treatment", "control"),
            "perf_90d": perf_90d,
        }
    )


if __name__ == "__main__":
    exp = generate_experiment()
    obs = generate_observational()
    print(f"experiment:     n={len(exp)}, treated={(exp.group == 1).mean():.3f}, "
          f"mean perf control={exp.loc[exp.group == 0, 'perf_90d'].mean():.3f}, "
          f"treatment={exp.loc[exp.group == 1, 'perf_90d'].mean():.3f}")
    print(f"observational:  n={len(obs)}, treated={(obs.group == 1).mean():.3f}, "
          f"mean X control={obs.loc[obs.group == 0, 'assessment_score'].mean():.3f}, "
          f"treatment={obs.loc[obs.group == 1, 'assessment_score'].mean():.3f}")
