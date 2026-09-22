"""A/B analysis of the randomized hiring-rubric experiment.

Estimators (all estimate the average treatment effect, tau):
1. diff_in_means — Welch two-sample t-test, 95% CI, p-value.
2. cuped — CUPED variance reduction (Deng et al., 2013): subtract the part of
   the outcome predictable from the pre-experiment covariate X
   (assessment_score) before differencing means. theta_hat is the pooled OLS
   slope of Y on X (estimated on pre-treatment data only — here X is a
   pre-experiment covariate, so this is legitimate).
3. ols_adjusted — ANCOVA: OLS of Y on (1, W, X); numerically close to CUPED
   with pooled theta; reports the regression-adjusted ATE with a 95% CI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def diff_in_means(df: pd.DataFrame, outcome: str = "perf_90d") -> dict:
    """Welch two-sample comparison of treatment vs control means."""
    yt = df.loc[df["group"] == 1, outcome].to_numpy()
    yc = df.loc[df["group"] == 0, outcome].to_numpy()
    t_stat, p_value = stats.ttest_ind(yt, yc, equal_var=False)
    ate = float(np.mean(yt) - np.mean(yc))
    se = float(np.sqrt(yt.var(ddof=1) / len(yt) + yc.var(ddof=1) / len(yc)))
    # Welch-Satterthwaite degrees of freedom
    v_t, v_c = yt.var(ddof=1) / len(yt), yc.var(ddof=1) / len(yc)
    dfree = (v_t + v_c) ** 2 / (v_t ** 2 / (len(yt) - 1) + v_c ** 2 / (len(yc) - 1))
    crit = stats.t.ppf(0.975, dfree)
    return {
        "ate": ate,
        "se": se,
        "ci_lo": ate - crit * se,
        "ci_hi": ate + crit * se,
        "p_value": float(p_value),
        "mean_treatment": float(np.mean(yt)),
        "mean_control": float(np.mean(yc)),
        "n_treatment": len(yt),
        "n_control": len(yc),
        "estimator_var": se ** 2,  # estimated sampling variance of the ATE estimator
    }


def cuped(df: pd.DataFrame, outcome: str = "perf_90d",
          covariate: str = "assessment_score") -> dict:
    """CUPED estimator using a pre-experiment covariate.

    Y_adj = Y - theta_hat * (X - X_bar); ATE = mean(Y_adj | W=1) - mean(Y_adj | W=0).
    Variance reduction % = 100 * (1 - Var(CUPED) / Var(raw diff-in-means)),
    where each variance is the usual Welch plug-in estimate.
    """
    y = df[outcome].to_numpy()
    x = df[covariate].to_numpy()
    # Pooled OLS slope of Y on X (pre-experiment covariate, so no post-treatment bias).
    theta = float(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1))
    y_adj = y - theta * (x - x.mean())

    raw = diff_in_means(df, outcome)
    df_adj = df.assign(_y_adj=y_adj)
    adj = diff_in_means(df_adj, "_y_adj")
    var_reduction = 100.0 * (1.0 - adj["estimator_var"] / raw["estimator_var"])
    return {
        "ate": adj["ate"],
        "se": adj["se"],
        "ci_lo": adj["ci_lo"],
        "ci_hi": adj["ci_hi"],
        "p_value": adj["p_value"],
        "theta": theta,
        "var_reduction_pct": float(var_reduction),
        "raw": raw,
    }


def ols_adjusted(df: pd.DataFrame, outcome: str = "perf_90d",
                 covariate: str = "assessment_score") -> dict:
    """ANCOVA: OLS of Y on intercept + treatment + covariate.

    Returns the regression-adjusted ATE (coefficient on treatment) with
    homoskedastic standard error and 95% CI.
    """
    y = df[outcome].to_numpy()
    w = df["group"].to_numpy()
    x = df[covariate].to_numpy()
    X = np.column_stack([np.ones_like(y), w, x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    sigma2 = float(resid @ resid / dof)
    cov_beta = sigma2 * np.linalg.inv(X.T @ X)
    se_w = float(np.sqrt(cov_beta[1, 1]))
    crit = stats.t.ppf(0.975, dof)
    return {
        "ate": float(beta[1]),
        "se": se_w,
        "ci_lo": float(beta[1] - crit * se_w),
        "ci_hi": float(beta[1] + crit * se_w),
        "r_squared": float(1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))),
    }


if __name__ == "__main__":
    from src.data_gen import TRUE_TAU, generate_experiment

    df = generate_experiment()
    raw = diff_in_means(df)
    cu = cuped(df)
    ols = ols_adjusted(df)
    print(f"true tau = {TRUE_TAU}")
    print(f"diff-in-means:  ATE={raw['ate']:.4f}  95% CI=[{raw['ci_lo']:.4f}, {raw['ci_hi']:.4f}]  p={raw['p_value']:.2e}")
    print(f"CUPED:          ATE={cu['ate']:.4f}  95% CI=[{cu['ci_lo']:.4f}, {cu['ci_hi']:.4f}]  "
          f"theta={cu['theta']:.4f}  variance reduction={cu['var_reduction_pct']:.1f}%")
    print(f"OLS-adjusted:   ATE={ols['ate']:.4f}  95% CI=[{ols['ci_lo']:.4f}, {ols['ci_hi']:.4f}]  R^2={ols['r_squared']:.3f}")
