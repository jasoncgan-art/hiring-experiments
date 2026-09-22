"""Causal estimates on the OBSERVATIONAL (non-randomized) dataset.

Estimators of the ATE:
1. naive — raw difference-in-means (biased upward here: confounding).
2. ipw — inverse-propensity weighting (Hajek normalized form) with
   propensity scores estimated by logistic regression of W on X (IRLS).
3. outcome_regression — OLS of Y on (1, W, X); regression adjustment.
4. doubly_robust — combines both: consistent if EITHER the propensity model
   OR the outcome model is correctly specified.

Assumption (unconfoundedness / conditional ignorability):
    (Y(0), Y(1)) independent of W  given  X.
In words: after conditioning on the assessment score, rubric assignment is
"as good as random". This is strong and may FAIL in practice — e.g. if
enthusiastic hiring managers both piloted the new rubric AND coached
candidates harder, an unmeasured confounder remains and all four estimators
above stay biased. Overlap is also required: 0 < P(W=1|X) < 1 for all X.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit


def fit_propensity(x: np.ndarray, w: np.ndarray,
                   max_iter: int = 100, tol: float = 1e-10) -> tuple[float, float]:
    """Logistic regression W ~ 1 + X via IRLS. Returns (intercept, slope)."""
    Xd = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(max_iter):
        p = np.clip(expit(Xd @ beta), 1e-8, 1 - 1e-8)
        wgt = p * (1 - p)
        z = Xd @ beta + (w - p) / wgt
        sqrt_w = np.sqrt(wgt)
        beta_new, *_ = np.linalg.lstsq(Xd * sqrt_w[:, None], z * sqrt_w, rcond=None)
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    return float(beta[0]), float(beta[1])


def propensity_scores(df: pd.DataFrame) -> np.ndarray:
    """Estimated P(W=1 | X) for each row, clipped away from 0/1."""
    b0, b1 = fit_propensity(df["assessment_score"].to_numpy(), df["group"].to_numpy())
    e = expit(b0 + b1 * df["assessment_score"].to_numpy())
    return np.clip(e, 0.01, 0.99)


def naive(df: pd.DataFrame) -> float:
    """Raw difference-in-means (no confounding adjustment)."""
    yt = df.loc[df["group"] == 1, "perf_90d"].to_numpy()
    yc = df.loc[df["group"] == 0, "perf_90d"].to_numpy()
    return float(np.mean(yt) - np.mean(yc))


def ipw(df: pd.DataFrame, e: np.ndarray | None = None) -> float:
    """Hajek (normalized) IPW estimator."""
    if e is None:
        e = propensity_scores(df)
    y = df["perf_90d"].to_numpy()
    w = df["group"].to_numpy()
    mu1 = np.sum(w * y / e) / np.sum(w / e)
    mu0 = np.sum((1 - w) * y / (1 - e)) / np.sum((1 - w) / (1 - e))
    return float(mu1 - mu0)


def outcome_regression(df: pd.DataFrame) -> float:
    """Regression adjustment: OLS of Y on (1, W, X), coefficient on W."""
    y = df["perf_90d"].to_numpy()
    w = df["group"].to_numpy()
    x = df["assessment_score"].to_numpy()
    X = np.column_stack([np.ones_like(y), w, x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[1])


def doubly_robust(df: pd.DataFrame, e: np.ndarray | None = None) -> float:
    """Doubly robust ATE: outcome-model residuals reweighted by 1/e.

    Consistent if either the propensity model or the arm-specific outcome
    models are correct.
    """
    if e is None:
        e = propensity_scores(df)
    y = df["perf_90d"].to_numpy()
    w = df["group"].to_numpy()
    x = df["assessment_score"].to_numpy()
    terms = []
    for arm, arm_w in ((1, w), (0, 1 - w)):
        mask = df["group"].to_numpy() == arm
        Xd = np.column_stack([np.ones(mask.sum()), x[mask]])
        b, *_ = np.linalg.lstsq(Xd, y[mask], rcond=None)
        mu = b[0] + b[1] * x  # predicted potential outcome under this arm
        denom = e if arm == 1 else 1 - e
        terms.append(mu + arm_w * (y - mu) / denom)
    return float(np.mean(terms[0] - terms[1]))


def bootstrap_ci(df: pd.DataFrame, estimator, B: int = 1000,
                 seed: int = 777) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for an estimator fn(df) -> float."""
    rng = np.random.default_rng(seed)
    n = len(df)
    ests = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        ests[b] = estimator(df.iloc[idx].reset_index(drop=True))
    return float(np.quantile(ests, 0.025)), float(np.quantile(ests, 0.975))


def summarize(df: pd.DataFrame) -> dict:
    """All four estimators + bootstrap CIs on the observational data."""
    e = propensity_scores(df)
    results = {}
    for name, fn in (("naive", naive),
                     ("ipw", lambda d: ipw(d, e=None)),
                     ("outcome_regression", outcome_regression),
                     ("doubly_robust", lambda d: doubly_robust(d, e=None))):
        est = fn(df)
        lo, hi = bootstrap_ci(df, fn, B=1000, seed=777)
        results[name] = {"ate": est, "ci_lo": lo, "ci_hi": hi}
    return results


if __name__ == "__main__":
    from src.data_gen import TRUE_TAU, generate_observational

    df = generate_observational()
    b0, b1 = fit_propensity(df["assessment_score"].to_numpy(), df["group"].to_numpy())
    print(f"true tau = {TRUE_TAU}; estimated propensity model: logit(e) = {b0:.3f} + {b1:.3f}*X")
    for name, r in summarize(df).items():
        print(f"{name:>18}: ATE={r['ate']:.4f}  95% CI=[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]")
