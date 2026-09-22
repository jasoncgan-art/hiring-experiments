"""Causal-estimate checks on the synthetic observational data:
the naive estimate is upward-biased by the documented confounding, while IPW
and the doubly robust estimator land closer to the known true effect."""

from src import causal
from src.data_gen import TRUE_TAU, generate_observational


def test_naive_is_upward_biased():
    df = generate_observational()
    naive_est = causal.naive(df)
    assert naive_est - TRUE_TAU > 0.2, (
        f"documented confounding should push naive well above tau, "
        f"got naive={naive_est:.4f}"
    )


def test_ipw_closer_than_naive():
    df = generate_observational()
    naive_err = abs(causal.naive(df) - TRUE_TAU)
    ipw_err = abs(causal.ipw(df) - TRUE_TAU)
    assert ipw_err < naive_err, (
        f"IPW (err={ipw_err:.4f}) should beat naive (err={naive_err:.4f})"
    )
    assert ipw_err < 0.10


def test_doubly_robust_close_to_truth():
    df = generate_observational()
    dr = causal.doubly_robust(df)
    assert abs(dr - TRUE_TAU) < 0.10, f"DR={dr:.4f} too far from tau={TRUE_TAU}"


def test_propensity_model_recovers_confounding_direction():
    # True propensity: logit(e) = 0.2 + 1.0 * X. Estimated slope should be
    # positive and in the right ballpark.
    import numpy as np

    df = generate_observational()
    b0, b1 = causal.fit_propensity(
        df["assessment_score"].to_numpy(), df["group"].to_numpy()
    )
    assert b1 > 0.5
    assert abs(b1 - 1.0) < 0.3
    e = causal.propensity_scores(df)
    assert np.all((e > 0.01) & (e < 0.99))
