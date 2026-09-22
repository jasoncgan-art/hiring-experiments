"""Analysis checks on the synthetic experiment:
CUPED reduces variance vs raw diff-in-means, and 95% CIs cover the known
true effect (tau = +0.35) on the fixed-seed data."""

from src import analysis
from src.data_gen import TRUE_TAU, generate_experiment


def test_cuped_reduces_variance():
    df = generate_experiment()
    cu = analysis.cuped(df)
    assert cu["var_reduction_pct"] > 10.0, (
        f"CUPED should cut variance substantially on this DGP, "
        f"got {cu['var_reduction_pct']:.1f}%"
    )


def test_raw_ci_covers_true_effect():
    df = generate_experiment()
    raw = analysis.diff_in_means(df)
    assert raw["ci_lo"] <= TRUE_TAU <= raw["ci_hi"], (
        f"raw CI [{raw['ci_lo']:.4f}, {raw['ci_hi']:.4f}] misses tau={TRUE_TAU}"
    )


def test_cuped_ci_covers_true_effect():
    df = generate_experiment()
    cu = analysis.cuped(df)
    assert cu["ci_lo"] <= TRUE_TAU <= cu["ci_hi"], (
        f"CUPED CI [{cu['ci_lo']:.4f}, {cu['ci_hi']:.4f}] misses tau={TRUE_TAU}"
    )


def test_ols_adjusted_close_to_cuped():
    # ANCOVA and CUPED-with-pooled-theta are algebraically near-identical.
    df = generate_experiment()
    cu = analysis.cuped(df)
    ols = analysis.ols_adjusted(df)
    assert abs(cu["ate"] - ols["ate"]) < 0.01


def test_effect_is_significant():
    df = generate_experiment()
    raw = analysis.diff_in_means(df)
    assert raw["p_value"] < 0.05
    assert raw["ate"] > 0
