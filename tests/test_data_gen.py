"""Data-generation determinism: same seed -> identical data; different seed -> different data."""

import numpy as np
import pandas as pd

from src.data_gen import generate_experiment, generate_observational


def test_experiment_deterministic():
    a = generate_experiment()
    b = generate_experiment()
    pd.testing.assert_frame_equal(a, b)


def test_experiment_seed_changes_output():
    a = generate_experiment(seed=1)
    b = generate_experiment(seed=2)
    assert not a.equals(b)
    # treatment assignment itself should differ, not just noise
    assert not np.array_equal(a["group"].to_numpy(), b["group"].to_numpy())


def test_observational_deterministic():
    a = generate_observational()
    b = generate_observational()
    pd.testing.assert_frame_equal(a, b)


def test_experiment_schema_and_balance():
    df = generate_experiment(n=2000)
    assert set(df.columns) == {
        "applicant_id", "assessment_score", "group", "group_label", "perf_90d"
    }
    assert df["group"].isin([0, 1]).all()
    # 50/50 randomization should be close to balanced at n=2000
    assert abs(df["group"].mean() - 0.5) < 0.05
    assert df["applicant_id"].is_unique
