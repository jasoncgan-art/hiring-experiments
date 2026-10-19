import pytest
from src.power import power_for, required_n_per_arm


def test_two_sided_null_rejection_rate_and_sign_symmetry():
    assert power_for(0, 100) == pytest.approx(0.05)
    assert power_for(0.1, 100) == pytest.approx(power_for(-0.1, 100))


def test_unadjusted_requires_more_observations_than_ideal_adjusted():
    assert required_n_per_arm(0.35) > required_n_per_arm(0.35, sigma=1)
