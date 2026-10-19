"""Statistical power analysis for the hiring-rubric A/B experiment.

Assumes the standard two-sample comparison (equal arms, common variance):
for n_per_arm applicants per arm and outcome residual std sigma,

    SE(diff-in-means) = sigma * sqrt(2 / n_per_arm)
    MDE(alpha, power) = (z_{1-alpha/2} + z_{power}) * SE
    n_per_arm(delta)  = ceil(2 * ((z_{1-alpha/2} + z_{power}) * sigma / delta)^2)

Default sigma is sqrt(2): within-arm outcome SD before covariate adjustment.
Use sigma=1 only for ideal adjustment under this known synthetic DGP.
"""

from __future__ import annotations

import math

from scipy.stats import norm

ALPHA = 0.05
POWER = 0.80
SIGMA = math.sqrt(2.0)


def mde(n_per_arm: int, sigma: float = SIGMA,
        alpha: float = ALPHA, power: float = POWER) -> float:
    """Minimum detectable effect at the given sample size."""
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return float(z * sigma * math.sqrt(2.0 / n_per_arm))


def required_n_per_arm(delta: float, sigma: float = SIGMA,
                       alpha: float = ALPHA, power: float = POWER) -> int:
    """Applicants per arm needed to detect an effect of size `delta`."""
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return math.ceil(2 * (z * sigma / delta) ** 2)


def power_for(delta: float, n_per_arm: int, sigma: float = SIGMA,
              alpha: float = ALPHA) -> float:
    """Statistical power to detect an effect of size `delta`."""
    se = sigma * math.sqrt(2.0 / n_per_arm)
    z = norm.ppf(1 - alpha / 2)
    return float(norm.cdf(delta / se - z) + norm.cdf(-delta / se - z))


def power_table(deltas=(0.20, 0.30, 0.35, 0.50), total_ns=(200, 500, 1000, 2000, 4000)):
    """Rows of (total N, per-arm n, MDE, power at each delta)."""
    rows = []
    for total_n in total_ns:
        n_arm = total_n // 2
        rows.append(
            {
                "total_n": total_n,
                "n_per_arm": n_arm,
                "mde": mde(n_arm),
                **{f"power_d{d}": power_for(d, n_arm) for d in deltas},
            }
        )
    return rows


def print_power_table() -> None:
    deltas = (0.20, 0.30, 0.35, 0.50)
    header = f"{'total N':>8} {'per arm':>8} {'MDE':>7} " + " ".join(f"{'pwr ' + str(d):>10}" for d in deltas)
    print(header)
    print("-" * len(header))
    for row in power_table(deltas=deltas):
        line = f"{row['total_n']:>8} {row['n_per_arm']:>8} {row['mde']:>7.3f} "
        line += " ".join(f"{row[f'power_d{d}']:>10.3f}" for d in deltas)
        print(line)
    print()
    print(f"To detect delta=0.35 (the true effect) at 80% power, alpha=0.05: "
          f"{required_n_per_arm(0.35)} per arm "
          f"({2 * required_n_per_arm(0.35)} total applicants).")


if __name__ == "__main__":
    print_power_table()

