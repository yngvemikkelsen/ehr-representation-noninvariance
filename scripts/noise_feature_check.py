#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
Implementation check for the held-out log-score estimator
(manuscript: Methods, Statistical analysis).

Out-of-sample log-likelihood gain is used in place of a permutation null on
the grounds that an uninformative feature costs likelihood on data it was not
fitted to, however many levels it has. This verifies that property on
synthetic data with a known answer:

  - a 20-level categorical feature carrying NO information should give a
    small positive in-sample gain (extra parameters always fit noise) and a
    small NEGATIVE held-out gain
  - the same feature carrying real information should give a positive gain
    both in sample and held out

No clinical data are used. Runs in seconds.

Usage:  ./noise_feature_check.py
"""

import numpy as np


def logistic(X, y, iters=40, ridge=1e-4):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = np.clip(X @ b, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        W = np.clip(p * (1 - p), 1e-9, None)
        g = X.T @ (y - p) - ridge * b
        H = (X * W[:, None]).T @ X + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        b = b + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return b


def loglik(X, y, b):
    eta = np.clip(X @ b, -30, 30)
    p = np.clip(1 / (1 + np.exp(-eta)), 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main() -> None:
    rng = np.random.default_rng(0)
    print(f"{'case':<26}{'in-sample gain':>18}{'held-out gain':>18}")
    for name, informative in (("noise feature, 20 levels", False),
                              ("real feature, 20 levels", True)):
        n = 200000
        z = rng.normal(size=n)
        x = rng.integers(0, 20, n)
        lin = -3.4 + 0.8 * z + (0.9 * (x >= 17) if informative else 0)
        y = (rng.random(n) < 1 / (1 + np.exp(-lin))).astype(float)
        tr = rng.random(n) < 0.8
        te = ~tr
        Xb = np.column_stack([np.ones(n), z])
        F = np.column_stack([(x == c).astype(float) for c in range(1, 20)])
        Xf = np.column_stack([Xb, F])
        b0, b1 = logistic(Xb[tr], y[tr]), logistic(Xf[tr], y[tr])
        ins = (loglik(Xf[tr], y[tr], b1) - loglik(Xb[tr], y[tr], b0)) / tr.sum()
        oos = (loglik(Xf[te], y[te], b1) - loglik(Xb[te], y[te], b0)) / te.sum()
        print(f"{name:<26}{ins:>+18.6f}{oos:>+18.6f}")
    print("\nExpected: noise feature positive in sample, negative held out;")
    print("real feature positive in both.")


if __name__ == "__main__":
    main()
