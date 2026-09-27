"""Bootstrap confidence intervals over test images.

Resampling is over images only. Variation between independently trained networks is a separate
source of uncertainty and is reported separately (see scripts/expanded_evaluation.py), not folded in.
"""

import numpy as np


def bootstrap_mean_ci(values, n_boot: int = 2000, level: float = 0.95, seed: int = 0):
    """Mean of `values` and a percentile bootstrap CI for it, resampling the values with replacement."""
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return float(v.mean()), float(lo), float(hi)


def paired_bootstrap_ci(a, b, **kwargs):
    """Mean of the per-image difference a - b with a bootstrap CI (images resampled as pairs)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired samples must have the same shape")
    return bootstrap_mean_ci(a - b, **kwargs)
