import numpy as np
import pytest

from src.stats import bootstrap_mean_ci, paired_bootstrap_ci


def test_ci_contains_the_sample_mean_and_is_ordered():
    v = np.random.default_rng(0).normal(3.0, 1.0, 200)
    m, lo, hi = bootstrap_mean_ci(v)
    assert lo < m < hi
    assert np.isclose(m, v.mean())


def test_ci_width_matches_the_standard_error():
    # for a large normal sample the 95% percentile interval is close to mean +/- 1.96 * sd / sqrt(n)
    v = np.random.default_rng(1).normal(0.0, 2.0, 400)
    _, lo, hi = bootstrap_mean_ci(v, n_boot=4000)
    expected = 2 * 1.96 * v.std(ddof=1) / np.sqrt(len(v))
    assert abs((hi - lo) - expected) / expected < 0.15


def test_ci_is_deterministic_for_a_seed():
    v = np.arange(50.0)
    assert bootstrap_mean_ci(v, seed=3) == bootstrap_mean_ci(v, seed=3)


def test_paired_ci_uses_per_image_differences():
    rng = np.random.default_rng(2)
    base = rng.normal(20, 5, 300)           # large between-image spread
    a, b = base + 1.0 + rng.normal(0, 0.1, 300), base
    m, lo, hi = paired_bootstrap_ci(a, b)
    assert np.isclose(m, 1.0, atol=0.05)
    assert hi - lo < 0.1                    # pairing removes the between-image spread
    with pytest.raises(ValueError):
        paired_bootstrap_ci(a, b[:-1])
