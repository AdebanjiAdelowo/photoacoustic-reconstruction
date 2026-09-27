import numpy as np

from src.calibration import fit_scale, fit_scales_per_sensor_count


def test_fit_scale_recovers_a_known_gain():
    rng = np.random.default_rng(0)
    gt = rng.random((5, 16, 16))
    assert np.isclose(fit_scale(gt / 7.0, gt), 7.0)


def test_fit_scale_is_the_least_squares_minimiser():
    rng = np.random.default_rng(1)
    recon, gt = rng.random((3, 8, 8)), rng.random((3, 8, 8))
    a = fit_scale(recon, gt)
    err = lambda s: np.sum((s * recon - gt) ** 2)
    assert err(a) <= min(err(a * 0.99), err(a * 1.01))


def test_per_sensor_count_scales_use_only_their_own_examples():
    rng = np.random.default_rng(2)
    gt = rng.random((6, 8, 8))
    n = np.array([16, 16, 16, 64, 64, 64])
    recon = gt / np.where(n == 16, 10.0, 2.0)[:, None, None]
    scales = fit_scales_per_sensor_count(recon, gt, n)
    assert set(scales) == {16, 64}
    assert np.isclose(scales[16], 10.0) and np.isclose(scales[64], 2.0)
