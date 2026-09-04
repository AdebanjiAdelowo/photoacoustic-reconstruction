import inspect

import numpy as np

from src.baselines import time_reversal_reconstruction
from src.forward_model import build_domain_and_medium, sparse_view_sensor_array, simulate_sensor_data
from src.phantoms import random_blob_phantom


def _run_reconstruction(seed=0, grid_size=64, n_sensors=32, radius=24):
    phantom = random_blob_phantom(size=grid_size, seed=seed, n_blobs=2)
    domain, medium = build_domain_and_medium(grid_size)
    sensor_pos = sparse_view_sensor_array(n_sensors, radius, (grid_size // 2, grid_size // 2))
    recording, time_axis = simulate_sensor_data(phantom, domain, medium, sensor_pos)
    recon = time_reversal_reconstruction(recording, sensor_pos, domain, medium, time_axis)
    return phantom, recon


def test_no_ground_truth_leakage_by_construction():
    # The reconstruction function's signature must not accept the phantom/ground truth at all —
    # this is a structural guarantee against accidental leakage, not just a runtime check.
    sig = inspect.signature(time_reversal_reconstruction)
    assert "phantom" not in sig.parameters
    assert "ground_truth" not in sig.parameters
    assert "p0" not in sig.parameters


def test_reconstruction_shape_and_finiteness():
    phantom, recon = _run_reconstruction()
    assert recon.shape == phantom.shape
    assert np.isfinite(recon).all()


def test_reconstruction_correlates_with_ground_truth():
    # Sanity check, not a strong quality claim: a correctly-wired time-reversal reconstruction
    # should be clearly, substantially correlated with the true source, well above chance.
    phantom, recon = _run_reconstruction()
    corr = np.corrcoef(phantom.ravel(), recon.ravel())[0, 1]
    assert corr > 0.5, f"correlation with ground truth too low: {corr}"


def test_sparser_array_reconstructs_worse_or_equal():
    # Physical expectation the MVP is built to test: fewer sensors should not improve
    # reconstruction quality. Checked at the two MVP sparsity settings.
    phantom, recon_sparse = _run_reconstruction(n_sensors=16)
    _, recon_dense = _run_reconstruction(n_sensors=64)
    corr_sparse = np.corrcoef(phantom.ravel(), recon_sparse.ravel())[0, 1]
    corr_dense = np.corrcoef(phantom.ravel(), recon_dense.ravel())[0, 1]
    assert corr_dense >= corr_sparse - 0.05  # small tolerance for stochastic PSTD numerics
