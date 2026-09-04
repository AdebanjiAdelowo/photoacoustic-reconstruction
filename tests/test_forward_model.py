import numpy as np

from src.forward_model import build_domain_and_medium, sparse_view_sensor_array, simulate_sensor_data
from src.phantoms import random_blob_phantom


def _run_small_case():
    grid_size = 32
    phantom = random_blob_phantom(size=grid_size, seed=0, n_blobs=2)
    domain, medium = build_domain_and_medium(grid_size)
    # radius=15, not 12: verified empirically (see IMPLEMENTATION_LOG.md Stage 3) that radius=12
    # placed one sensor within the Gaussian phantom's non-negligible tail, contaminating the
    # "signal starts near zero" check with near-field pickup rather than wave-arrival delay.
    sensor_pos = sparse_view_sensor_array(n_sensors=8, radius=15, centre=(grid_size // 2, grid_size // 2))
    recording, time_axis = simulate_sensor_data(phantom, domain, medium, sensor_pos)
    return recording, time_axis


def test_recording_shape_and_finiteness():
    recording, time_axis = _run_small_case()
    assert recording.shape[0] == int(time_axis.Nt)
    assert recording.shape[1] == 8  # n_sensors
    assert np.isfinite(recording).all()


def test_recording_is_not_trivially_zero():
    recording, _ = _run_small_case()
    assert np.abs(recording).max() > 1e-4


def test_signal_starts_near_zero_before_wave_arrival():
    # Qualitative physical sanity check, not a strict numerical assertion (per
    # IMPLEMENTATION_PLAN.md Stage 10): the very first time sample should be near zero, since the
    # wave has not yet had time to reach any sensor.
    recording, _ = _run_small_case()
    first_sample = recording[0, :, 0]
    assert np.abs(first_sample).max() < 1e-3
