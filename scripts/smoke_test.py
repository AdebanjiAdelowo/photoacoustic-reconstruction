"""Environment/API verification smoke test.

Proves: jwave imports, JAX runs, a tiny acoustic initial-value-problem simulation executes and
produces sensor recordings of plausible shape. This is a verification script, not a research
experiment — no result here is a project finding.

Run: python scripts/smoke_test.py
"""
import time

import jax.numpy as jnp
import numpy as np
from jaxdf.discretization import FourierSeries
from jwave.acoustics import simulate_wave_propagation
from jwave.geometry import Domain, Medium, Sensors, TimeAxis, points_on_circle


def main():
    t0 = time.time()

    # Small 2D domain: 64x64 grid, 0.1mm spacing (matches a few-mm phantom, laptop-scale).
    N = (64, 64)
    dx = (1e-4, 1e-4)
    domain = Domain(N, dx)

    # Homogeneous water-like medium.
    medium = Medium(domain=domain, sound_speed=1500.0, density=1000.0)

    time_axis = TimeAxis.from_medium(medium, cfl=0.3)

    # Tiny Gaussian blob as the initial pressure p0 (stand-in phantom for the smoke test only —
    # real phantom generation is Stage 2, not this script).
    x = np.arange(N[0]) - N[0] // 2
    y = np.arange(N[1]) - N[1] // 2
    X, Y = np.meshgrid(x, y, indexing="ij")
    r2 = X**2 + Y**2
    p0_np = np.exp(-r2 / (2 * 4.0**2)).astype(np.float32)
    p0 = FourierSeries(jnp.array(p0_np), domain)

    # Circular sensor array, verifying points_on_circle's real signature.
    n_sensors = 16
    radius_px = 24
    centre = (N[0] // 2, N[1] // 2)
    sx, sy = points_on_circle(n_sensors, radius_px, centre)
    sensor_positions = (jnp.array(sx), jnp.array(sy))
    sensors = Sensors(positions=sensor_positions)

    print(f"Domain: {N}, dx: {dx}, Nt: {time_axis.Nt}")
    print(f"Sensor array: {n_sensors} sensors at radius {radius_px}px")

    recording = simulate_wave_propagation(medium, time_axis, p0=p0, sensors=sensors)

    recording_np = np.asarray(recording)
    elapsed = time.time() - t0

    print(f"Recording shape: {recording_np.shape}")
    print(f"Recording dtype: {recording_np.dtype}")
    print(f"Recording min/max: {recording_np.min():.6e} / {recording_np.max():.6e}")
    print(f"Recording is finite everywhere: {bool(np.isfinite(recording_np).all())}")
    print(f"Elapsed: {elapsed:.2f}s")

    # Verified empirically (not assumed): recording shape is (Nt, n_sensors, 1) — axis 0 is time,
    # axis 1 is the sensor dimension, axis 2 is a trailing singleton (scalar field component).
    assert np.isfinite(recording_np).all(), "Simulation produced non-finite values"
    assert recording_np.shape[0] == time_axis.Nt, "Recording length does not match Nt"
    assert recording_np.shape[1] == n_sensors, "Recording sensor dimension mismatch"
    assert np.abs(recording_np).max() > 0, "Recording is all-zero — simulation did nothing"

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
