"""Classical reconstruction baseline: time-reversal.

Implemented for the MVP (Stage 4). Tikhonov inversion (originally planned as a second baseline) is
explicitly deferred past the MVP per IMPLEMENTATION_PLAN.md Stage 5b — not implemented here.

Method (standard time-reversal, verified against jwave's actual API — see IMPLEMENTATION_LOG.md
Stage 4): the recorded sensor signals are reversed in time and re-injected as sources (via
jwave.geometry.Sources) into a second forward simulation with zero initial pressure. The resulting
field at the *final* time step of this reversed simulation is the reconstruction estimate — this
is the standard time-reversal convention, and was verified empirically (not assumed) that
`simulate_wave_propagation` with `sensors=None` returns the full field at every time step (shape
`(Nt, Nx, Ny, 1)`), from which the final step is taken.
"""

import jax.numpy as jnp
import numpy as np
from jwave.acoustics import simulate_wave_propagation
from jwave.geometry import Sources


def time_reversal_reconstruction(recording: np.ndarray, sensor_positions, domain, medium, time_axis):
    """Reconstruct an initial-pressure estimate from recorded sensor data via time-reversal.

    Args:
        recording: (Nt, n_sensors, 1) array from src.forward_model.simulate_sensor_data.
        sensor_positions: same (x, y) tuple used to record `recording`.
        domain, medium, time_axis: same objects used for the forward simulation that produced
            `recording` (time-reversal re-uses the same medium/geometry, not a different one).

    Returns:
        (grid_size, grid_size) float32 reconstruction estimate.
    """
    reversed_signals = jnp.array(recording[::-1, :, 0].T)  # (n_sensors, Nt), time-reversed
    sources = Sources(positions=sensor_positions, signals=reversed_signals,
                       dt=time_axis.dt, domain=domain)
    field_over_time = simulate_wave_propagation(medium, time_axis, sources=sources)
    grid_data = np.asarray(field_over_time.on_grid)  # (Nt, grid, grid, 1)
    reconstruction = grid_data[-1, ..., 0]
    return reconstruction.astype(np.float32)
