"""Forward acoustic-wave simulation, implemented on top of j-Wave.

Documented configuration (verified, not assumed):
- Grid: 2D, square, N x N points, spacing dx (metres) — both configurable, see DEFAULT_* below.
- Discretisation: j-Wave's default FourierSeries/PSTD scheme (spectral method), the same class of
  solver as k-Wave's kspaceFirstOrderND (confirmed via jwave's own docstring).
- Time discretisation: derived automatically from a CFL condition via jwave.geometry.TimeAxis.from_
  medium — not manually chosen.
- Sound speed: homogeneous medium, constant scalar (default 1500 m/s, water-like).
- Boundary treatment: PML (perfectly matched layer) absorbing boundary, jwave's default
  (pml_size=20.0 grid points) — an open-domain assumption appropriate for photoacoustic imaging in
  a coupling medium, not a periodic or reflecting boundary.
- Initial-pressure convention: p0 is passed directly as the FourierSeries-wrapped phantom array;
  initial velocity u0 defaults to zero (standard photoacoustic initial-value-problem setup).
- Sensor arrangement: circular array via jwave.geometry.points_on_circle, positions in grid-index
  units.
- Output tensor: (Nt, n_sensors, 1) float32 — verified empirically in Stage 1's smoke test, not
  assumed from documentation.
"""

import jax.numpy as jnp
import numpy as np
from jaxdf.discretization import FourierSeries
from jwave.acoustics import simulate_wave_propagation
from jwave.geometry import Domain, Medium, Sensors, TimeAxis, points_on_circle

DEFAULT_DX = 1e-4  # metres per grid point
DEFAULT_SOUND_SPEED = 1500.0  # m/s, water-like
DEFAULT_DENSITY = 1000.0  # kg/m^3, water-like


def build_domain_and_medium(grid_size: int, dx: float = DEFAULT_DX,
                             sound_speed: float = DEFAULT_SOUND_SPEED,
                             density: float = DEFAULT_DENSITY):
    """Construct a square (grid_size, grid_size) homogeneous-medium j-Wave domain."""
    domain = Domain((grid_size, grid_size), (dx, dx))
    medium = Medium(domain=domain, sound_speed=sound_speed, density=density)
    return domain, medium


def sparse_view_sensor_array(n_sensors: int, radius: float, centre: tuple):
    """Circular sensor array of n_sensors positions (grid-index units), via jwave's own
    points_on_circle helper (verified to exist and have this exact signature — Stage 1).

    "Sparse-view" in this project means: fewer sensors on the same fixed-radius circular array,
    i.e. angular under-sampling of an otherwise full 360-degree array — not a reduced radius or a
    partial-arc (limited-angle) array. Both MVP sparsity settings use this same geometry, differing
    only in n_sensors.
    """
    sx, sy = points_on_circle(n_sensors, radius, centre)
    return jnp.array(sx), jnp.array(sy)


def simulate_sensor_data(p0: np.ndarray, domain: Domain, medium: Medium, sensor_positions):
    """Run the forward simulation: p0 (as a numpy phantom array) -> sensor recordings.

    Returns:
        numpy array of shape (Nt, n_sensors, 1) — see module docstring for the verified layout.
    """
    p0_field = FourierSeries(jnp.array(p0), domain)
    time_axis = TimeAxis.from_medium(medium, cfl=0.3)
    sensors = Sensors(positions=sensor_positions)
    recording = simulate_wave_propagation(medium, time_axis, p0=p0_field, sensors=sensors)
    return np.asarray(recording), time_axis
