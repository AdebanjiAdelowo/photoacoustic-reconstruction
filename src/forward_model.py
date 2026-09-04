"""Forward acoustic-wave simulation wrapper.

Planned to wrap `jwave` or `k-wave-python` (see ARCHITECTURE.md Section 5) — deliberately not a
from-scratch FDTD implementation, to protect the project timeline. Not implemented.
"""


def simulate_sensor_data(p0, sensor_positions, sound_speed: float = 1500.0):
    """Simulate sensor time series y_k(t) from an initial pressure field p0.

    Governing equation: (1/c^2) d^2p/dt^2 - grad^2 p = 0, p(x,0) = p0, dp/dt(x,0) = 0.
    See ARCHITECTURE.md Section 2 for the full formulation.

    Planned. Not implemented.
    """
    raise NotImplementedError("Planned — see ARCHITECTURE.md Sections 2 and 5")


def sparse_view_sensor_array(n_sensors: int, radius: float):
    """Construct a sparse circular sensor array with n_sensors positions.

    Planned. Not implemented.
    """
    raise NotImplementedError("Planned — see ARCHITECTURE.md Section 2")
