"""Forward acoustic-wave simulation wrapper around j-Wave (verified choice — ARCHITECTURE.md
Section 5). Not a from-scratch FDTD implementation. Not implemented.

Implementation note: follow j-Wave's official example notebook for photoacoustic initial-value
problems (ucl-bug.github.io/jwave) at implementation time rather than trusting the specific class
names sketched in comments below — this project's architecture review verified that j-Wave
supports this workflow, not the exact current API surface.
"""


def simulate_sensor_data(p0, sensor_positions, sound_speed: float = 1500.0):
    """Simulate sensor time series y_k(t) from an initial pressure field p0, via j-Wave.

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
