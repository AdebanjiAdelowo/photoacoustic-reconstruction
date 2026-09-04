"""Classical reconstruction baselines: time-reversal and Tikhonov inversion.

Both are required baselines (ARCHITECTURE.md Section 5) — the learned model must be evaluated
against both, not a strawman. Not implemented.
"""


def time_reversal_reconstruction(sensor_data, sensor_positions, grid_shape):
    """Time-reversal reconstruction of the initial pressure field via j-Wave's documented
    photoacoustic-reconstruction workflow (homogeneous medium).

    This is the wave-equation analogue of filtered back-projection for this problem class.
    Planned. Not implemented — implement by following j-Wave's official example notebook for
    initial-value-problem photoacoustic reconstruction (ARCHITECTURE.md Section 5), not by
    guessing at API details.
    """
    raise NotImplementedError("Planned — see ARCHITECTURE.md Sections 2 and 5")


def tikhonov_reconstruction(sensor_data, forward_simulate_fn, reg_lambda: float):
    """Tikhonov-regularised reconstruction via gradient descent through j-Wave's differentiable
    forward simulation:
    argmin_p0 ||forward_simulate_fn(p0) - y||_2^2 + lambda * ||p0||_2^2

    Takes the forward simulation function directly (not a precomputed operator matrix) — this is
    only possible because j-Wave is JAX-differentiable, a technical advantage discovered during
    architecture review (ARCHITECTURE.md Section 2). Planned. Not implemented.
    """
    raise NotImplementedError("Planned — see ARCHITECTURE.md Section 2")
