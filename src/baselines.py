"""Classical reconstruction baselines: filtered back-projection and Tikhonov inversion.

Both are required baselines (ARCHITECTURE.md Section 5) — the learned model must be evaluated
against both, not a strawman. Not implemented.
"""


def filtered_back_projection(sensor_data, sensor_positions, grid_shape):
    """Closed-form FBP reconstruction of the initial pressure field.

    Planned. Not implemented.
    """
    raise NotImplementedError("Planned — see ARCHITECTURE.md Section 2")


def tikhonov_reconstruction(sensor_data, forward_operator, reg_lambda: float):
    """Tikhonov-regularised least-squares reconstruction:
    argmin_p0 ||A p0 - y||_2^2 + lambda * ||p0||_2^2

    Planned. Not implemented.
    """
    raise NotImplementedError("Planned — see ARCHITECTURE.md Section 2")
