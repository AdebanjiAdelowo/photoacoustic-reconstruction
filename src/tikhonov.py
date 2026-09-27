"""Tikhonov-regularised reconstruction through the j-Wave forward model.

Inverse problem. For a fixed sensor geometry the forward simulation (src/forward_model.py) is a
linear map A: p0 (N x N initial pressure, flattened) -> y (Nt x n_sensors recordings, flattened),
because the medium is homogeneous and time-invariant. The estimate is

    p_hat = argmin_p  ||A p - y||^2 + lam * ||p||^2,

whose unique minimiser solves the normal equations (A^T A + lam I) p = A^T y.

Solver. A is assembled explicitly, one column per pixel, by running the forward simulation on each
unit image (N^2 = 4096 runs on the 64 x 64 grid, about 90 s per sensor geometry). A^T is then the
exact matrix transpose and the normal equations are solved directly by Cholesky factorisation in
float64, so there is no iterative convergence question and one factorisation per lam serves every
image. Automatic differentiation was not used for A^T: on this j-Wave version neither jax.vjp nor
jax.linear_transpose through simulate_wave_propagation satisfies the adjoint identity
<A x, r> = <x, A^T r> (relative errors of about 0.25 to 0.5 were measured), so an autodiff-based
gradient solver would not be solving this problem.

The weight is parameterised as lam = mu * ||A^T A||_2 so that mu is dimensionless and comparable
between sensor geometries. mu must be selected on training/validation data only.
"""

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg
from jaxdf.discretization import FourierSeries
from jwave.acoustics import simulate_wave_propagation
from jwave.geometry import Sensors, TimeAxis


def forward_function(domain, medium, sensor_positions, cfl: float = 0.3):
    """The project's forward simulation as a jit-compiled, batched function (B, N, N) -> (B, Nt, n)."""
    time_axis = TimeAxis.from_medium(medium, cfl=cfl)
    sensors = Sensors(positions=sensor_positions)

    def A(p):
        return simulate_wave_propagation(medium, time_axis, p0=FourierSeries(p[..., None], domain),
                                         sensors=sensors)[..., 0]

    return jax.jit(jax.vmap(A))


def assemble_matrix(domain, medium, sensor_positions, batch: int = 64) -> np.ndarray:
    """Dense forward matrix, shape (Nt * n_sensors, N * N), column j = A applied to unit image j."""
    A = forward_function(domain, medium, sensor_positions)
    n_pix = int(np.prod(domain.N))
    cols = []
    for start in range(0, n_pix, batch):
        idx = np.arange(start, min(start + batch, n_pix))
        basis = np.zeros((len(idx), n_pix), np.float32)
        basis[np.arange(len(idx)), idx] = 1.0
        out = np.asarray(A(jnp.asarray(basis.reshape((len(idx),) + tuple(domain.N)))))
        cols.append(out.reshape(len(idx), -1))
    return np.concatenate(cols, axis=0).T


class Tikhonov:
    """Direct Tikhonov solver for one sensor geometry, given its dense forward matrix."""

    def __init__(self, A: np.ndarray):
        self.A = np.asarray(A, np.float64)
        self.AtA = self.A.T @ self.A
        self.norm_AtA = float(scipy.linalg.eigvalsh(self.AtA, subset_by_index=[len(self.AtA) - 1] * 2)[0])
        self._factors = {}

    def lam(self, mu: float) -> float:
        return mu * self.norm_AtA

    def solve(self, y, mu: float) -> np.ndarray:
        """Minimiser for recordings y of shape (m,) or (k, m); returns (n,) or (k, n)."""
        if mu not in self._factors:
            self._factors[mu] = scipy.linalg.cho_factor(self.AtA + self.lam(mu) * np.eye(len(self.AtA)))
        y = np.asarray(y, np.float64)
        rhs = (y.reshape(-1, self.A.shape[0]) @ self.A).T
        p = scipy.linalg.cho_solve(self._factors[mu], rhs).T
        return p[0] if y.ndim == 1 else p

    def objective(self, p, y, mu: float) -> float:
        r = self.A @ p - y
        return float(r @ r + self.lam(mu) * (p @ p))

    def gradient(self, p, y, mu: float) -> np.ndarray:
        return 2.0 * self.A.T @ (self.A @ p - y) + 2.0 * self.lam(mu) * p
