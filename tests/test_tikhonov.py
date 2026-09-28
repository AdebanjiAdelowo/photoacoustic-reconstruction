"""Mathematical checks of the Tikhonov solver on a small grid."""
import numpy as np
import pytest

from src.forward_model import build_domain_and_medium, simulate_sensor_data, sparse_view_sensor_array
from src.phantoms import random_blob_phantom
from src.tikhonov import Tikhonov, assemble_matrix

N = 32


@pytest.fixture(scope="module")
def setup():
    domain, medium = build_domain_and_medium(N)
    sensors = sparse_view_sensor_array(16, 12, (N // 2, N // 2))
    return domain, medium, sensors, Tikhonov(assemble_matrix(domain, medium, sensors))


def test_matrix_reproduces_the_project_forward_model(setup):
    domain, medium, sensors, tk = setup
    p = random_blob_phantom(size=N, seed=1, n_blobs=3)
    recording, _ = simulate_sensor_data(p, domain, medium, sensors)
    y = recording[..., 0].reshape(-1)
    assert np.linalg.norm(tk.A @ p.reshape(-1) - y) / np.linalg.norm(y) < 1e-5


def test_adjoint_identity(setup):
    tk = setup[3]
    rng = np.random.default_rng(0)
    x, r = rng.normal(size=tk.A.shape[1]), rng.normal(size=tk.A.shape[0])
    assert np.isclose((tk.A @ x) @ r, x @ (tk.A.T @ r), rtol=1e-12)


@pytest.mark.parametrize("point_seed", [1, 11, 21, 31])
@pytest.mark.parametrize("mu", [1e-6, 1e-2])
def test_gradient_matches_finite_differences(setup, point_seed, mu):
    # several random points and directions; the objective is quadratic, so the central difference
    # is exact up to round-off and the relative error must be tiny for any step
    tk = setup[3]
    rng = np.random.default_rng(point_seed)
    y = tk.A @ random_blob_phantom(size=N, seed=2, n_blobs=2).reshape(-1)
    p, d = rng.normal(size=tk.A.shape[1]), rng.normal(size=tk.A.shape[1])
    d /= np.linalg.norm(d)
    g = tk.gradient(p, y, mu) @ d
    for h in (1e-2, 1e-3):
        fd = (tk.objective(p + h * d, y, mu) - tk.objective(p - h * d, y, mu)) / (2 * h)
        assert abs(fd - g) / abs(g) < 1e-6


@pytest.mark.parametrize("mu", [1e-5, 1e-3, 1e-1])
def test_solution_is_the_minimiser(setup, mu):
    tk = setup[3]
    rng = np.random.default_rng(3)
    y = tk.A @ random_blob_phantom(size=N, seed=4, n_blobs=2).reshape(-1) + 1e-3 * rng.normal(size=tk.A.shape[0])
    p = tk.solve(y, mu)
    grad = tk.gradient(p, y, mu)
    assert np.linalg.norm(grad) / np.linalg.norm(2 * tk.A.T @ y) < 1e-8   # normal equations hold
    f0 = tk.objective(p, y, mu)
    for _ in range(5):                                                      # any perturbation is worse
        assert tk.objective(p + 1e-3 * rng.normal(size=p.shape), y, mu) > f0


def test_batched_solve_matches_single_solves(setup):
    tk = setup[3]
    Y = np.stack([tk.A @ random_blob_phantom(size=N, seed=s, n_blobs=2).reshape(-1) for s in (5, 6)])
    assert np.allclose(tk.solve(Y, 1e-3), np.stack([tk.solve(y, 1e-3) for y in Y]))


def test_regularisation_behaviour_on_a_known_phantom(setup):
    # With 16 sensors part of the image is unobservable (about a fifth of A's singular values are
    # below 1e-6 of the largest on this grid), so even noiseless data cannot give exact recovery.
    # What weak regularisation must do is fit the data; stronger regularisation trades data fit for
    # a smaller-norm estimate further from the truth.
    tk = setup[3]
    p_true = random_blob_phantom(size=N, seed=7, n_blobs=2).reshape(-1)
    y = tk.A @ p_true
    mus = (1e-8, 1e-3, 1e-1)
    est = {mu: tk.solve(y, mu) for mu in mus}
    residual = [np.linalg.norm(tk.A @ est[mu] - y) / np.linalg.norm(y) for mu in mus]
    error = [np.linalg.norm(est[mu] - p_true) / np.linalg.norm(p_true) for mu in mus]
    norm = [np.linalg.norm(est[mu]) for mu in mus]
    assert residual[0] < 1e-4
    assert residual == sorted(residual) and error == sorted(error)
    assert norm == sorted(norm, reverse=True)
    assert error[0] < 0.15
