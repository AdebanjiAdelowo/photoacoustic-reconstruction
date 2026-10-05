"""Mathematical checks of the unrolled Tikhonov network on small random operators (no simulation)."""
import numpy as np
import pytest
import torch

from src.device import cpu_state_dict
from src.tikhonov import Tikhonov
from src.unrolled import SpectralOperator, UnrolledTikhonov, data_consistency, normalised_backprojection

GRID = 8
N = GRID * GRID


@pytest.fixture(scope="module")
def problems():
    """Two 'sensor geometries': random forward matrices with different numbers of measurements."""
    rng = np.random.default_rng(0)
    out = []
    for m in (40, 90):  # one under-determined, one over-determined
        tk = Tikhonov(rng.normal(size=(m, N)))
        out.append((tk, SpectralOperator(tk.AtA)))
    return out


def _b(tk, op, y):
    return torch.as_tensor(normalised_backprojection(tk.A, op.norm, y))


def test_spectral_operator_is_normalised(problems):
    tk, op = problems[1]
    assert op.norm == pytest.approx(tk.norm_AtA, rel=1e-10)
    assert op.s.max() == pytest.approx(1.0) and op.s.min() >= 0.0
    assert np.allclose(op.V @ np.diag(op.s * op.norm) @ op.V.T, tk.AtA, atol=1e-8 * op.norm)


@pytest.mark.parametrize("mu", [1e-6, 1e-3, 1e-1])
def test_data_consistency_solves_the_regularised_normal_equations(problems, mu):
    rng = np.random.default_rng(1)
    for tk, op in problems:
        y, z = rng.normal(size=(3, tk.A.shape[0])), rng.normal(size=(3, N))
        V, s = op.tensors("cpu")
        x = data_consistency(V, s, _b(tk, op, y), torch.as_tensor(z), torch.tensor(mu, dtype=torch.float64)).numpy()
        lam = mu * tk.norm_AtA  # minimiser of ||A x - y||^2 + lam ||x - z||^2
        expected = np.linalg.solve(tk.AtA + lam * np.eye(N), (y @ tk.A + lam * z).T).T
        assert np.allclose(x, expected, rtol=1e-7, atol=1e-9)


def test_without_iterations_the_model_is_plain_tikhonov(problems):
    tk, op = problems[1]
    y = np.random.default_rng(2).normal(size=(4, tk.A.shape[0]))
    model = UnrolledTikhonov(geometries=(16,), iterations=0, init_log10_mu=-3.0)
    out = model(_b(tk, op, y), torch.zeros(4, dtype=torch.long), [op.tensors("cpu")], GRID)
    assert out.shape == (4, 1, GRID, GRID) and out.dtype == torch.float32
    assert np.allclose(out.detach().numpy().reshape(4, N), tk.solve(y, 1e-3), atol=1e-5)


def test_untrained_model_is_iterated_tikhonov(problems):
    tk, op = problems[0]
    y = np.random.default_rng(3).normal(size=(2, tk.A.shape[0]))
    mu, K = 1e-2, 3
    model = UnrolledTikhonov(geometries=(16,), iterations=K, init_log10_mu=-2.0)
    out = model(_b(tk, op, y), torch.zeros(2, dtype=torch.long), [op.tensors("cpu")], GRID).detach().numpy().reshape(2, N)
    lam = mu * tk.norm_AtA
    solve = lambda rhs: np.linalg.solve(tk.AtA + lam * np.eye(N), rhs.T).T  # noqa: E731
    x = solve(y @ tk.A)
    for _ in range(K):
        x = solve(y @ tk.A + lam * x.astype(np.float32))  # the image passes through float32 between steps
    assert np.allclose(out, x, atol=1e-5)


def test_each_sample_uses_the_operator_of_its_own_geometry(problems):
    rng = np.random.default_rng(4)
    (tk_a, op_a), (tk_b, op_b) = problems
    ya, yb = rng.normal(size=(2, tk_a.A.shape[0])), rng.normal(size=(1, tk_b.A.shape[0]))
    model = UnrolledTikhonov(geometries=(16, 64), iterations=0, init_log10_mu=-2.0)
    b = torch.cat([_b(tk_a, op_a, ya)[:1], _b(tk_b, op_b, yb), _b(tk_a, op_a, ya)[1:]])
    out = model(b, torch.tensor([0, 1, 0]), [op_a.tensors("cpu"), op_b.tensors("cpu")], GRID).detach().numpy().reshape(3, N)
    assert np.allclose(out[[0, 2]], tk_a.solve(ya, 1e-2), atol=1e-5)
    assert np.allclose(out[1], tk_b.solve(yb[0], 1e-2), atol=1e-5)


def test_gradients_reach_the_network_and_the_weights(problems):
    tk, op = problems[1]
    rng = np.random.default_rng(5)
    y, target = rng.normal(size=(4, tk.A.shape[0])), torch.as_tensor(rng.random((4, 1, GRID, GRID)), dtype=torch.float32)
    torch.manual_seed(0)
    model = UnrolledTikhonov(geometries=(16,), iterations=2)
    assert model.mu().shape == (1, 3) and np.allclose(model.mu(), 1e-2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    args = (_b(tk, op, y), torch.zeros(4, dtype=torch.long), [op.tensors("cpu")], GRID)
    losses = []
    for _ in range(30):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(*args), target)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        opt.step()
        losses.append(loss.item())
    assert model.log10_mu.grad.abs().sum() > 0 and model.denoiser.net[0].weight.grad.abs().sum() > 0
    assert losses[-1] < losses[0]


def test_checkpoint_round_trip(problems, tmp_path):
    tk, op = problems[1]
    torch.manual_seed(1)
    model = UnrolledTikhonov(geometries=(16,), iterations=2)
    torch.nn.init.normal_(model.denoiser.net[-1].weight, std=0.05)
    torch.save({"model_state": cpu_state_dict(model)}, tmp_path / "m.pt")
    other = UnrolledTikhonov(geometries=(16,), iterations=2)
    other.load_state_dict(torch.load(tmp_path / "m.pt", weights_only=True)["model_state"])
    args = (_b(tk, op, np.random.default_rng(6).normal(size=(2, tk.A.shape[0]))), torch.zeros(2, dtype=torch.long),
            [op.tensors("cpu")], GRID)
    with torch.no_grad():
        assert torch.equal(model(*args), other(*args))
