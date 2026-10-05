"""Unrolled Tikhonov reconstruction: exact data-consistency solves with a learned network between them.

The two existing methods fail in opposite ways. Tikhonov (src/tikhonov.py) is the most accurate when
the forward model and the noise level are known, but its weight has to be chosen for the noise
level. The U-Net (src/reconstruction_net.py) needs no tuning but never uses the forward model: it
only refines a time-reversal image. This method uses both, in the form of a model-based unrolled
network (the scheme of Aggarwal, Mani and Jacob, "MoDL", IEEE TMI 2019, applied here to the explicit
photoacoustic forward matrix):

    x_0     = argmin_x ||A x - y||^2 + lam_0 ||x||^2                    (plain Tikhonov)
    z_k     = x_{k-1} + D(x_{k-1})                                      (learned residual CNN)
    x_k     = argmin_x ||A x - y||^2 + lam_k ||x - z_k||^2,  k = 1..K   (data consistency)

Each minimisation is solved exactly. With the eigendecomposition A^T A = V diag(s) V^T,

    x_k = V [ (V^T (A^T y + lam_k z_k)) / (s + lam_k) ],

so one factorisation per sensor geometry serves every weight, and the weights can be learned by
back-propagation. As in src/tikhonov.py the weights are lam = mu * ||A^T A||_2 with dimensionless mu.
The method is not told the noise level: mu_k is one learned number per iteration and sensor
geometry, the same for every image. D starts at zero, where the scheme is iterated Tikhonov
regularisation (a classical method in its own right), and with K = 0 it is plain Tikhonov.

The solves run in float64 (the 16-sensor matrix has condition number 3e10); the CNN runs in float32.
"""

import numpy as np
import scipy.linalg
import torch
import torch.nn as nn


class SpectralOperator:
    """Eigendecomposition of A^T A for one sensor geometry, scaled so that the largest eigenvalue is 1."""

    def __init__(self, AtA: np.ndarray):
        s, V = scipy.linalg.eigh(np.asarray(AtA, np.float64))
        self.norm = float(s[-1])                      # ||A^T A||_2
        self.s = np.clip(s / self.norm, 0.0, None)    # eigenvalues in [0, 1]
        self.V = V

    def tensors(self, device):
        return (torch.as_tensor(self.V, dtype=torch.float64, device=device),
                torch.as_tensor(self.s, dtype=torch.float64, device=device))


def normalised_backprojection(A: np.ndarray, norm: float, y: np.ndarray) -> np.ndarray:
    """b = A^T y / ||A^T A||_2 for recordings y of shape (k, m); returns (k, n) float64."""
    return (np.asarray(y, np.float64) @ A) / norm


def data_consistency(V, s, b, z, mu):
    """argmin_x ||A x - y||^2 + lam ||x - z||^2 in normalised units: (b + mu z) solved against
    (s + mu). b, z: (k, n) float64; mu: scalar tensor."""
    rhs = b + mu * z
    return ((rhs @ V) / (s + mu)) @ V.T


class ResidualDenoiser(nn.Module):
    """Small residual CNN applied between data-consistency steps (shared by all iterations)."""

    def __init__(self, features: int = 32, layers: int = 5):
        super().__init__()
        blocks = [nn.Conv2d(1, features, 3, padding=1), nn.LeakyReLU(0.1, inplace=True)]
        for _ in range(layers - 2):
            blocks += [nn.Conv2d(features, features, 3, padding=1), nn.LeakyReLU(0.1, inplace=True)]
        blocks.append(nn.Conv2d(features, 1, 3, padding=1))
        self.net = nn.Sequential(*blocks)
        nn.init.zeros_(self.net[-1].weight)  # start as the identity: the untrained model is Tikhonov
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class UnrolledTikhonov(nn.Module):
    """K learned iterations around exact Tikhonov solves.

    Args:
        geometries: sensor counts, in the order of the operators passed to `forward`.
        iterations: K, the number of denoise + data-consistency rounds after the initial solve.
        init_log10_mu: starting value of every log10(mu).
    """

    def __init__(self, geometries=(16, 64), iterations: int = 5, features: int = 32, layers: int = 5,
                 init_log10_mu: float = -2.0):
        super().__init__()
        self.geometries = tuple(int(g) for g in geometries)
        self.iterations = iterations
        self.denoiser = ResidualDenoiser(features, layers)
        self.log10_mu = nn.Parameter(torch.full((len(self.geometries), iterations + 1), float(init_log10_mu)))

    def mu(self):
        return 10.0 ** self.log10_mu.detach().cpu().numpy()

    def forward(self, b, geometry_index, operators, grid_size: int):
        """b: (B, n) float64 normalised back-projections; geometry_index: (B,) long, position of each
        sample's sensor count in `geometries`; operators: [(V, s), ...] per geometry.
        Returns (B, 1, grid, grid) float32."""
        out = torch.zeros(b.shape[0], 1, grid_size, grid_size, dtype=torch.float32, device=b.device)
        for g, (V, s) in enumerate(operators):
            sel = torch.nonzero(geometry_index == g).squeeze(1)
            if sel.numel() == 0:
                continue
            bg = b[sel]
            mu = (10.0 ** self.log10_mu[g]).to(torch.float64)
            x = data_consistency(V, s, bg, torch.zeros_like(bg), mu[0])
            for k in range(1, self.iterations + 1):
                image = x.to(torch.float32).reshape(-1, 1, grid_size, grid_size)
                z = (image + self.denoiser(image)).reshape(x.shape).to(torch.float64)
                x = data_consistency(V, s, bg, z, mu[k])
            out[sel] = x.to(torch.float32).reshape(-1, 1, grid_size, grid_size)
        return out
