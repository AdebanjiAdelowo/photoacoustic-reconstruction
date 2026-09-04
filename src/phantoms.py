"""Synthetic phantom generation.

Implemented for the MVP. See IMPLEMENTATION_LOG.md Stage 2 for the design correction this
represents relative to the original architecture plan (which said "Shepp-Logan-style").

Coordinate convention: phantoms are (size, size) float32 numpy arrays, axis 0 = x, axis 1 = y
(numpy 'ij' meshgrid indexing), origin at array index [0, 0]. Values are a dimensionless initial
pressure amplitude in [0, 1] (0 = no absorber, 1 = peak absorber response) — this is a synthetic
convention, not calibrated to any real photoacoustic units.
"""

import numpy as np


def random_blob_phantom(size: int, seed: int, n_blobs: int = 3) -> np.ndarray:
    """Generate a deterministic phantom made of a small number of Gaussian blobs.

    Design rationale (correction from the original "Shepp-Logan-style" plan): Shepp-Logan is a
    CT X-ray-attenuation phantom (a stylised head cross-section) and is not a physically
    appropriate model for a photoacoustic initial-pressure source, which is a localised optical
    absorber (e.g. a blood vessel or lesion), not an anatomical attenuation map. Randomly
    (but reproducibly) placed Gaussian blobs are a standard, honest stand-in for such absorbers
    and — unlike scikit-image's fixed single Shepp-Logan reference image — give a real family of
    distinct instances for building train/val/test splits (Stage 7).

    Args:
        size: phantom is (size, size).
        seed: integer seed — same (size, seed, n_blobs) always produces the same phantom.
        n_blobs: number of Gaussian blobs.

    Returns:
        (size, size) float32 array, values in [0, 1].
    """
    rng = np.random.default_rng(seed)
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y, indexing="ij")

    img = np.zeros((size, size), dtype=np.float64)
    for _ in range(n_blobs):
        cx = rng.uniform(0.3, 0.7) * size
        cy = rng.uniform(0.3, 0.7) * size
        sigma = rng.uniform(0.03, 0.07) * size
        amplitude = rng.uniform(0.5, 1.0)
        img += amplitude * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma**2))

    img = np.clip(img, 0.0, None)
    peak = img.max()
    if peak > 0:
        img = img / peak

    return img.astype(np.float32)
