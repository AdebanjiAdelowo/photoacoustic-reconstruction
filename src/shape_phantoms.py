"""Phantom families beyond the Gaussian blobs of src/phantoms.py.

Used to train and test reconstruction networks on a broader image distribution. Conventions are those
of src/phantoms.py: (size, size) float32, axis 0 = x, axis 1 = y, values in [0, 1] with peak 1, and a
given (family, size, seed) always produces the same image. Shapes are placed in the central region
so that they lie inside the sensor circle.

  blobs       1 to 3 smooth Gaussian blobs (src/phantoms.py, unchanged)
  discs       2 or 3 sharp-edged discs
  ellipses    1 to 3 sharp-edged rotated ellipses
  rectangles  1 to 3 sharp-edged rotated rectangles
  vessels     1 or 2 thin curved lines with a soft profile (vessel-like)
"""

import numpy as np

from src.phantoms import random_blob_phantom

FAMILIES = ("blobs", "discs", "ellipses", "rectangles", "vessels")


def _grid(size):
    return np.meshgrid(np.arange(size), np.arange(size), indexing="ij")


def _centre(rng, size):
    return rng.uniform(0.31, 0.69) * size, rng.uniform(0.31, 0.69) * size


def _rotated(X, Y, cx, cy, theta):
    c, s = np.cos(theta), np.sin(theta)
    return (X - cx) * c + (Y - cy) * s, -(X - cx) * s + (Y - cy) * c


def _discs(rng, size):
    X, Y = _grid(size)
    img = np.zeros((size, size))
    for _ in range(int(rng.integers(2, 4))):
        (cx, cy), r, a = _centre(rng, size), rng.uniform(0.05, 0.11) * size, rng.uniform(0.5, 1.0)
        img = np.maximum(img, a * ((X - cx) ** 2 + (Y - cy) ** 2 <= r * r))
    return img


def _ellipses(rng, size):
    X, Y = _grid(size)
    img = np.zeros((size, size))
    for _ in range(int(rng.integers(1, 4))):
        (cx, cy), a = _centre(rng, size), rng.uniform(0.5, 1.0)
        ra, rb = rng.uniform(0.06, 0.14) * size, rng.uniform(0.03, 0.08) * size
        u, v = _rotated(X, Y, cx, cy, rng.uniform(0, np.pi))
        img = np.maximum(img, a * ((u / ra) ** 2 + (v / rb) ** 2 <= 1.0))
    return img


def _rectangles(rng, size):
    X, Y = _grid(size)
    img = np.zeros((size, size))
    for _ in range(int(rng.integers(1, 4))):
        (cx, cy), a = _centre(rng, size), rng.uniform(0.5, 1.0)
        ha, hb = rng.uniform(0.05, 0.12) * size, rng.uniform(0.03, 0.08) * size
        u, v = _rotated(X, Y, cx, cy, rng.uniform(0, np.pi))
        img = np.maximum(img, a * ((np.abs(u) <= ha) & (np.abs(v) <= hb)))
    return img


def _vessels(rng, size, width_px=None):
    """`width_px` replaces the drawn profile width (standard deviation, in pixels) of every line; the
    random draws are the same either way, so the curves do not change."""
    X, Y = _grid(size)
    img = np.zeros((size, size))
    t = np.linspace(0.0, 1.0, 200)
    for _ in range(int(rng.integers(1, 3))):
        (x0, y0), (x1, y1) = _centre(rng, size), _centre(rng, size)
        length = max(np.hypot(x1 - x0, y1 - y0), 1e-6)
        nx, ny = -(y1 - y0) / length, (x1 - x0) / length              # unit normal of the chord
        bend = rng.uniform(-0.08, 0.08) * size * np.sin(np.pi * t * rng.uniform(1.0, 2.0))
        px, py = x0 + (x1 - x0) * t + nx * bend, y0 + (y1 - y0) * t + ny * bend
        d2 = ((X[..., None] - px) ** 2 + (Y[..., None] - py) ** 2).min(axis=-1)
        width, a = rng.uniform(0.015, 0.03) * size, rng.uniform(0.5, 1.0)
        if width_px is not None:
            width = float(width_px)
        img = np.maximum(img, a * np.exp(-d2 / (2 * width ** 2)))
    return img


_GENERATORS = {"discs": _discs, "ellipses": _ellipses, "rectangles": _rectangles, "vessels": _vessels}


def shape_phantom(family: str, size: int, seed: int) -> np.ndarray:
    """One phantom of the given family; deterministic in (family, size, seed)."""
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; choose one of {', '.join(FAMILIES)}")
    rng = np.random.default_rng(seed)
    if family == "blobs":
        return random_blob_phantom(size=size, seed=seed, n_blobs=int(rng.integers(1, 4)))
    img = _GENERATORS[family](rng, size)
    return (img / img.max()).astype(np.float32)


def vessel_phantom(size: int, seed: int, width_px: float) -> np.ndarray:
    """The vessel phantom of `seed` with every line given the profile width `width_px` (standard
    deviation in pixels; the family itself draws 0.96 to 1.92 px on a 64-point grid)."""
    img = _vessels(np.random.default_rng(seed), size, width_px)
    return (img / img.max()).astype(np.float32)
