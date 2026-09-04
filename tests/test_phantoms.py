import numpy as np

from src.phantoms import random_blob_phantom


def test_shape_and_dtype():
    p = random_blob_phantom(size=64, seed=0)
    assert p.shape == (64, 64)
    assert p.dtype == np.float32


def test_value_range():
    p = random_blob_phantom(size=64, seed=0)
    assert p.min() >= 0.0
    assert p.max() <= 1.0 + 1e-6
    assert p.max() > 0.5  # peak is normalised to 1.0, so max should be near it


def test_reproducibility():
    p1 = random_blob_phantom(size=64, seed=42)
    p2 = random_blob_phantom(size=64, seed=42)
    np.testing.assert_array_equal(p1, p2)


def test_different_seeds_differ():
    p1 = random_blob_phantom(size=64, seed=1)
    p2 = random_blob_phantom(size=64, seed=2)
    assert not np.array_equal(p1, p2)


def test_n_blobs_affects_output():
    p1 = random_blob_phantom(size=64, seed=0, n_blobs=1)
    p2 = random_blob_phantom(size=64, seed=0, n_blobs=5)
    assert not np.array_equal(p1, p2)
