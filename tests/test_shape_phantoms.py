import numpy as np
import pytest

from src.shape_phantoms import FAMILIES, shape_phantom


@pytest.mark.parametrize("family", FAMILIES)
def test_shape_range_and_reproducibility(family):
    a, b = shape_phantom(family, 64, 5), shape_phantom(family, 64, 5)
    assert a.shape == (64, 64) and a.dtype == np.float32
    assert a.min() >= 0.0 and a.max() == pytest.approx(1.0)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, shape_phantom(family, 64, 6))


@pytest.mark.parametrize("family", FAMILIES[1:])  # the blob family of src/phantoms.py has wider tails
def test_shapes_stay_inside_the_sensor_circle(family):
    X, Y = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    outside = (X - 32) ** 2 + (Y - 32) ** 2 >= 24 ** 2  # sensors sit on radius 24
    for seed in range(40):
        assert shape_phantom(family, 64, seed)[outside].max() < 0.05, seed


def test_sharp_families_are_piecewise_constant_and_smooth_ones_are_not():
    for family in ("discs", "ellipses", "rectangles"):
        assert len(np.unique(shape_phantom(family, 64, 3))) <= 4  # background plus at most 3 shapes
    for family in ("blobs", "vessels"):
        assert len(np.unique(shape_phantom(family, 64, 3))) > 100


def test_unknown_family_is_rejected():
    with pytest.raises(ValueError):
        shape_phantom("stars", 64, 0)
