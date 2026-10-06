import numpy as np
import pytest

from src.image_statistics import FEATURES, ReferenceDistribution, feature_matrix, image_statistics
from src.shape_phantoms import shape_phantom, vessel_phantom

X, Y = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
R2 = (X - 32) ** 2 + (Y - 32) ** 2


def test_sharp_disc_against_smooth_blob():
    disc, blob = image_statistics((R2 <= 8 ** 2).astype(float)), image_statistics(np.exp(-R2 / (2 * 5.0 ** 2)))
    assert 0.5 <= disc["max_gradient"] <= 0.71 and blob["max_gradient"] < 0.15
    assert disc["hf_fraction"] > 10 * blob["hf_fraction"]
    assert disc["spectral_centroid"] > blob["spectral_centroid"]
    assert disc["tv_ratio"] > blob["tv_ratio"]
    assert disc["n_components"] == blob["n_components"] == 1.0


def test_thickness_area_and_components():
    disc = image_statistics((R2 <= 8 ** 2).astype(float))
    assert disc["thickness"] == pytest.approx(16.0, abs=2.0)
    assert disc["area_fraction"] == pytest.approx(np.pi * 64 / 64 ** 2, rel=0.05)
    two = ((X - 20) ** 2 + (Y - 20) ** 2 <= 16) | ((X - 44) ** 2 + (Y - 44) ** 2 <= 16)
    assert image_statistics(two.astype(float))["n_components"] == 2.0
    line = np.exp(-(X - 32.0) ** 2 / (2 * 1.5 ** 2)) * (np.abs(Y - 32) < 15)
    assert image_statistics(line)["thickness"] < 6.0


def test_vessel_width_controls_thickness_and_keeps_the_curve():
    thin, thick = vessel_phantom(64, 3, 1.0), vessel_phantom(64, 3, 4.0)
    assert image_statistics(thick)["thickness"] > 2 * image_statistics(thin)["thickness"]
    assert np.unravel_index(thin.argmax(), thin.shape) == pytest.approx(np.unravel_index(thick.argmax(), thick.shape), abs=6)
    # the family itself is unchanged by the optional width argument
    assert shape_phantom("vessels", 64, 3).max() == pytest.approx(1.0)


def test_distance_separates_families():
    blobs = feature_matrix([shape_phantom("blobs", 64, s) for s in range(40)])
    assert blobs.shape == (40, len(FEATURES))
    ref = ReferenceDistribution(blobs)
    own = ref.distance(feature_matrix([shape_phantom("blobs", 64, s) for s in range(100, 120)]))
    discs = ref.distance(feature_matrix([shape_phantom("discs", 64, s) for s in range(20)]))
    assert np.median(discs) > 3 * np.median(own)
    assert ref.distance(blobs.mean(axis=0))[0] == pytest.approx(0.0, abs=1e-9)
    assert np.isfinite(ReferenceDistribution(np.ones((5, len(FEATURES)))).distance(blobs)).all()  # degenerate reference
