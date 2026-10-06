"""Line phantoms with separate core width and edge width: the profile, its limits and the geometry controls."""
import numpy as np
import pytest

from src.image_statistics import image_statistics
from src.shape_phantoms import vessel_phantom
from src.thin_structures import GEOMETRIES, line_parameters, line_phantom, line_profile, profile_descriptors

SIZE = 64
SEEDS = range(600_000, 600_012)


def test_profile_is_one_on_the_centreline_and_has_the_two_exact_limits():
    d = np.linspace(0.0, 8.0, 81)
    for w, s in ((0, 1.0), (1, 0.5), (2, 1.0), (6, 2.0), (4, 0)):
        p = line_profile(d, w, s)
        assert p[0] == pytest.approx(1.0) and np.all(np.diff(p) <= 1e-12) and p.min() >= 0.0
    assert np.array_equal(line_profile(d, 4, 0), (d <= 2.0).astype(float))              # sharp bar
    assert np.allclose(line_profile(d, 0, 1.5), np.exp(-d ** 2 / (2 * 1.5 ** 2)))        # Gaussian line
    assert np.allclose(line_profile(d, 1e-4, 1.5), line_profile(d, 0, 1.5), atol=1e-6)  # the limit is continuous
    with pytest.raises(ValueError):
        line_profile(d, 0, 0)


def test_core_width_and_edge_width_are_independent_for_thick_cores():
    """For w >= 4 and s <= 1 the width at half maximum is the core width and the slope is set by s."""
    for w in (4, 6):
        for s in (0.5, 1.0):
            assert profile_descriptors(w, s)["fwhm"] == pytest.approx(w, abs=0.15)
    for s in (0.5, 1.0):
        slopes = [profile_descriptors(w, s)["max_slope"] for w in (4, 6)]
        assert slopes[0] == pytest.approx(slopes[1], rel=0.05)
        assert slopes[1] == pytest.approx(1.0 / (s * np.sqrt(2 * np.pi)), rel=0.01)      # an isolated blurred step
    assert profile_descriptors(6, 0.5)["plateau_width"] > 4.0 > profile_descriptors(6, 1.5)["plateau_width"]


def test_stated_limit_thin_cores_are_close_to_a_gaussian_and_do_not_separate_the_parameters():
    """For w <= 1 and s >= 1 the profile is a Gaussian of variance s^2 + w^2/12 to within 2 %: here
    the core width is not a separate property of the image, as the pre-registration states."""
    for s in (1.0, 1.5, 2.0):
        d = profile_descriptors(1, s)
        assert d["fwhm"] == pytest.approx(2.3548 * np.sqrt(s ** 2 + 1 / 12), rel=0.02)
        assert d["fwhm"] == pytest.approx(profile_descriptors(0, s)["fwhm"], rel=0.05)
    assert profile_descriptors(0, 1.0)["max_slope"] == pytest.approx(np.exp(-0.5), rel=0.01)   # steeper than a blurred step


def test_factors_vary_separately_on_the_pixel_grid():
    def measured(w, s, name):
        return np.mean([image_statistics(line_phantom(SIZE, seed, w, s, "straight"))[name] for seed in SEEDS])
    thickness = [measured(w, 1.0, "thickness") for w in (2, 4, 6)]
    assert thickness[0] + 0.8 < thickness[1] and thickness[1] + 1.5 < thickness[2]       # width varies at fixed edge
    gradient = [measured(6, s, "max_gradient") for s in (0.5, 1.0, 2.0)]
    assert gradient[0] > gradient[1] + 0.15 and gradient[1] > gradient[2] + 0.1           # edge varies at fixed width
    assert abs(measured(6, 0.5, "thickness") - measured(6, 1.0, "thickness")) < 0.5
    # the grid cannot show an edge much sharper than one pixel: s = 0.5 is close to a sharp edge
    assert measured(6, 0, "max_gradient") - measured(6, 0.5, "max_gradient") < 0.2


def test_curved_gaussian_line_is_the_vessel_family():
    for seed in (400_000, 400_007, 600_003):
        for s in (1.0, 1.5, 2.0):
            assert np.array_equal(line_phantom(SIZE, seed, 0, s, "curved"), vessel_phantom(SIZE, seed, s))


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_phantom_conventions_and_determinism(geometry):
    for w, s in ((1, 0), (0, 0.5), (2, 1.0), (6, 2.0)):
        for seed in SEEDS:
            p = line_phantom(SIZE, seed, w, s, geometry)
            assert p.shape == (SIZE, SIZE) and p.dtype == np.float32
            assert p.max() == pytest.approx(1.0) and p.min() >= 0.0 and p.sum() > 0
            assert np.array_equal(p, line_phantom(SIZE, seed, w, s, geometry))
    assert not np.array_equal(line_phantom(SIZE, 600_000, 2, 1.0, geometry), line_phantom(SIZE, 600_001, 2, 1.0, geometry))
    with pytest.raises(ValueError):
        line_phantom(SIZE, 0, 2, 1.0, "ring")


def test_straight_and_curved_share_end_points_amplitudes_and_profile():
    differ = 0
    for seed in SEEDS:
        lines = line_parameters(SIZE, seed)
        assert 1 <= len(lines) <= 2 and all(0.5 <= line["amplitude"] <= 1.0 for line in lines)
        straight, curved = (line_phantom(SIZE, seed, 0, 1.0, g) for g in GEOMETRIES)
        differ += not np.array_equal(straight, curved)
        for line in lines:  # both pass through the common end points, where the bend is zero at t = 0
            x, y = (int(round(v)) for v in line["start"])
            assert straight[x, y] > 0.3 * line["amplitude"] and curved[x, y] > 0.3 * line["amplitude"]
    assert differ == len(SEEDS)
    for w, s in ((0, 1.0), (2, 1.0), (6, 1.0)):   # the cross-section is the same: equal edge steepness
        g = [np.mean([image_statistics(line_phantom(SIZE, seed, w, s, geo))["max_gradient"] for seed in SEEDS]) for geo in GEOMETRIES]
        assert g[0] == pytest.approx(g[1], abs=0.03)


def test_a_single_straight_line_lies_on_its_chord():
    seed = next(s for s in range(600_000, 600_050) if len(line_parameters(SIZE, s)) == 1)
    (x0, y0), (x1, y1) = (line_parameters(SIZE, seed)[0][k] for k in ("start", "end"))
    X, Y = np.meshgrid(np.arange(SIZE), np.arange(SIZE), indexing="ij")
    t = np.clip(((X - x0) * (x1 - x0) + (Y - y0) * (y1 - y0)) / ((x1 - x0) ** 2 + (y1 - y0) ** 2), 0, 1)
    distance = np.hypot(X - (x0 + t * (x1 - x0)), Y - (y0 + t * (y1 - y0)))
    for w, s in ((1, 0), (4, 0), (2, 1.0)):
        support = line_phantom(SIZE, seed, w, s, "straight") > 0.5
        assert support.any() and distance[support].max() <= profile_descriptors(w, s)["fwhm"] / 2 + 0.05
