"""Line phantoms whose structural width and edge-transition width are separate parameters.

The vessel family of src/shape_phantoms.py draws a line with a Gaussian cross-section, so one number
sets both how thick the line is and how sharp its edges are. Here the cross-section is a flat core of
full width w (pixels) convolved with a Gaussian of standard deviation s (pixels), as a function of
the distance d from the centreline and scaled to 1 on the centreline:

    p(d; w, s) = [erf((w/2 - d) / (sqrt(2) s)) + erf((w/2 + d) / (sqrt(2) s))] / [2 erf(w / (2 sqrt(2) s))]

with the two limits taken exactly:

    s = 0:  p = 1 for d <= w/2, else 0        a sharp-edged bar, like the sharp training families
    w = 0:  p = exp(-d^2 / (2 s^2))           the Gaussian line of the vessel family, of width s

The centreline is that of the vessel family: `geometry="curved"` reproduces its random draws exactly,
and `geometry="straight"` uses the same end points, amplitudes and number of lines with the bend set
to zero. A curved phantom with w = 0 is identical to shape_phantoms.vessel_phantom(size, seed, s).

What the two parameters can and cannot separate is stated in `profile_descriptors`: for w well above
s the profile has a flat top of width about w and edges of width s; for w below about 2 s it is close
to a Gaussian of variance s^2 + w^2/12, and w and s are no longer distinct properties of the image.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.special import erf

from src.shape_phantoms import _centre, _grid

GEOMETRIES = ("straight", "curved")
N_CENTRELINE_POINTS = 200   # as in the vessel family


def _profile(d2, core_width, edge_sigma):
    """p as a function of the squared distance from the centreline."""
    if core_width < 0 or edge_sigma < 0 or (core_width == 0 and edge_sigma == 0):
        raise ValueError("need core_width >= 0 and edge_sigma >= 0, not both zero")
    if edge_sigma == 0:
        return (d2 <= (core_width / 2.0) ** 2).astype(np.float64)
    if core_width == 0:
        return np.exp(-d2 / (2 * edge_sigma ** 2))
    d, h, s = np.sqrt(d2), core_width / 2.0, np.sqrt(2.0) * edge_sigma
    return (erf((h - d) / s) + erf((h + d) / s)) / (2.0 * erf(h / s))


def line_profile(d, core_width: float, edge_sigma: float) -> np.ndarray:
    """Cross-section p(d; w, s) at distance d (pixels) from the centreline; 1 at d = 0."""
    return _profile(np.asarray(d, np.float64) ** 2, float(core_width), float(edge_sigma))


def profile_descriptors(core_width: float, edge_sigma: float) -> dict:
    """Properties of the continuous cross-section (before sampling on the pixel grid):

      fwhm           full width at half maximum, in pixels
      plateau_width  full width over which p >= 0.95: the part that is flat
      max_slope      largest |dp/dd| per pixel (None for s = 0, where the grid sets it)
    """
    w, s = float(core_width), float(edge_sigma)
    if s == 0:
        return {"fwhm": w, "plateau_width": w, "max_slope": None}
    upper = w / 2.0 + 6.0 * s

    def crossing(level):
        return 2.0 * brentq(lambda d: float(line_profile(d, w, s)) - level, 0.0, upper)
    d = np.linspace(0.0, upper, 20001)
    return {"fwhm": crossing(0.5), "plateau_width": crossing(0.95),
            "max_slope": float(np.abs(np.gradient(line_profile(d, w, s), d)).max())}


def line_parameters(size: int, seed: int) -> list:
    """The random draws of the vessel family for `seed`, one dict per line (1 or 2 lines):
    end points, chord length, bend amplitude and frequency, amplitude. The family's own width draw
    is made and discarded, so that every other draw is the one the family makes."""
    rng = np.random.default_rng(seed)
    lines = []
    for _ in range(int(rng.integers(1, 3))):
        (x0, y0), (x1, y1) = _centre(rng, size), _centre(rng, size)
        bend_amplitude = rng.uniform(-0.08, 0.08) * size
        bend_frequency = rng.uniform(1.0, 2.0)
        rng.uniform(0.015, 0.03)
        lines.append({"start": (x0, y0), "end": (x1, y1), "chord_length": float(np.hypot(x1 - x0, y1 - y0)),
                      "bend_amplitude": float(bend_amplitude), "bend_frequency": float(bend_frequency),
                      "amplitude": float(rng.uniform(0.5, 1.0))})
    return lines


def line_phantom(size: int, seed: int, core_width: float, edge_sigma: float, geometry: str) -> np.ndarray:
    """(size, size) float32 phantom with peak 1; deterministic in all arguments."""
    if geometry not in GEOMETRIES:
        raise ValueError(f"unknown geometry {geometry!r}; choose one of {', '.join(GEOMETRIES)}")
    X, Y = _grid(size)
    img = np.zeros((size, size))
    t = np.linspace(0.0, 1.0, N_CENTRELINE_POINTS)
    for line in line_parameters(size, seed):
        (x0, y0), (x1, y1) = line["start"], line["end"]
        length = max(np.hypot(x1 - x0, y1 - y0), 1e-6)
        nx, ny = -(y1 - y0) / length, (x1 - x0) / length              # unit normal of the chord
        bend = line["bend_amplitude"] * np.sin(np.pi * t * line["bend_frequency"]) if geometry == "curved" else 0.0
        px, py = x0 + (x1 - x0) * t + nx * bend, y0 + (y1 - y0) * t + ny * bend
        d2 = ((X[..., None] - px) ** 2 + (Y[..., None] - py) ** 2).min(axis=-1)
        img = np.maximum(img, line["amplitude"] * _profile(d2, float(core_width), float(edge_sigma)))
    return (img / img.max()).astype(np.float32)
