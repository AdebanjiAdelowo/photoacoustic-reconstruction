"""Structural statistics of an image and a distance from a reference set of images.

Used to quantify how far a test phantom is from the images a network was trained on. All statistics
are computed on the ground-truth image (values in [0, 1], peak 1) and are simple by design:

  tv_ratio            total variation divided by the image sum: edge content per unit of signal
  max_gradient        largest gradient magnitude: 0.5 to 0.71 at a one-pixel step, small for smooth images
  hf_fraction         share of spectral power (mean removed) above 1/8 cycle per pixel
  spectral_centroid   power-weighted mean radial spatial frequency, in cycles per pixel
  area_fraction       share of pixels above half maximum
  thickness           width of the widest structure: twice the largest inscribed radius of the
                      half-maximum support, in pixels
  n_components        number of connected components of the half-maximum support
"""

import numpy as np
from scipy import ndimage

FEATURES = ("tv_ratio", "max_gradient", "hf_fraction", "spectral_centroid", "area_fraction", "thickness",
            "n_components")
HF_CUTOFF = 0.125  # cycles per pixel (a quarter of the Nyquist frequency)


def image_statistics(image: np.ndarray) -> dict:
    img = np.asarray(image, np.float64)
    gx, gy = np.gradient(img)
    grad = np.hypot(gx, gy)
    power = np.abs(np.fft.fft2(img - img.mean())) ** 2
    fx, fy = np.meshgrid(np.fft.fftfreq(img.shape[0]), np.fft.fftfreq(img.shape[1]), indexing="ij")
    radius = np.hypot(fx, fy)
    total = power.sum()
    mask = img > 0.5
    return {
        "tv_ratio": float(grad.sum() / img.sum()),
        "max_gradient": float(grad.max()),
        "hf_fraction": float(power[radius > HF_CUTOFF].sum() / total),
        "spectral_centroid": float((power * radius).sum() / total),
        "area_fraction": float(mask.mean()),
        "thickness": float(2.0 * ndimage.distance_transform_edt(mask).max()),
        "n_components": float(ndimage.label(mask)[1]),
    }


def feature_matrix(images) -> np.ndarray:
    """(N, len(FEATURES)) array, columns in the order of FEATURES."""
    return np.array([[image_statistics(img)[f] for f in FEATURES] for img in images])


class ReferenceDistribution:
    """Mahalanobis distance from the feature distribution of a reference set of images.

    Features are standardised by the reference mean and standard deviation; the covariance of the
    standardised features is regularised (ridge) so that a small or degenerate reference set, such as
    one family in which every image has the same number of components, stays invertible.
    """

    def __init__(self, reference_features: np.ndarray, ridge: float = 0.05):
        x = np.asarray(reference_features, np.float64)
        self.mean = x.mean(axis=0)
        self.scale = np.where(x.std(axis=0) > 1e-12, x.std(axis=0), 1.0)
        z = (x - self.mean) / self.scale
        cov = np.cov(z, rowvar=False) + ridge * np.eye(x.shape[1])
        self.precision = np.linalg.inv(cov)

    def distance(self, features: np.ndarray) -> np.ndarray:
        z = (np.atleast_2d(features) - self.mean) / self.scale
        return np.sqrt(np.einsum("ij,jk,ik->i", z, self.precision, z))
