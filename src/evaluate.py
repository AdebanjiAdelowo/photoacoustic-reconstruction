"""Evaluation: PSNR/SSIM against ground truth, via scikit-image's verified implementations.

Implemented for the MVP (Stage 7).
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(reconstruction: np.ndarray, ground_truth: np.ndarray) -> float:
    """Peak signal-to-noise ratio, via skimage.metrics.peak_signal_noise_ratio.
    data_range is fixed to 1.0, matching the [0,1] phantom/reconstruction convention documented
    in src/phantoms.py (reconstructions are not guaranteed to stay exactly in [0,1], but the
    convention they are compared against is).
    """
    return float(peak_signal_noise_ratio(ground_truth, reconstruction, data_range=1.0))


def ssim(reconstruction: np.ndarray, ground_truth: np.ndarray) -> float:
    """Structural similarity index, via skimage.metrics.structural_similarity."""
    return float(structural_similarity(ground_truth, reconstruction, data_range=1.0))
