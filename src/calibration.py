"""Amplitude calibration of time-reversal (TR) reconstructions.

Raw TR output is not on the [0, 1] scale of the phantoms that PSNR/SSIM are computed against
(data_range = 1.0). TR is linear in the sensor recording, so a single gain per sensor geometry
corrects its amplitude. The gain is fitted by least squares on TRAINING data only and then applied
unchanged to any other split or noise level; it must never be fitted on test ground truth.
"""

import numpy as np


def fit_scale(recon: np.ndarray, gt: np.ndarray) -> float:
    """Least-squares scalar a minimising ||a * recon - gt||^2 over all given pixels."""
    return float((recon * gt).sum() / (recon * recon).sum())


def fit_scales_per_sensor_count(recon: np.ndarray, gt: np.ndarray, n_sensors: np.ndarray) -> dict:
    """One least-squares scalar per distinct sensor count, keyed by int(n_sensors)."""
    return {int(k): fit_scale(recon[n_sensors == k], gt[n_sensors == k])
            for k in np.unique(n_sensors)}


def training_scales(train_npz_path: str = "data/train.npz") -> dict:
    """Per-sensor-count TR gains fitted on the training split (the U-Net's own training data)."""
    d = np.load(train_npz_path)
    return fit_scales_per_sensor_count(d["recon"], d["phantom"], d["n_sensors"])
