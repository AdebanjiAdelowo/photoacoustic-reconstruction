"""Amplitude-calibrated time-reversal baseline.

Raw time-reversal (TR) reconstructions are not on the [0, 1] scale of the phantoms that PSNR/SSIM
use (data_range = 1.0), so part of the TR-vs-U-Net PSNR gap is amplitude calibration. This script
fits a scalar a minimising ||a * TR - ground truth||^2 on the TRAINING split only (the data the U-Net
was trained on), once over all training images and once per sensor count, and applies it unchanged
to the test split. No test-set information is used. A per-image scale fitted against each test
ground truth is also reported, labelled as an oracle upper bound, not as a baseline.

Usage: python scripts/evaluate_calibrated_baseline.py
Requires data/{train,val,test}.npz from scripts/generate_training_data.py. Does not use the U-Net.
Writes report/calibrated_baseline_results.txt
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate import psnr, ssim

SPARSITIES = (16, 64)


def fit_scale(recon: np.ndarray, gt: np.ndarray) -> float:
    """Least-squares scalar a minimising ||a * recon - gt||^2."""
    return float((recon * gt).sum() / (recon * recon).sum())


def main():
    train, val, test = (np.load(f"data/{s}.npz") for s in ("train", "val", "test"))
    a_global = fit_scale(train["recon"], train["phantom"])
    a_sparsity = {k: fit_scale(train["recon"][train["n_sensors"] == k],
                               train["phantom"][train["n_sensors"] == k]) for k in SPARSITIES}
    a_val = fit_scale(val["recon"], val["phantom"])

    lines = [
        "Amplitude-calibrated time-reversal baseline (test split; scale fitted on the training split only)",
        f"train n={len(train['phantom'])}, test n={len(test['phantom'])}",
        f"scale fitted on train: global a={a_global:.3f}; per sensor count "
        + ", ".join(f"{k}: a={a_sparsity[k]:.3f}" for k in SPARSITIES),
        f"(for reference, a global scale fitted on the validation split instead: a={a_val:.3f})",
        "",
        f"{'sensors':<9}{'n':<4}{'variant':<34}{'PSNR':>8}{'SSIM':>8}",
    ]
    for k in SPARSITIES:
        m = test["n_sensors"] == k
        gts, recs = test["phantom"][m], test["recon"][m]
        variants = [
            ("raw TR (as in mvp_results.txt)", lambda r, g: r),
            ("TR x global train scale", lambda r, g: a_global * r),
            ("TR x per-sensor-count train scale", lambda r, g: a_sparsity[k] * r),
            ("oracle: per-image scale (uses GT)", lambda r, g: fit_scale(r, g) * r),
        ]
        for name, f in variants:
            ps = np.mean([psnr(f(r, g), g) for r, g in zip(recs, gts)])
            ss = np.mean([ssim(f(r, g), g) for r, g in zip(recs, gts)])
            lines.append(f"{k:<9}{int(m.sum()):<4}{name:<34}{ps:>8.2f}{ss:>8.3f}")
    os.makedirs("report", exist_ok=True)
    with open("report/calibrated_baseline_results.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nSaved report/calibrated_baseline_results.txt")


if __name__ == "__main__":
    main()
