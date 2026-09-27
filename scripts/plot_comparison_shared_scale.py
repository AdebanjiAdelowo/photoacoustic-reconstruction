"""Comparison figure on a common display range: ground truth, time-reversal, learned refinement and
the absolute error of both reconstructions, for the first four test examples.

Same data, checkpoint and inference as scripts/evaluate_mvp.py; only the display differs. All
intensity panels use the fixed [0, 1] range that PSNR/SSIM are computed against (data_range = 1.0 in
src/evaluate.py), so values outside [0, 1] saturate instead of being rescaled per panel, and both
error rows share one colour scale.

Usage: python scripts/plot_comparison_shared_scale.py   ->   report/comparison_shared_scale.png
Requires data/test.npz and experiments/unet_checkpoint.pt (generate_training_data.py, train.py).
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate import psnr
from src.reconstruction_net import ReconstructionUNet


def main():
    d = np.load("data/test.npz")
    phantoms, recons_tr, n_sensors = d["phantom"], d["recon"], d["n_sensors"]

    checkpoint = torch.load("experiments/unet_checkpoint.pt", map_location="cpu", weights_only=True)
    model = ReconstructionUNet(base_features=16)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        recons_learned = model(torch.from_numpy(recons_tr).unsqueeze(1).float()).squeeze(1).numpy()

    n_show = min(4, len(phantoms))
    err_tr = np.abs(recons_tr[:n_show] - phantoms[:n_show])
    err_learned = np.abs(recons_learned[:n_show] - phantoms[:n_show])
    err_max = float(np.percentile(np.concatenate([err_tr.ravel(), err_learned.ravel()]), 99.5))

    rows = ["ground truth", "time-reversal", "learned refinement",
            "|time-reversal - truth|", "|learned - truth|"]
    fig, axes = plt.subplots(5, n_show, figsize=(2.5 * n_show + 1.2, 12.6))
    for i in range(n_show):
        gt = phantoms[i]
        panels = [
            (gt, f"{n_sensors[i]} sensors", "inferno", 0, 1),
            (recons_tr[i], f"PSNR {psnr(recons_tr[i], gt):.2f} dB", "inferno", 0, 1),
            (recons_learned[i], f"PSNR {psnr(recons_learned[i], gt):.2f} dB", "inferno", 0, 1),
            (err_tr[i], "", "magma", 0, err_max),
            (err_learned[i], "", "magma", 0, err_max),
        ]
        for r, (img, title, cmap, lo, hi) in enumerate(panels):
            im = axes[r, i].imshow(img, cmap=cmap, vmin=lo, vmax=hi)
            axes[r, i].set_xticks([])
            axes[r, i].set_yticks([])
            if title:
                axes[r, i].set_title(title, fontsize=9)
            if i == 0:
                axes[r, i].set_ylabel(rows[r], fontsize=9)
            if i == n_show - 1:
                fig.colorbar(im, ax=axes[r, :].tolist(), fraction=0.025, pad=0.02)
    fig.suptitle("Held-out test examples 0-3, common display range [0, 1]; error rows share one scale",
                 fontsize=10, y=0.93)
    os.makedirs("report", exist_ok=True)
    fig.savefig("report/comparison_shared_scale.png", dpi=110, bbox_inches="tight")
    print(f"checkpoint epoch {checkpoint['epoch']}; error colour limit {err_max:.3f}")
    for i in range(n_show):
        print(f"example {i}: n_sensors={n_sensors[i]} TR {psnr(recons_tr[i], phantoms[i]):.2f} dB, "
              f"learned {psnr(recons_learned[i], phantoms[i]):.2f} dB, "
              f"TR range [{recons_tr[i].min():.2f}, {recons_tr[i].max():.2f}], "
              f"learned range [{recons_learned[i].min():.2f}, {recons_learned[i].max():.2f}]")
    print("Saved report/comparison_shared_scale.png")


if __name__ == "__main__":
    main()
