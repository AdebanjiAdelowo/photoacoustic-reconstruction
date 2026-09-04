"""Stage 7 — MVP evaluation: time-reversal baseline vs. learned refinement, on the held-out test
split, at both MVP sparsity settings. Produces genuine metrics and a real comparison figure —
nothing here is invented or adjusted by hand.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate import psnr, ssim
from src.reconstruction_net import ReconstructionUNet


def main():
    d = np.load("data/test.npz")
    phantoms = d["phantom"]
    recons_tr = d["recon"]  # time-reversal baseline, already computed in Stage 5
    n_sensors = d["n_sensors"]

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    checkpoint = torch.load("experiments/unet_checkpoint.pt", map_location=device, weights_only=True)
    model = ReconstructionUNet(base_features=16).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    x = torch.from_numpy(recons_tr).unsqueeze(1).float().to(device)
    with torch.no_grad():
        recons_learned = model(x).squeeze(1).cpu().numpy()

    print(f"Checkpoint from epoch {checkpoint['epoch']}, val_loss {checkpoint['val_loss']:.6f}")
    print(f"Test set: {len(phantoms)} examples, n_sensors distribution: "
          f"{dict(zip(*np.unique(n_sensors, return_counts=True)))}")
    print()

    results = {16: {"tr_psnr": [], "tr_ssim": [], "learned_psnr": [], "learned_ssim": []},
               64: {"tr_psnr": [], "tr_ssim": [], "learned_psnr": [], "learned_ssim": []}}

    for i in range(len(phantoms)):
        gt = phantoms[i]
        tr = recons_tr[i]
        learned = recons_learned[i]
        k = int(n_sensors[i])

        results[k]["tr_psnr"].append(psnr(tr, gt))
        results[k]["tr_ssim"].append(ssim(tr, gt))
        results[k]["learned_psnr"].append(psnr(learned, gt))
        results[k]["learned_ssim"].append(ssim(learned, gt))

    print(f"{'sparsity':<10}{'n':<4}{'TR PSNR':<12}{'TR SSIM':<12}{'Learned PSNR':<14}{'Learned SSIM':<14}")
    summary_rows = []
    for k in [16, 64]:
        n = len(results[k]["tr_psnr"])
        if n == 0:
            print(f"{k:<10}{n:<4} (no test examples at this sparsity)")
            continue
        row = {
            "sparsity": k, "n": n,
            "tr_psnr": np.mean(results[k]["tr_psnr"]), "tr_ssim": np.mean(results[k]["tr_ssim"]),
            "learned_psnr": np.mean(results[k]["learned_psnr"]), "learned_ssim": np.mean(results[k]["learned_ssim"]),
        }
        summary_rows.append(row)
        print(f"{k:<10}{n:<4}{row['tr_psnr']:<12.3f}{row['tr_ssim']:<12.4f}"
              f"{row['learned_psnr']:<14.3f}{row['learned_ssim']:<14.4f}")

    # Save genuine numeric results, not just print them.
    os.makedirs("report", exist_ok=True)
    with open("report/mvp_results.txt", "w") as f:
        f.write("MVP Evaluation Results (real, measured — see IMPLEMENTATION_LOG.md Stage 7)\n")
        f.write(f"Checkpoint: epoch {checkpoint['epoch']}, val_loss {checkpoint['val_loss']:.6f}\n")
        f.write(f"Test set size: {len(phantoms)}\n\n")
        f.write(f"{'sparsity':<10}{'n':<4}{'TR PSNR':<12}{'TR SSIM':<12}{'Learned PSNR':<14}{'Learned SSIM':<14}\n")
        for row in summary_rows:
            f.write(f"{row['sparsity']:<10}{row['n']:<4}{row['tr_psnr']:<12.3f}{row['tr_ssim']:<12.4f}"
                    f"{row['learned_psnr']:<14.3f}{row['learned_ssim']:<14.4f}\n")

    # Genuine qualitative comparison figure — first 4 test examples.
    n_show = min(4, len(phantoms))
    fig, axes = plt.subplots(3, n_show, figsize=(3 * n_show, 9))
    for i in range(n_show):
        axes[0, i].imshow(phantoms[i], cmap="inferno", vmin=0, vmax=1)
        axes[0, i].set_title(f"ground truth\n(n_sensors={n_sensors[i]})")
        axes[0, i].axis("off")
        axes[1, i].imshow(recons_tr[i], cmap="inferno")
        axes[1, i].set_title(f"time-reversal\nPSNR={psnr(recons_tr[i], phantoms[i]):.2f}")
        axes[1, i].axis("off")
        axes[2, i].imshow(recons_learned[i], cmap="inferno")
        axes[2, i].set_title(f"learned refinement\nPSNR={psnr(recons_learned[i], phantoms[i]):.2f}")
        axes[2, i].axis("off")
    plt.tight_layout()
    plt.savefig("report/mvp_comparison.png", dpi=120)
    print("\nSaved report/mvp_results.txt and report/mvp_comparison.png")


if __name__ == "__main__":
    main()
