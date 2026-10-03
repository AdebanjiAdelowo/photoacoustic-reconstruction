"""Noise-robustness sensitivity analysis.

Motivation: the forward simulation (src/forward_model.py) has no noise model at all in the
reproducibility path used by scripts/generate_training_data.py and scripts/evaluate_mvp.py — the
simulated sensor recordings are noiseless. Real photoacoustic acquisitions are noise-dominated, so
this script asks a narrower, scoped question: using the EXISTING trained checkpoint (no
retraining), the EXISTING forward model (unmodified physics), and the EXISTING evaluation code
(src/evaluate.py), how does time-reversal and U-Net-refined reconstruction quality degrade if
i.i.d. Gaussian noise is added to the simulated sensor recordings at evaluation time?

This is deliberately NOT a new noise-aware training run, a new physical noise model, or a
posterior-sampling method — see README.md "Noise robustness" for the scope note.

Noise model: for each test phantom, the clean sensor recording is simulated exactly as in the
existing pipeline (simulate_sensor_data, unchanged). i.i.d. Gaussian noise is then added directly
to that recording, with the noise standard deviation set as a fraction of that example's own clean
recording RMS amplitude (so "5%" means the same relative corruption regardless of a given
phantom's absolute signal scale, since the reference sound speed/density are fixed but phantom
amplitude varies). relative_std=0.0 reproduces the noiseless recording bit-for-bit, so the existing
scripts/evaluate_mvp.py numbers are exactly the "noiseless" row here (verified below).

Dataset: the full existing test split (data/test.npz, 8 examples: 4 at 16 sensors, 4 at 64
sensors). 8 examples is already within (below) the 10-20-example "representative subset" this kind
of check calls for, so no further subsampling is applied -- the full test split is used at every
noise level.

Calibrated time-reversal: TR is linear in the recording, so TR(signal + noise) = TR(signal) +
TR(noise), and its amplitude gain does not depend on the noise. Each noisy TR reconstruction is
therefore also scored after multiplying by the per-sensor-count gain fitted on the noiseless
TRAINING split (src/calibration.py, the same gains as scripts/evaluate_calibrated_baseline.py). No
gain is fitted on noisy data or on test ground truth. The U-Net input is the raw TR reconstruction,
as in training; calibration affects only the TR baseline row.

Outputs: report/noise_sensitivity_results.txt (human-readable table) and
report/noise_sensitivity_results.json (machine-readable), plus console output.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baselines import time_reversal_reconstruction
from src.calibration import training_scales
from src.device import DEVICE_CHOICES, resolve_device, to_numpy
from src.evaluate import psnr, ssim
from src.forward_model import build_domain_and_medium, simulate_sensor_data, sparse_view_sensor_array
from src.reconstruction_net import ReconstructionUNet
from src.run_safety import ensure_writable

GRID_SIZE = 64
CENTRE = (GRID_SIZE // 2, GRID_SIZE // 2)
RADIUS = 24  # must match scripts/generate_training_data.py -- same sensor geometry the checkpoint
             # was trained on

# Noise levels: additive i.i.d. Gaussian noise on the simulated sensor recording, expressed as a
# fraction of that example's own clean-recording RMS amplitude. 0.0 is the existing noiseless
# pipeline (reproduces scripts/evaluate_mvp.py exactly); the other three are intended to span
# small / moderate / large relative to the clean signal.
NOISE_LEVELS = [
    ("noiseless", 0.00),
    ("low", 0.01),
    ("moderate", 0.05),
    ("high", 0.20),
    ("severe", 0.50),
]

NOISE_SEED = 12345  # fixed -- noise draws are reproducible across reruns


def add_sensor_noise(recording: np.ndarray, relative_std: float, rng: np.random.Generator) -> np.ndarray:
    """Add i.i.d. Gaussian noise to a (Nt, n_sensors, 1) sensor recording, sized relative to the
    recording's own RMS amplitude. relative_std=0.0 returns the recording completely unchanged
    (not just "with zero noise added"), so the noiseless path is bit-for-bit identical to the
    existing pipeline.
    """
    if relative_std == 0.0:
        return recording
    sigma = relative_std * float(np.sqrt(np.mean(recording ** 2)))
    noise = rng.normal(0.0, sigma, size=recording.shape).astype(recording.dtype)
    return recording + noise


def main(device_choice="auto", checkpoint_path="experiments/unet_checkpoint.pt", data_dir="data",
         out_dir="report", overwrite=False):
    device = resolve_device(device_choice)
    json_path = os.path.join(out_dir, "noise_sensitivity_results.json")
    txt = os.path.join(out_dir, "noise_sensitivity_results.txt")
    for path in (json_path, txt):
        ensure_writable(path, device=device, overwrite=overwrite)

    d = np.load(os.path.join(data_dir, "test.npz"))
    phantoms = d["phantom"]
    n_sensors_arr = d["n_sensors"]
    n_examples = len(phantoms)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = ReconstructionUNet(base_features=16).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    tr_gain = training_scales(os.path.join(data_dir, "train.npz"))  # per sensor count, noiseless training split only
    gains = np.array([tr_gain[int(k)] for k in n_sensors_arr])

    domain, medium = build_domain_and_medium(GRID_SIZE)
    # Sensor positions depend only on n_sensors (not on the individual phantom), so build once per
    # sparsity setting present in the test split.
    sensor_cache = {int(k): sparse_view_sensor_array(int(k), RADIUS, CENTRE)
                     for k in np.unique(n_sensors_arr)}

    print(f"Checkpoint from epoch {checkpoint['epoch']}, val_loss {checkpoint['val_loss']:.6f}")
    print(f"Test set: {n_examples} examples (full existing test split -- no retraining, no "
          f"subsampling; already below the 10-20-example guideline for this kind of check)")
    print()

    # Simulate each phantom's CLEAN recording exactly once (the forward physics is unchanged and
    # deterministic), then reuse it to build every noise level -- this isolates the noise variable
    # exactly instead of resimulating the (identical) clean signal redundantly.
    clean_recordings = []
    sensor_positions_per_example = []
    time_axes = []
    for i in range(n_examples):
        n_sensors = int(n_sensors_arr[i])
        sensor_pos = sensor_cache[n_sensors]
        recording, time_axis = simulate_sensor_data(phantoms[i], domain, medium, sensor_pos)
        clean_recordings.append(recording)
        sensor_positions_per_example.append(sensor_pos)
        time_axes.append(time_axis)

    all_results = {label: None for label, _ in NOISE_LEVELS}
    per_level_rows = []

    header = (f"{'noise':<12}{'rel.std':<10}{'SNR(dB)':<10}{'n':<4}{'TR PSNR':<10}{'TR SSIM':<10}"
              f"{'UNet PSNR':<11}{'UNet SSIM':<11}{'TRcal PSNR':<12}{'TRcal SSIM':<12}")
    print(header)

    for label, rel_std in NOISE_LEVELS:
        rng = np.random.default_rng(NOISE_SEED + int(round(rel_std * 100000)))
        recons_tr = np.zeros_like(phantoms)

        for i in range(n_examples):
            noisy_recording = add_sensor_noise(clean_recordings[i], rel_std, rng)
            tr = time_reversal_reconstruction(
                noisy_recording, sensor_positions_per_example[i], domain, medium, time_axes[i])
            recons_tr[i] = tr

        x = torch.from_numpy(recons_tr).unsqueeze(1).float().to(device)
        with torch.no_grad():
            recons_unet = to_numpy(model(x).squeeze(1))

        tr_psnrs = [psnr(recons_tr[i], phantoms[i]) for i in range(n_examples)]
        tr_ssims = [ssim(recons_tr[i], phantoms[i]) for i in range(n_examples)]
        unet_psnrs = [psnr(recons_unet[i], phantoms[i]) for i in range(n_examples)]
        unet_ssims = [ssim(recons_unet[i], phantoms[i]) for i in range(n_examples)]
        recons_trcal = recons_tr * gains[:, None, None]
        trcal_psnrs = [psnr(recons_trcal[i], phantoms[i]) for i in range(n_examples)]
        trcal_ssims = [ssim(recons_trcal[i], phantoms[i]) for i in range(n_examples)]

        # None (not float("inf")) for the noiseless row, so the JSON output stays strictly valid
        # JSON (RFC 8259 has no Infinity literal) rather than relying on Python json's non-standard
        # allow_nan extension.
        snr_db = None if rel_std == 0.0 else 20.0 * np.log10(1.0 / rel_std)

        # Breakdown by sparsity setting, matching scripts/evaluate_mvp.py's convention.
        by_sparsity = {}
        for k in sorted(np.unique(n_sensors_arr).tolist()):
            idx = [i for i in range(n_examples) if int(n_sensors_arr[i]) == k]
            by_sparsity[int(k)] = {
                "n": len(idx),
                "tr_psnr": float(np.mean([tr_psnrs[i] for i in idx])),
                "tr_ssim": float(np.mean([tr_ssims[i] for i in idx])),
                "unet_psnr": float(np.mean([unet_psnrs[i] for i in idx])),
                "unet_ssim": float(np.mean([unet_ssims[i] for i in idx])),
                "trcal_psnr": float(np.mean([trcal_psnrs[i] for i in idx])),
                "trcal_ssim": float(np.mean([trcal_ssims[i] for i in idx])),
            }

        row = {
            "label": label,
            "relative_std": rel_std,
            "snr_db": snr_db,
            "n": n_examples,
            "tr_psnr": float(np.mean(tr_psnrs)),
            "tr_ssim": float(np.mean(tr_ssims)),
            "unet_psnr": float(np.mean(unet_psnrs)),
            "unet_ssim": float(np.mean(unet_ssims)),
            "trcal_psnr": float(np.mean(trcal_psnrs)),
            "trcal_ssim": float(np.mean(trcal_ssims)),
            "by_sparsity": by_sparsity,
        }
        all_results[label] = row
        per_level_rows.append(row)

        snr_str = "inf" if rel_std == 0.0 else f"{snr_db:.1f}"
        print(f"{label:<12}{rel_std:<10.2f}{snr_str:<10}{row['n']:<4}{row['tr_psnr']:<10.3f}"
              f"{row['tr_ssim']:<10.4f}{row['unet_psnr']:<11.3f}{row['unet_ssim']:<11.4f}"
              f"{row['trcal_psnr']:<12.3f}{row['trcal_ssim']:<12.4f}")

    # Sanity check: the noiseless row here must match scripts/evaluate_mvp.py's already-verified
    # numbers (report/mvp_results.txt), since relative_std=0.0 leaves the recording untouched.
    noiseless = all_results["noiseless"]
    print()
    print(f"Sanity check -- noiseless row (n={noiseless['n']}, overall): "
          f"TR PSNR={noiseless['tr_psnr']:.3f} TR SSIM={noiseless['tr_ssim']:.4f} "
          f"UNet PSNR={noiseless['unet_psnr']:.3f} UNet SSIM={noiseless['unet_ssim']:.4f}")
    print("(compare per-sparsity rows against report/mvp_results.txt: 16-sensor and 64-sensor "
          "TR/UNet PSNR/SSIM should match exactly)")

    os.makedirs(out_dir, exist_ok=True)

    with open(json_path, "w") as f:
        json.dump({
            "checkpoint_epoch": checkpoint["epoch"],
            "checkpoint_val_loss": float(checkpoint["val_loss"]),
            "test_set_size": n_examples,
            "test_set_note": "full existing test split (data/test.npz), no retraining, no subsampling",
            "noise_model": "i.i.d. Gaussian noise added to the simulated sensor recording, "
                            "sigma = relative_std * RMS(clean recording), evaluation-time only",
            "tr_calibration": "trcal_* = TR x per-sensor-count least-squares gain fitted on the "
                              "noiseless training split (src/calibration.py)",
            "tr_gain_per_sensor_count": {str(k): v for k, v in tr_gain.items()},
            "levels": per_level_rows,
        }, f, indent=2)

    with open(txt, "w") as f:
        f.write("Noise-robustness sensitivity analysis (real, measured -- see "
                "scripts/evaluate_noise_sensitivity.py)\n")
        f.write(f"Checkpoint: epoch {checkpoint['epoch']}, val_loss {checkpoint['val_loss']:.6f} "
                "(existing checkpoint, not retrained)\n")
        f.write(f"Test set: {n_examples} examples (full existing test split, data/test.npz)\n")
        f.write("Noise model: i.i.d. Gaussian noise added to the simulated sensor recording only "
                "(forward physics unchanged); sigma = relative_std * RMS(clean recording)\n")
        f.write("TRcal: TR x per-sensor-count gain fitted on the noiseless training split ("
                + ", ".join(f"{k} sensors: {v:.3f}" for k, v in tr_gain.items()) + ")\n\n")
        f.write("Overall (both sparsity settings pooled):\n")
        f.write(header + "\n")
        for row in per_level_rows:
            snr_str = "inf" if row["relative_std"] == 0.0 else f"{row['snr_db']:.1f}"
            f.write(f"{row['label']:<12}{row['relative_std']:<10.2f}{snr_str:<10}{row['n']:<4}"
                    f"{row['tr_psnr']:<10.3f}{row['tr_ssim']:<10.4f}{row['unet_psnr']:<11.3f}"
                    f"{row['unet_ssim']:<11.4f}{row['trcal_psnr']:<12.3f}{row['trcal_ssim']:<12.4f}\n")

        f.write("\nBy sparsity setting:\n")
        f.write(f"{'noise':<12}{'sensors':<9}{'n':<4}{'TR PSNR':<10}{'TR SSIM':<10}"
                f"{'UNet PSNR':<11}{'UNet SSIM':<11}{'TRcal PSNR':<12}{'TRcal SSIM':<12}\n")
        for row in per_level_rows:
            for k, sub in row["by_sparsity"].items():
                f.write(f"{row['label']:<12}{k:<9}{sub['n']:<4}{sub['tr_psnr']:<10.3f}"
                        f"{sub['tr_ssim']:<10.4f}{sub['unet_psnr']:<11.3f}{sub['unet_ssim']:<11.4f}"
                        f"{sub['trcal_psnr']:<12.3f}{sub['trcal_ssim']:<12.4f}\n")

    # strip the padding left after the last column of each table row
    with open(txt) as f:
        stripped = "\n".join(line.rstrip() for line in f.read().splitlines()) + "\n"
    with open(txt, "w") as f:
        f.write(stripped)

    print(f"\nSaved {txt} and {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument("--checkpoint", default="experiments/unet_checkpoint.pt")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="report")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args.device, args.checkpoint, args.data_dir, args.out_dir, args.overwrite)
