"""Expanded evaluation: 400 held-out test images, 5 independently trained U-Nets, 5 noise levels.

Protocol (fixed before any result was seen):
  * Training/validation data: the existing data/train.npz (40) and data/val.npz (8), unchanged.
  * Test data: 400 new phantoms, seeds 100000-100399 (disjoint from train 0-39, val 10000-10007 and
    the original test 20000-20007; checked below), exactly 200 per sensor count (even index -> 16
    sensors, odd -> 64). Phantoms, forward model and time reversal are generated exactly as in
    scripts/generate_training_data.py. Written to data/test_expanded.npz (gitignored).
  * U-Net: the existing recipe (scripts/train.py: 60 epochs, Adam 1e-3, batch 8, best-validation
    checkpoint) trained on CPU with deterministic algorithms for seeds 0-4, written to
    experiments/expanded/unet_seed{s}.pt (gitignored). The original MPS checkpoint
    (experiments/unet_checkpoint.pt) is also scored, for continuity only.
  * Time-reversal calibration: one least-squares gain per sensor count fitted on the training split
    (src/calibration.py), applied unchanged to every test image and noise level. Its uncertainty is a
    bootstrap over training images; the gain fitted on the validation split is reported for comparison.
  * Noise: the levels, noise model and noise seeds of scripts/evaluate_noise_sensitivity.py.
  * Uncertainty: 95% percentile bootstrap intervals over test images (2000 resamples, seed 0). For the
    U-Net the headline per-image score is the mean over the 5 trained networks; the spread between
    networks is reported separately (min/max/sd of the 5 per-network means), never combined with the
    image-level interval.

Outputs: report/expanded_eval_results.txt, report/expanded_eval_results.json,
report/expanded_eval_per_image.npz (per-image PSNR/SSIM) and report/expanded_eval_noise.png.

    python scripts/expanded_evaluation.py
"""
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from scripts.evaluate_noise_sensitivity import NOISE_LEVELS, NOISE_SEED, add_sensor_noise  # noqa: E402
from scripts.generate_training_data import CENTRE, GRID_SIZE, RADIUS  # noqa: E402
from scripts.train import get_device, train  # noqa: E402
from src.baselines import time_reversal_reconstruction  # noqa: E402
from src.calibration import fit_scale, fit_scales_per_sensor_count  # noqa: E402
from src.evaluate import psnr, ssim  # noqa: E402
from src.forward_model import build_domain_and_medium, simulate_sensor_data, sparse_view_sensor_array  # noqa: E402
from src.phantoms import random_blob_phantom  # noqa: E402
from src.reconstruction_net import ReconstructionUNet  # noqa: E402
from src.stats import bootstrap_mean_ci, paired_bootstrap_ci  # noqa: E402

TEST_SEED_OFFSET = 100_000
N_PER_SENSOR = 200
SENSOR_COUNTS = (16, 64)
TRAIN_SEEDS = (0, 1, 2, 3, 4)
TEST_PATH = "data/test_expanded.npz"
CKPT_DIR = "experiments/expanded"


def generate_test_set():
    n = N_PER_SENSOR * len(SENSOR_COUNTS)
    rng = np.random.default_rng(TEST_SEED_OFFSET)
    domain, medium = build_domain_and_medium(GRID_SIZE)
    sensors = {k: sparse_view_sensor_array(k, RADIUS, CENTRE) for k in SENSOR_COUNTS}
    phantoms = np.zeros((n, GRID_SIZE, GRID_SIZE), np.float32)
    recons = np.zeros_like(phantoms)
    n_sensors = np.array([SENSOR_COUNTS[i % 2] for i in range(n)], np.int32)
    seeds = TEST_SEED_OFFSET + np.arange(n, dtype=np.int64)
    for i in range(n):
        n_blobs = int(rng.integers(1, 4))
        phantoms[i] = random_blob_phantom(size=GRID_SIZE, seed=int(seeds[i]), n_blobs=n_blobs)
        rec, t_axis = simulate_sensor_data(phantoms[i], domain, medium, sensors[int(n_sensors[i])])
        recons[i] = time_reversal_reconstruction(rec, sensors[int(n_sensors[i])], domain, medium, t_axis)
    used = set()
    for split in ("train", "val", "test"):
        used |= set(np.load(f"data/{split}.npz")["seed"].tolist())
    assert not used & set(seeds.tolist()), "expanded test seeds overlap an existing split"
    np.savez(TEST_PATH, phantom=phantoms, recon=recons, n_sensors=n_sensors, seed=seeds)


def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = ReconstructionUNet(base_features=16)
    model.load_state_dict(ckpt["model_state"])
    return model.eval(), ckpt


def summarise(values):
    m, lo, hi = bootstrap_mean_ci(values)
    return {"mean": m, "ci95": [lo, hi]}


def main():
    t0 = time.time()
    if not os.path.exists(TEST_PATH):
        generate_test_set()
    test = np.load(TEST_PATH)
    phantoms, n_sensors = test["phantom"], test["n_sensors"]
    print(f"[{time.time() - t0:.0f}s] test set: {len(phantoms)} images, "
          + ", ".join(f"{k} sensors: {int((n_sensors == k).sum())}" for k in SENSOR_COUNTS))

    os.makedirs(CKPT_DIR, exist_ok=True)
    device = get_device("cpu")
    for s in TRAIN_SEEDS:
        path = f"{CKPT_DIR}/unet_seed{s}.pt"
        if not os.path.exists(path):
            train(device, seed=s, checkpoint_path=path, verbose=False)
    models = {f"seed{s}": load_model(f"{CKPT_DIR}/unet_seed{s}.pt") for s in TRAIN_SEEDS}
    reference = load_model("experiments/unet_checkpoint.pt")
    print(f"[{time.time() - t0:.0f}s] networks: "
          + ", ".join(f"{k} (epoch {c['epoch']}, val {c['val_loss']:.6f})" for k, (_, c) in models.items()))

    # --- calibration, training split only ---
    train_split, val_split = np.load("data/train.npz"), np.load("data/val.npz")
    gain = fit_scales_per_sensor_count(train_split["recon"], train_split["phantom"], train_split["n_sensors"])
    calib = {}
    rng = np.random.default_rng(0)
    for k in SENSOR_COUNTS:
        idx = np.nonzero(train_split["n_sensors"] == k)[0]
        boots = [fit_scale(train_split["recon"][b], train_split["phantom"][b])
                 for b in (rng.choice(idx, len(idx)) for _ in range(2000))]
        vidx = val_split["n_sensors"] == k
        calib[k] = {"train_gain": gain[k], "n_train_images": int(len(idx)),
                    "train_gain_bootstrap_ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
                    "val_gain": fit_scale(val_split["recon"][vidx], val_split["phantom"][vidx]) if vidx.any() else None,
                    "n_val_images": int(vidx.sum()),
                    "test_oracle_gain_for_reference": fit_scale(test["recon"][n_sensors == k], phantoms[n_sensors == k])}
    gains = np.array([gain[int(k)] for k in n_sensors], np.float32)[:, None, None]

    # --- reconstruct at every noise level ---
    domain, medium = build_domain_and_medium(GRID_SIZE)
    sensors = {k: sparse_view_sensor_array(k, RADIUS, CENTRE) for k in SENSOR_COUNTS}
    clean = [simulate_sensor_data(phantoms[i], domain, medium, sensors[int(n_sensors[i])]) for i in range(len(phantoms))]
    methods = ["tr_raw", "tr_cal"] + list(models) + ["reference_checkpoint"]
    per_image = {}  # (level, method, metric) -> (N,)
    for label, rel_std in NOISE_LEVELS:
        noise_rng = np.random.default_rng(NOISE_SEED + int(round(rel_std * 100000)))
        tr = np.zeros_like(phantoms)
        for i, (rec, t_axis) in enumerate(clean):
            noisy = add_sensor_noise(rec, rel_std, noise_rng)
            tr[i] = time_reversal_reconstruction(noisy, sensors[int(n_sensors[i])], domain, medium, t_axis)
        if rel_std == 0.0:
            assert np.array_equal(tr, test["recon"]), "noiseless reconstruction differs from the stored test set"
        outputs = {"tr_raw": tr, "tr_cal": tr * gains}
        x = torch.from_numpy(tr).unsqueeze(1)
        with torch.no_grad():
            for name, (model, _) in list(models.items()) + [("reference_checkpoint", reference)]:
                outputs[name] = model(x).squeeze(1).numpy()
        for name in methods:
            per_image[(label, name, "psnr")] = np.array([psnr(outputs[name][i], phantoms[i]) for i in range(len(phantoms))])
            per_image[(label, name, "ssim")] = np.array([ssim(outputs[name][i], phantoms[i]) for i in range(len(phantoms))])
        print(f"[{time.time() - t0:.0f}s] noise level {label} done")

    seed_names = list(models)
    for label, _ in NOISE_LEVELS:
        for metric in ("psnr", "ssim"):
            per_image[(label, "unet_mean_over_seeds", metric)] = np.mean(
                [per_image[(label, s, metric)] for s in seed_names], axis=0)

    # --- summaries ---
    results = {"protocol": __doc__.split("Outputs:")[0].strip(), "calibration": {str(k): v for k, v in calib.items()},
               "networks": {k: {"epoch": int(c["epoch"]), "val_loss": float(c["val_loss"])} for k, (_, c) in models.items()},
               "levels": []}
    base = NOISE_LEVELS[0][0]
    for label, rel_std in NOISE_LEVELS:
        level = {"label": label, "relative_std": rel_std, "by_sensors": {}}
        for k in SENSOR_COUNTS:
            m = n_sensors == k
            entry = {"n_images": int(m.sum())}
            for metric in ("psnr", "ssim"):
                g = lambda name: per_image[(label, name, metric)][m]
                seed_means = np.array([g(s).mean() for s in seed_names])
                entry[metric] = {
                    "tr_raw": summarise(g("tr_raw")),
                    "tr_cal": summarise(g("tr_cal")),
                    "unet": {**summarise(g("unet_mean_over_seeds")),
                             "across_training_seeds": {"per_seed_means": seed_means.tolist(),
                                                       "sd": float(seed_means.std(ddof=1)),
                                                       "min": float(seed_means.min()), "max": float(seed_means.max())}},
                    "reference_checkpoint": summarise(g("reference_checkpoint")),
                    "unet_minus_tr_cal": dict(zip(("mean", "ci95_lo", "ci95_hi"),
                                                  paired_bootstrap_ci(g("unet_mean_over_seeds"), g("tr_cal")))),
                    "unet_minus_tr_cal_per_seed": [float((g(s) - g("tr_cal")).mean()) for s in seed_names],
                }
                if label != base:
                    for name in ("tr_raw", "tr_cal", "unet_mean_over_seeds"):
                        drop = paired_bootstrap_ci(per_image[(label, name, metric)][m], per_image[(base, name, metric)][m])
                        entry[metric].setdefault("change_from_noiseless", {})[name] = dict(
                            zip(("mean", "ci95_lo", "ci95_hi"), drop))
            level["by_sensors"][str(k)] = entry
        results["levels"].append(level)

    os.makedirs("report", exist_ok=True)
    with open("report/expanded_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    np.savez_compressed("report/expanded_eval_per_image.npz", n_sensors=n_sensors, seed=test["seed"],
                        **{f"{lvl}__{name}__{met}": v for (lvl, name, met), v in per_image.items()})
    write_text(results)
    plot(results)
    print(f"[{time.time() - t0:.0f}s] wrote report/expanded_eval_*")


def fmt(s, digits):
    return f"{s['mean']:.{digits}f} [{s['ci95'][0]:.{digits}f}, {s['ci95'][1]:.{digits}f}]"


def write_text(r):
    L = ["Expanded evaluation (scripts/expanded_evaluation.py)", "", r["protocol"], "",
         "Calibration gains (training split; bootstrap over training images):"]
    for k, c in r["calibration"].items():
        L.append(f"  {k} sensors: train {c['train_gain']:.2f} (95% CI {c['train_gain_bootstrap_ci95'][0]:.2f}"
                 f"-{c['train_gain_bootstrap_ci95'][1]:.2f}, n={c['n_train_images']}); validation "
                 f"{c['val_gain']:.2f} (n={c['n_val_images']}); test-set oracle, reference only "
                 f"{c['test_oracle_gain_for_reference']:.2f}")
    L += ["", "Networks: " + ", ".join(f"{k} epoch {v['epoch']} val {v['val_loss']:.6f}" for k, v in r["networks"].items()), ""]
    for lvl in r["levels"]:
        L.append(f"== noise {lvl['label']} (relative std {lvl['relative_std']})")
        for k, e in lvl["by_sensors"].items():
            for metric, d in (("psnr", 2), ("ssim", 3)):
                x = e[metric]; u = x["unet"]["across_training_seeds"]; diff = x["unet_minus_tr_cal"]
                L.append(f"  {k} sensors, {metric.upper()} (n={e['n_images']}): raw TR {fmt(x['tr_raw'], d)} | "
                         f"cal TR {fmt(x['tr_cal'], d)} | U-Net {fmt(x['unet'], d)} "
                         f"(5 networks: {u['min']:.{d}f}-{u['max']:.{d}f}, sd {u['sd']:.{d}f}) | "
                         f"U-Net - cal TR {diff['mean']:.{d}f} [{diff['ci95_lo']:.{d}f}, {diff['ci95_hi']:.{d}f}] | "
                         f"original checkpoint {fmt(x['reference_checkpoint'], d)}")
                if "change_from_noiseless" in x:
                    c = x["change_from_noiseless"]
                    L.append("      change from noiseless: " + ", ".join(
                        f"{n} {v['mean']:+.{d}f} [{v['ci95_lo']:+.{d}f}, {v['ci95_hi']:+.{d}f}]" for n, v in c.items()))
    with open("report/expanded_eval_results.txt", "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[-60:]))


def plot(r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def snr(rel_std):
        return "inf" if rel_std == 0 else "{:.0f}".format(20 * np.log10(1 / rel_std))
    labels = ["{}\n{} dB".format(l["label"], snr(l["relative_std"])) for l in r["levels"]]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    styles = {"tr_raw": ("raw time-reversal", "C7", "s"), "tr_cal": ("calibrated time-reversal", "C1", "o"),
              "unet": ("U-Net (mean of 5 networks)", "C0", "D")}
    for col, k in enumerate(("16", "64")):
        for row, (metric, name) in enumerate((("psnr", "PSNR [dB]"), ("ssim", "SSIM"))):
            ax = axes[row, col]
            for key, (lab, color, marker) in styles.items():
                s = [l["by_sensors"][k][metric][key] for l in r["levels"]]
                mean = np.array([v["mean"] for v in s]); lo = np.array([v["ci95"][0] for v in s]); hi = np.array([v["ci95"][1] for v in s])
                ax.errorbar(x, mean, yerr=[mean - lo, hi - mean], color=color, marker=marker, ms=5, capsize=3, label=lab)
                if key == "unet":
                    seeds = np.array([l["by_sensors"][k][metric]["unet"]["across_training_seeds"]["per_seed_means"] for l in r["levels"]])
                    ax.fill_between(x, seeds.min(1), seeds.max(1), color=color, alpha=0.18, lw=0,
                                    label="range of the 5 networks' means")
            ax.set_title(f"{k} sensors: {name}", fontsize=10)
            ax.grid(alpha=0.3)
            if col == 0:
                ax.set_ylabel(name)
            if row == 1:
                ax.set_xticks(x, labels, fontsize=8)
                ax.set_xlabel("sensor noise level (SNR)")
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle("400 held-out test images (200 per sensor count); error bars: 95% bootstrap CI over images",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig("report/expanded_eval_noise.png", dpi=130)


if __name__ == "__main__":
    main()
