"""Model-mismatch check: how much of each method's accuracy depends on knowing the exact forward model?

Every other evaluation in this repository generates test data with exactly the operator that the
reconstruction methods assume (an "inverse crime"): Tikhonov inverts the same discrete matrix, and
time reversal and the U-Net's training data use the same simulator. This script perturbs one
physical parameter of the DATA only and keeps every method on the nominal model.

Protocol (fixed before any result was seen):
  * Mismatch: the true sound speed of the medium is c_true = 1500 * (1 + delta) m/s for
    delta in {0, +1 %, +2 %}; every method assumes the nominal 1500 m/s. (Sound speed in water
    changes by about 2 % over roughly 10 degrees C; soft tissue is about 1540 m/s.) delta = 0 is a
    control and must reproduce the saved results of scripts/expanded_evaluation.py and
    scripts/tikhonov_evaluation.py.
  * Sensors sample on the NOMINAL time axis (dt, Nt of the 1500 m/s medium), as a physical
    acquisition system would; only the wave propagation uses c_true.
  * Test data: the 400 phantoms of data/test_expanded.npz (200 per sensor count), unchanged.
  * Noise: noiseless and "high" (relative std 0.20, about 14 dB SNR), with the generator, seeds and
    draw order of scripts/expanded_evaluation.py, so draws are identical across delta.
  * Methods, all nominal and unchanged: raw and training-calibrated time reversal (gains from the
    training split); the 5 U-Nets of scripts/expanded_evaluation.py applied to nominal time
    reversal; Tikhonov with the nominal matrices (data/tikhonov_A_{16,64}.npy) and the mu already
    selected on the training split for that noise level (report/tikhonov_results.json). Nothing is
    refitted or reselected under mismatch.
  * Uncertainty: 95 % bootstrap intervals over test images; U-Net spread across the 5 networks
    reported separately.

Outputs: report/model_mismatch_results.{txt,json}, report/model_mismatch_per_image.npz,
report/model_mismatch.png.

    python scripts/model_mismatch_evaluation.py
"""
import json
import os
import sys
import time

import jax.numpy as jnp
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from jaxdf.discretization import FourierSeries  # noqa: E402
from jwave.acoustics import simulate_wave_propagation  # noqa: E402
from jwave.geometry import Sensors, TimeAxis  # noqa: E402

from scripts.evaluate_noise_sensitivity import NOISE_LEVELS, NOISE_SEED, add_sensor_noise  # noqa: E402
from scripts.expanded_evaluation import CKPT_DIR, SENSOR_COUNTS, TEST_PATH, TRAIN_SEEDS, load_model  # noqa: E402
from scripts.generate_training_data import CENTRE, GRID_SIZE, RADIUS  # noqa: E402
from src.baselines import time_reversal_reconstruction  # noqa: E402
from src.calibration import training_scales  # noqa: E402
from src.evaluate import psnr, ssim  # noqa: E402
from src.forward_model import DEFAULT_SOUND_SPEED, build_domain_and_medium, sparse_view_sensor_array  # noqa: E402
from src.stats import bootstrap_mean_ci, paired_bootstrap_ci  # noqa: E402
from src.tikhonov import Tikhonov  # noqa: E402

DELTAS = (0.0, 0.01, 0.02)
LEVELS = [(l, s) for l, s in NOISE_LEVELS if l in ("noiseless", "high")]


def simulate(p0, domain, medium_true, time_axis_nominal, sensor_positions):
    """Forward simulation in the TRUE medium, sampled on the NOMINAL time axis; (Nt, n, 1)."""
    out = simulate_wave_propagation(medium_true, time_axis_nominal, p0=FourierSeries(jnp.array(p0), domain),
                                    sensors=Sensors(positions=sensor_positions))
    return np.asarray(out)


def main():
    t0 = time.time()
    test = np.load(TEST_PATH)
    phantoms, n_sensors = test["phantom"], test["n_sensors"]
    domain, medium_nom = build_domain_and_medium(GRID_SIZE)
    time_axis = TimeAxis.from_medium(medium_nom, cfl=0.3)
    sensors = {k: sparse_view_sensor_array(k, RADIUS, CENTRE) for k in SENSOR_COUNTS}
    gains = training_scales()
    gain_img = np.array([gains[int(k)] for k in n_sensors], np.float32)[:, None, None]
    models = {f"seed{s}": load_model(f"{CKPT_DIR}/unet_seed{s}.pt")[0] for s in TRAIN_SEEDS}
    solvers = {k: Tikhonov(np.load(f"data/tikhonov_A_{k}.npy")) for k in SENSOR_COUNTS}
    selection = json.load(open("report/tikhonov_results.json"))["selection"]
    mu = {(l, k): selection[f"{l}|{k}"]["selected_mu"] for l, _ in LEVELS for k in SENSOR_COUNTS}
    saved_eval = np.load("report/expanded_eval_per_image.npz")
    saved_tik = np.load("report/tikhonov_per_image.npz")

    per_image = {}
    for delta in DELTAS:
        _, medium_true = build_domain_and_medium(GRID_SIZE, sound_speed=DEFAULT_SOUND_SPEED * (1 + delta))
        clean = [simulate(phantoms[i], domain, medium_true, time_axis, sensors[int(n_sensors[i])])
                 for i in range(len(phantoms))]
        for label, rel_std in LEVELS:
            rng = np.random.default_rng(NOISE_SEED + int(round(rel_std * 100000)))
            recs = [add_sensor_noise(c, rel_std, rng) for c in clean]
            tr = np.stack([time_reversal_reconstruction(r, sensors[int(k)], domain, medium_nom, time_axis)
                           for r, k in zip(recs, n_sensors)])
            out = {"tr_raw": tr, "tr_cal": tr * gain_img}
            x = torch.from_numpy(tr).unsqueeze(1)
            with torch.no_grad():
                for name, model in models.items():
                    out[name] = model(x).squeeze(1).numpy()
            tik = np.zeros_like(phantoms)
            for k in SENSOR_COUNTS:
                m = np.nonzero(n_sensors == k)[0]
                Y = np.stack([recs[i][..., 0].reshape(-1) for i in m])
                tik[m] = solvers[k].solve(Y, mu[(label, k)]).reshape(len(m), GRID_SIZE, GRID_SIZE)
            out["tikhonov"] = tik
            for name, img in out.items():
                for metric, f in (("psnr", psnr), ("ssim", ssim)):
                    per_image[(delta, label, name, metric)] = np.array([f(img[i], phantoms[i]) for i in range(len(phantoms))])
            for metric in ("psnr", "ssim"):
                per_image[(delta, label, "unet_mean_over_seeds", metric)] = np.mean(
                    [per_image[(delta, label, s, metric)] for s in models], axis=0)
            print(f"[{time.time() - t0:.0f}s] delta {delta:+.0%}, noise {label} done", flush=True)

    # control: delta = 0 must reproduce the saved per-image results
    control = {}
    for label, _ in LEVELS:
        for metric in ("psnr", "ssim"):
            for name, saved in (("tr_cal", saved_eval[f"{label}__tr_cal__{metric}"]),
                                ("unet_mean_over_seeds", saved_eval[f"{label}__unet_mean_over_seeds__{metric}"]),
                                ("tikhonov", saved_tik[f"{label}__tikhonov_per_level__{metric}"])):
                control[f"{label}|{name}|{metric}"] = float(np.max(np.abs(per_image[(0.0, label, name, metric)] - saved)))

    results = {"protocol": __doc__.split("Outputs:")[0].strip(), "deltas": DELTAS,
               "mu": {f"{l}|{k}": v for (l, k), v in mu.items()},
               "control_max_abs_difference_vs_saved": control, "rows": []}
    for delta in DELTAS:
        for label, _ in LEVELS:
            for k in SENSOR_COUNTS:
                m = n_sensors == k
                row = {"delta": delta, "noise": label, "sensors": k}
                for metric in ("psnr", "ssim"):
                    g = lambda n: per_image[(delta, label, n, metric)][m]
                    row[metric] = {n: bootstrap_mean_ci(g(n)) for n in ("tr_raw", "tr_cal", "unet_mean_over_seeds", "tikhonov")}
                    row[metric]["unet_per_seed_means"] = [float(g(s).mean()) for s in models]
                    row[metric]["tikhonov_minus_unet"] = paired_bootstrap_ci(g("tikhonov"), g("unet_mean_over_seeds"))
                    row[metric]["unet_minus_tr_cal"] = paired_bootstrap_ci(g("unet_mean_over_seeds"), g("tr_cal"))
                    row[metric]["tikhonov_minus_best_unet_seed"] = float(g("tikhonov").mean() - max(g(s).mean() for s in models))
                results["rows"].append(row)

    os.makedirs("report", exist_ok=True)
    with open("report/model_mismatch_results.json", "w") as f:
        json.dump(results, f, indent=2)
    np.savez_compressed("report/model_mismatch_per_image.npz", seed=test["seed"], n_sensors=n_sensors,
                        **{f"{d:+.2f}__{l}__{n}__{m}": v for (d, l, n, m), v in per_image.items()})
    write_text(results)
    plot(results)
    print(f"[{time.time() - t0:.0f}s] wrote report/model_mismatch_*")


def write_text(r):
    L = ["Model-mismatch check (scripts/model_mismatch_evaluation.py)", "", r["protocol"], "",
         "Control (delta = 0) max |difference| from saved per-image results: "
         + ", ".join(f"{k} {v:.2e}" for k, v in r["control_max_abs_difference_vs_saved"].items()), ""]
    for row in r["rows"]:
        for metric, d in (("psnr", 2), ("ssim", 3)):
            x = row[metric]
            f = lambda t: f"{t[0]:.{d}f} [{t[1]:.{d}f}, {t[2]:.{d}f}]"
            L.append(f"c {1500 * (1 + row['delta']):.0f} m/s ({row['delta']:+.0%}), noise {row['noise']:9s}, {row['sensors']} sensors, "
                     f"{metric.upper()}: raw TR {x['tr_raw'][0]:.{d}f} | cal TR {f(x['tr_cal'])} | U-Net {f(x['unet_mean_over_seeds'])} "
                     f"(5 networks {min(x['unet_per_seed_means']):.{d}f}-{max(x['unet_per_seed_means']):.{d}f}) | Tikhonov {f(x['tikhonov'])} "
                     f"| Tikhonov - U-Net {f(x['tikhonov_minus_unet'])} | Tikhonov - best network {x['tikhonov_minus_best_unet_seed']:+.{d}f}")
    with open("report/model_mismatch_results.txt", "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


def plot(r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lo_lim, hi_lim = 15.0, 45.0
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), sharex=True)
    styles = (("tr_cal", "calibrated time reversal", "C1", "o"), ("unet_mean_over_seeds", "U-Net (mean of 5 networks)", "C0", "D"),
              ("tikhonov", "Tikhonov (nominal operator, mu from training split)", "C2", "s"))
    x = np.array([100 * d for d in r["deltas"]])
    for row_i, (label, snr) in enumerate((("noiseless", "noiseless"), ("high", "14 dB SNR"))):
        for col, k in enumerate(SENSOR_COUNTS):
            ax = axes[row_i, col]
            rows = [q for q in r["rows"] if q["noise"] == label and q["sensors"] == k]
            seeds = np.array([q["psnr"]["unet_per_seed_means"] for q in rows])
            ax.fill_between(x, seeds.min(1), seeds.max(1), color="C0", alpha=0.18, lw=0, label="range of the 5 networks")
            for key, lab, c, mk in styles:
                v = np.array([q["psnr"][key] for q in rows])
                shown = np.clip(v[:, 0], lo_lim, hi_lim)
                ax.errorbar(x, shown, yerr=[np.where(v[:, 0] == shown, v[:, 0] - v[:, 1], 0),
                                            np.where(v[:, 0] == shown, v[:, 2] - v[:, 0], 0)],
                            color=c, marker=mk, capsize=3, label=lab)
                for xi, vi, si in zip(x, v[:, 0], shown):   # values outside the axis are drawn at the edge and labelled
                    if vi != si:
                        ax.annotate(f"{vi:.1f} dB", (xi, si), xytext=(6, -12 if si == hi_lim else 6),
                                    textcoords="offset points", fontsize=8, color=c)
            ax.set_ylim(lo_lim - 1, hi_lim + 1)
            ax.set_title(f"{k} sensors, {snr}", fontsize=10)
            ax.grid(alpha=0.3)
            if col == 0:
                ax.set_ylabel("PSNR [dB]")
            if row_i == 1:
                ax.set_xticks(x, [f"+{d:.0f} %" if d else "0 (matched)" for d in x])
                ax.set_xlabel("sound-speed error of the assumed model")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Model mismatch: data generated with c = 1500 (1 + delta) m/s, every method assumes 1500 m/s; 400 test images "
                 "(200 per sensor count).\nError bars: 95 % bootstrap CI over images; values outside the axis are drawn at "
                 "its edge and labelled.", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig("report/model_mismatch.png", dpi=130)


if __name__ == "__main__":
    main()
