"""Tikhonov baseline on the expanded test set (run scripts/expanded_evaluation.py first).

Method: src/tikhonov.py (explicit forward matrix per sensor geometry, direct solve of the normal
equations, lam = mu * ||A^T A||). The forward matrices are cached in data/tikhonov_A_{16,64}.npy
(gitignored, about 80 MB and 310 MB).

Regularisation selection, on the TRAINING split only (16 and 24 images for 16 and 64 sensors), by
mean PSNR over the grid MU_GRID, separately per sensor count:
  * "fixed": mu selected on noiseless training recordings and then used at every noise level (the
    analogue of the U-Net, which was trained on noiseless data only);
  * "per noise level": mu selected on training recordings with the same relative noise level. The
    training noise uses its own seeds (TRAIN_NOISE_SEED), independent of the test noise.
No test image influences any choice. Validation-split PSNR at the selected mu is reported as a check.

Test data and noise: exactly the recordings of scripts/expanded_evaluation.py (same phantoms, same
noise generator, seeds and draw order), so results pair image by image with raw/calibrated
time-reversal and the U-Net in report/expanded_eval_per_image.npz.

Caveat: Tikhonov inverts the same discrete forward model that generated the synthetic data (an
"inverse crime"), which favours it relative to a setting with model mismatch.

Outputs: report/tikhonov_results.{txt,json}, report/tikhonov_per_image.npz, report/tikhonov_noise.png.

    python scripts/tikhonov_evaluation.py
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from scripts.evaluate_noise_sensitivity import NOISE_LEVELS, NOISE_SEED, add_sensor_noise  # noqa: E402
from scripts.expanded_evaluation import SENSOR_COUNTS, TEST_PATH  # noqa: E402
from scripts.generate_training_data import CENTRE, GRID_SIZE, RADIUS  # noqa: E402
from src.evaluate import psnr, ssim  # noqa: E402
from src.forward_model import build_domain_and_medium, simulate_sensor_data, sparse_view_sensor_array  # noqa: E402
from src.stats import bootstrap_mean_ci, paired_bootstrap_ci  # noqa: E402
from src.tikhonov import Tikhonov, assemble_matrix  # noqa: E402

MU_GRID = [10.0 ** e for e in range(-12, 2)]  # 1e-12 to 1e1; widened after the first run selected mu at the edges of 1e-9 to 1e-1 on training data
TRAIN_NOISE_SEED = 777_000


def recordings(phantoms, n_sensors, domain, medium, sensors):
    return [simulate_sensor_data(p, domain, medium, sensors[int(k)])[0][..., 0].reshape(-1)
            for p, k in zip(phantoms, n_sensors)]


def noisy(clean, rel_std, seed):
    """Same generator and draw order as scripts/expanded_evaluation.py for a given seed."""
    rng = np.random.default_rng(seed + int(round(rel_std * 100000)))
    return [add_sensor_noise(c[:, None, None], rel_std, rng)[:, 0, 0] if rel_std else c for c in clean]


def mean_psnr(tk, Y, P, mu):
    est = tk.solve(np.stack(Y), mu)
    return float(np.mean([psnr(e.reshape(P[0].shape).astype(np.float32), p) for e, p in zip(est, P)]))


def main():
    t0 = time.time()
    domain, medium = build_domain_and_medium(GRID_SIZE)
    sensors = {k: sparse_view_sensor_array(k, RADIUS, CENTRE) for k in SENSOR_COUNTS}
    solvers = {}
    for k in SENSOR_COUNTS:
        path = f"data/tikhonov_A_{k}.npy"
        if not os.path.exists(path):
            np.save(path, assemble_matrix(domain, medium, sensors[k]).astype(np.float32))
        solvers[k] = Tikhonov(np.load(path))
        print(f"[{time.time() - t0:.0f}s] {k} sensors: A {solvers[k].A.shape}, ||A^T A|| {solvers[k].norm_AtA:.4g}")

    train, val, test = (np.load(p) for p in ("data/train.npz", "data/val.npz", TEST_PATH))
    clean = {name: recordings(d["phantom"], d["n_sensors"], domain, medium, sensors)
             for name, d in (("train", train), ("val", val), ("test", test))}
    # the expanded evaluation adds noise to (Nt, n, 1) recordings; check the reshaped draw is identical
    probe = clean["test"][0]
    rng_a, rng_b = np.random.default_rng(1), np.random.default_rng(1)
    shaped = probe.reshape(-1, int(test["n_sensors"][0]))[..., None]
    assert np.array_equal(add_sensor_noise(shaped, 0.2, rng_a).reshape(-1),
                          add_sensor_noise(probe[:, None, None], 0.2, rng_b).reshape(-1))

    # --- regularisation selection on the training split ---
    selection = {}
    for label, rel_std in NOISE_LEVELS:
        Ytr = noisy(clean["train"], rel_std, TRAIN_NOISE_SEED)
        Yval = noisy(clean["val"], rel_std, TRAIN_NOISE_SEED + 1)
        for k in SENSOR_COUNTS:
            m = train["n_sensors"] == k
            scores = {mu: mean_psnr(solvers[k], [y for y, keep in zip(Ytr, m) if keep], train["phantom"][m], mu)
                      for mu in MU_GRID}
            best = max(scores, key=scores.get)
            vm = val["n_sensors"] == k
            selection[(label, k)] = {"train_psnr_by_mu": {f"{mu:.0e}": v for mu, v in scores.items()},
                                     "selected_mu": best,
                                     "val_psnr_at_selected": mean_psnr(solvers[k], [y for y, keep in zip(Yval, vm) if keep],
                                                                       val["phantom"][vm], best)}
        print(f"[{time.time() - t0:.0f}s] selection at noise {label}: "
              + ", ".join(f"{k} sensors mu={selection[(label, k)]['selected_mu']:.0e}" for k in SENSOR_COUNTS))

    # --- test evaluation ---
    ref = np.load("report/expanded_eval_per_image.npz")
    n_s = test["n_sensors"]
    assert np.array_equal(ref["seed"], test["seed"])
    per_image, results = {}, {"protocol": __doc__.split("Outputs:")[0].strip(), "mu_grid": MU_GRID,
                              "selection": {f"{l}|{k}": v for (l, k), v in selection.items()}, "levels": []}
    for label, rel_std in NOISE_LEVELS:
        Y = noisy(clean["test"], rel_std, NOISE_SEED)
        for variant in ("fixed", "per_level"):
            est = np.zeros_like(test["phantom"])
            for k in SENSOR_COUNTS:
                m = np.nonzero(n_s == k)[0]
                mu = selection[(NOISE_LEVELS[0][0] if variant == "fixed" else label, k)]["selected_mu"]
                est[m] = solvers[k].solve(np.stack([Y[i] for i in m]), mu).reshape(len(m), GRID_SIZE, GRID_SIZE)
            per_image[(label, variant, "psnr")] = np.array([psnr(e, p) for e, p in zip(est, test["phantom"])])
            per_image[(label, variant, "ssim")] = np.array([ssim(e, p) for e, p in zip(est, test["phantom"])])
        level = {"label": label, "relative_std": rel_std, "by_sensors": {}}
        for k in SENSOR_COUNTS:
            m = n_s == k
            entry = {}
            for metric in ("psnr", "ssim"):
                other = lambda name: ref[f"{label}__{name}__{metric}"][m]
                e = {}
                for variant in ("fixed", "per_level"):
                    v = per_image[(label, variant, metric)][m]
                    e[variant] = {"mean_ci95": bootstrap_mean_ci(v),
                                  "minus_unet": paired_bootstrap_ci(v, other("unet_mean_over_seeds")),
                                  "minus_tr_cal": paired_bootstrap_ci(v, other("tr_cal"))}
                e["reference_means"] = {n: float(other(n).mean()) for n in ("tr_raw", "tr_cal", "unet_mean_over_seeds")}
                entry[metric] = e
            level["by_sensors"][str(k)] = entry
        results["levels"].append(level)
        print(f"[{time.time() - t0:.0f}s] test noise {label} done")

    with open("report/tikhonov_results.json", "w") as f:
        json.dump(results, f, indent=2)
    np.savez_compressed("report/tikhonov_per_image.npz", seed=test["seed"], n_sensors=n_s,
                        **{f"{l}__tikhonov_{v}__{m}": a for (l, v, m), a in per_image.items()})
    write_text(results)
    plot(results)


def write_text(r):
    L = ["Tikhonov baseline (scripts/tikhonov_evaluation.py)", "", r["protocol"], "",
         "Selected mu (training split) and validation PSNR at that mu:"]
    for key, s in r["selection"].items():
        label, k = key.split("|")
        L.append(f"  noise {label:9s} {k} sensors: mu {s['selected_mu']:.0e}, val PSNR {s['val_psnr_at_selected']:.2f}; "
                 "train PSNR by mu: " + ", ".join(f"{mu} {v:.2f}" for mu, v in s["train_psnr_by_mu"].items()))
    L.append("")
    for lvl in r["levels"]:
        L.append(f"== noise {lvl['label']}")
        for k, e in lvl["by_sensors"].items():
            for metric, d in (("psnr", 2), ("ssim", 3)):
                x = e[metric]; ref = x["reference_means"]
                parts = []
                for v in ("fixed", "per_level"):
                    m, lo, hi = x[v]["mean_ci95"]; du = x[v]["minus_unet"]; dc = x[v]["minus_tr_cal"]
                    parts.append(f"Tikhonov {v} {m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}] (vs U-Net {du[0]:+.{d}f} "
                                 f"[{du[1]:+.{d}f}, {du[2]:+.{d}f}], vs cal TR {dc[0]:+.{d}f} [{dc[1]:+.{d}f}, {dc[2]:+.{d}f}])")
                L.append(f"  {k} sensors {metric.upper()}: " + " | ".join(parts)
                         + f" | raw TR {ref['tr_raw']:.{d}f}, cal TR {ref['tr_cal']:.{d}f}, U-Net {ref['unet_mean_over_seeds']:.{d}f}")
    with open("report/tikhonov_results.txt", "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def plot(r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def snr(rel_std):
        return "inf" if rel_std == 0 else "{:.0f}".format(20 * np.log10(1 / rel_std))
    labels = ["{}\n{} dB".format(l["label"], snr(l["relative_std"])) for l in r["levels"]]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, k in zip(axes, ("16", "64")):
        e = [l["by_sensors"][k]["psnr"] for l in r["levels"]]
        for key, lab, c, mk in (("tr_cal", "calibrated time-reversal", "C1", "o"),
                                ("unet_mean_over_seeds", "U-Net (mean of 5 networks)", "C0", "D")):
            ax.plot(x, [v["reference_means"][key] for v in e], color=c, marker=mk, label=lab)
        m = np.array([q["per_level"]["mean_ci95"][0] for q in e])
        lo = np.array([q["per_level"]["mean_ci95"][1] for q in e]); hi = np.array([q["per_level"]["mean_ci95"][2] for q in e])
        ax.errorbar(x, m, yerr=[m - lo, hi - m], color="C2", marker="s", ms=4, capsize=3,
                    label="Tikhonov, mu selected per noise level on training data")
        top = 60.0
        for xi, mi in zip(x, m):
            if mi > top:  # off-scale point: mark it at the top edge with its value
                ax.annotate(f"{mi:.1f} dB", (xi, top), xytext=(xi + 0.15, top - 4), fontsize=8, color="C2",
                            arrowprops=dict(arrowstyle="->", color="C2"))
        ax.set_ylim(15, top)
        ax.set_title(f"{k} sensors", fontsize=10)
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_xlabel("sensor noise level (SNR)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("PSNR [dB]")
    axes[1].legend(fontsize=7.5, loc="upper right")
    fig.suptitle("Tikhonov on the 400-image expanded test set; error bars: 95% bootstrap CI over images", fontsize=10)
    fig.tight_layout()
    fig.savefig("report/tikhonov_noise.png", dpi=130)


if __name__ == "__main__":
    main()
