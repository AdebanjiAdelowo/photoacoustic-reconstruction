"""Why does the unrolled network lose its advantage on some images? Distance from the training
distribution, and two controlled shifts.

The mixed-family experiment (scripts/mixed_phantom_experiment.py) showed that the advantage of the
unrolled network over Tikhonov depends on the test image family. This analysis asks whether that is
explained by how far a test image lies from the images the network was trained on.

Protocol (fixed before any result of this analysis was seen):

  1. Distance and advantage on the saved shape test set (no new reconstructions).
     For each of the 500 test images of set C, src/image_statistics.py gives seven structural
     statistics of the ground-truth image. The distance of a test image from a training set is the
     Mahalanobis distance of its statistics from those of the training images: the 40 blob images of
     data/train.npz for the blob-trained networks, the 160 mixed images for the mixed160 networks.
     The outcome is the saved per-image PSNR advantage of the network (mean of 5) over Tikhonov.
     Hypothesis H1: the advantage decreases with distance. Reported: Spearman rank correlation with a
     95 % bootstrap interval, pooled over the 250 images of each sensor count and within each family,
     for every noise level, and the correlation of the advantage with each single statistic.

  2. Controlled shifts (new images, existing networks, nothing retrained).
     One structural property is varied while everything else is held fixed:
       * vessel width: the same 40 vessel curves drawn with profile widths (standard deviation) of
         1, 1.5, 2, 3, 4 and 6 pixels. Thin lines are unlike every training image; wide ones
         approach elongated blobs.
       * edge blur: the same 40 disc images blurred with a Gaussian of 0, 0.5, 1, 2, 3 and 4
         pixels. Sharp discs are a mixed-training family and unlike blobs; blurred discs approach
         blobs.
     20 images per sensor count and level, without noise and at the "high" noise level (14 dB SNR).
     Tikhonov uses the weights selected on the mixed160 training split
     (report/mixed_phantom_results.json). Phantom seeds 400000+ and 410000+, noise seed 969000.
     Hypothesis H2: the advantage of both network groups over Tikhonov grows with vessel width.
     Hypothesis H3: with increasing blur the blob-trained networks gain an advantage on discs, and
     the mixed networks keep theirs.

Outputs: report/prior_shift_results.{txt,json}, report/prior_shift_distance.png,
report/prior_shift_sweeps.png. Requires the outputs of mixed_phantom_experiment.py.

    python scripts/prior_shift_analysis.py
"""
import json
import os
import sys
import time

import numpy as np
from scipy import ndimage
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import scripts.mixed_phantom_experiment as M  # noqa: E402
import scripts.unrolled_evaluation as U  # noqa: E402
from scripts.evaluate_noise_sensitivity import NOISE_LEVELS  # noqa: E402
from scripts.tikhonov_evaluation import noisy  # noqa: E402
from src.evaluate import psnr  # noqa: E402
from src.image_statistics import FEATURES, ReferenceDistribution, feature_matrix  # noqa: E402
from src.run_safety import ensure_writable  # noqa: E402
from src.shape_phantoms import shape_phantom, vessel_phantom  # noqa: E402
from src.stats import bootstrap_mean_ci, paired_bootstrap_ci  # noqa: E402

GROUPS = (M.BASELINE, "mixed160")
VESSEL_WIDTHS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
DISC_BLURS = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0)
N_SWEEP = 40
VESSEL_SEED_OFFSET, DISC_SEED_OFFSET, SWEEP_NOISE_SEED = 400_000, 410_000, 969_000
SWEEP_LEVELS = [(l, s) for l, s in NOISE_LEVELS if l in ("noiseless", "high")]
OUTPUT_NAMES = ("prior_shift_results.json", "prior_shift_results.txt", "prior_shift_distance.png", "prior_shift_sweeps.png")


def spearman_ci(x, y, n_boot=2000, seed=0):
    """Spearman rank correlation with a percentile bootstrap interval over images."""
    x, y = np.asarray(x), np.asarray(y)
    rho = float(spearmanr(x, y)[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = [spearmanr(x[i], y[i])[0] for i in idx]
    lo, hi = np.nanquantile(boots, [0.025, 0.975])
    return [rho, float(lo), float(hi)]


def training_references():
    blob_train = np.load("data/train.npz")["phantom"]
    mixed_train = M.mixed_split(M.TRAIN_SEED_OFFSET, M.MIXED_SETS["mixed160"]["n_train"])["phantom"]
    return {M.BASELINE: ReferenceDistribution(feature_matrix(blob_train)),
            "mixed160": ReferenceDistribution(feature_matrix(mixed_train))}


def distance_analysis(references):
    saved = np.load("report/mixed_phantom_per_image.npz")
    test = M.shape_test_set()
    assert np.array_equal(saved["shape_test_seed"], test["seed"]), "saved results belong to another test set"
    family, n_s = test["family"], test["n_sensors"]
    features = feature_matrix(test["phantom"])
    distance = {g: references[g].distance(features) for g in GROUPS}
    out = {"n_images": int(len(family)),
           "family_feature_means": {f: dict(zip(FEATURES, features[family == f].mean(axis=0).tolist())) for f in M.TEST_FAMILIES},
           "family_distance_medians": {g: {f: float(np.median(distance[g][family == f])) for f in M.TEST_FAMILIES} for g in GROUPS},
           "correlations": []}
    scatter = {}
    for label, _ in NOISE_LEVELS:
        for k in U.SENSOR_COUNTS:
            m = n_s == k
            for g in GROUPS:
                adv = saved[f"C__{label}__{g}__psnr"] - saved[f"C__{label}__tikhonov__psnr"]
                row = {"label": label, "sensors": k, "group": g,
                       "pooled": spearman_ci(distance[g][m], adv[m]),
                       "within_family": {f: spearman_ci(distance[g][m & (family == f)], adv[m & (family == f)])
                                         for f in M.TEST_FAMILIES},
                       "by_feature": {name: float(spearmanr(features[m, j], adv[m])[0]) for j, name in enumerate(FEATURES)},
                       "family_mean_advantage": {f: float(adv[m & (family == f)].mean()) for f in M.TEST_FAMILIES}}
                out["correlations"].append(row)
                if label == "high":
                    scatter[(g, k)] = (distance[g][m], adv[m], family[m])
    return out, scatter


def sweep_images():
    sensors = np.array([U.SENSOR_COUNTS[i % 2] for i in range(N_SWEEP)], np.int32)
    discs = [shape_phantom("discs", U.GRID_SIZE, DISC_SEED_OFFSET + i) for i in range(N_SWEEP)]

    def blurred(img, sigma):
        out = ndimage.gaussian_filter(img.astype(np.float64), sigma) if sigma > 0 else img
        return (out / out.max()).astype(np.float32)
    return {
        "vessel_width": {w: np.stack([vessel_phantom(U.GRID_SIZE, VESSEL_SEED_OFFSET + i, w) for i in range(N_SWEEP)])
                         for w in VESSEL_WIDTHS},
        "disc_blur": {b: np.stack([blurred(d, b) for d in discs]) for b in DISC_BLURS},
    }, sensors


def sweep_analysis(references, t0):
    setup = M.Setup()
    models = {}
    for s in M.SEEDS:
        models[f"{M.BASELINE}_seed{s}"], _ = U.load_model(f"{U.CKPT_DIR}/noise_seed{s}.pt", setup.device)
        models[f"mixed160_seed{s}"], _ = U.load_model(f"{M.CKPT_DIR}/mixed160_seed{s}.pt", setup.device)
    with open("report/mixed_phantom_results.json") as f:
        mu = json.load(f)["tikhonov_weights_selected_on_mixed160"]
    sweeps, n_s = sweep_images()
    rows = []
    for sweep, levels in sweeps.items():
        for value, images in levels.items():
            features = feature_matrix(images)
            clean = setup.recordings({"phantom": images, "n_sensors": n_s})
            for label, rel_std in SWEEP_LEVELS:
                Y = noisy(clean, rel_std, SWEEP_NOISE_SEED)
                outputs = U.reconstruct(models, setup.physics, Y, n_s)
                tik = M.tikhonov(setup, Y, n_s, lambda k: mu[f"{label}|{k}"])
                tik_psnr = np.array([psnr(tik[i], images[i]) for i in range(len(images))])
                net_psnr = {g: np.mean([[psnr(outputs[f"{g}_seed{s}"][i], images[i]) for i in range(len(images))]
                                        for s in M.SEEDS], axis=0) for g in GROUPS}
                for k in U.SENSOR_COUNTS:
                    m = n_s == k
                    rows.append({"sweep": sweep, "value": value, "label": label, "sensors": k, "n_images": int(m.sum()),
                                 "thickness": float(features[m, FEATURES.index("thickness")].mean()),
                                 "max_gradient": float(features[m, FEATURES.index("max_gradient")].mean()),
                                 "distance": {g: float(np.median(references[g].distance(features[m]))) for g in GROUPS},
                                 "tikhonov_psnr": bootstrap_mean_ci(tik_psnr[m]),
                                 **{g: {"psnr": bootstrap_mean_ci(net_psnr[g][m]),
                                        "minus_tikhonov": paired_bootstrap_ci(net_psnr[g][m], tik_psnr[m])} for g in GROUPS}})
            print(f"[{time.time() - t0:.0f}s] sweep {sweep} = {value} done", flush=True)
    return rows


def main(out_dir="report", overwrite=False):
    t0 = time.time()
    out_paths = [ensure_writable(os.path.join(out_dir, n), overwrite=overwrite) for n in OUTPUT_NAMES]
    references = training_references()
    distance, scatter = distance_analysis(references)
    print(f"[{time.time() - t0:.0f}s] distance analysis done", flush=True)
    results = {"protocol": __doc__.split("Outputs:")[0].strip(), "features": list(FEATURES), "distance": distance,
               "sweeps": sweep_analysis(references, t0)}
    os.makedirs(out_dir, exist_ok=True)
    with open(out_paths[0], "w") as f:
        json.dump(results, f, indent=2)
    write_text(results, out_paths[1])
    plot_distance(scatter, distance, out_paths[2])
    plot_sweeps(results["sweeps"], out_paths[3])
    print(f"[{time.time() - t0:.0f}s] wrote {out_dir}/prior_shift_*")


def _ci(t, d=2):
    return f"{t[0]:+.{d}f} [{t[1]:+.{d}f}, {t[2]:+.{d}f}]"


def write_text(r, path):
    d = r["distance"]
    L = ["Prior-shift analysis (scripts/prior_shift_analysis.py)", "", r["protocol"], "",
         "== 1. Distance from the training distribution and advantage over Tikhonov (shape test set)", "",
         "Mean structural statistics by test family:"]
    for fam, means in d["family_feature_means"].items():
        L.append(f"  {fam:10s} " + ", ".join(f"{k} {v:.3g}" for k, v in means.items()))
    L.append("Median distance from the training set, by test family:")
    for g, by in d["family_distance_medians"].items():
        L.append(f"  {g + '-trained':17s} " + ", ".join(f"{f} {v:.1f}" for f, v in by.items()))
    L += ["", "Spearman correlation of distance with the PSNR advantage over Tikhonov, [95 % CI]:"]
    for row in d["correlations"]:
        L.append(f"  noise {row['label']:9s} {row['sensors']:2d} sensors, {row['group']}-trained: pooled {_ci(row['pooled'])} | within family: "
                 + ", ".join(f"{f} {v[0]:+.2f}" for f, v in row["within_family"].items()))
    L += ["", "Spearman correlation of each statistic with the advantage (noise high):"]
    for row in d["correlations"]:
        if row["label"] == "high":
            L.append(f"  {row['sensors']:2d} sensors, {row['group']}-trained: " + ", ".join(f"{k} {v:+.2f}" for k, v in row["by_feature"].items()))
    L += ["", "== 2. Controlled shifts (20 images per row; PSNR in dB; difference to Tikhonov with 95 % CI)"]
    for row in r["sweeps"]:
        L.append(f"  {row['sweep']} {row['value']:g}, noise {row['label']:9s} {row['sensors']:2d} sensors (thickness {row['thickness']:.1f} px, "
                 f"max gradient {row['max_gradient']:.2f}): Tikhonov {row['tikhonov_psnr'][0]:.2f} | "
                 + " | ".join(f"{g}-trained {row[g]['psnr'][0]:.2f}, {_ci(row[g]['minus_tikhonov'])}, distance {row['distance'][g]:.1f}"
                              for g in GROUPS))
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[-110:]))


NAMES = {M.BASELINE: "trained on blobs", "mixed160": "trained on 4 families (160 images)"}


def plot_distance(scatter, distance, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401

    colours = dict(zip(M.TEST_FAMILIES, ("C0", "C1", "C2", "C3", "C4")))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey="col")
    rho = {(row["group"], row["sensors"]): row["pooled"] for row in distance["correlations"] if row["label"] == "high"}
    for i, g in enumerate(GROUPS):
        for j, k in enumerate(U.SENSOR_COUNTS):
            ax = axes[i, j]
            dist, adv, fam = scatter[(g, k)]
            for f in M.TEST_FAMILIES:
                sel = fam == f
                ax.scatter(dist[sel], adv[sel], s=14, color=colours[f], alpha=0.75,
                           label=f + (" (unseen)" if f == M.UNSEEN_FAMILY else ""))
            ax.axhline(0, color="k", lw=0.8)
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_xticks([t for t in (1, 2, 5, 10, 20) if dist.min() * 0.8 <= t <= dist.max() * 1.2])
            r = rho[(g, k)]
            ax.set_title(f"{NAMES[g]}, {k} sensors: Spearman {r[0]:+.2f} [{r[1]:+.2f}, {r[2]:+.2f}]", fontsize=9)
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel("PSNR minus Tikhonov [dB]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="lower center", ncol=5, frameon=False)
    fig.supxlabel("distance of the test image from that network's training images (Mahalanobis, log scale)", fontsize=9, y=0.045)
    fig.suptitle("Advantage over Tikhonov against distance from the training distribution (14 dB SNR, one point per test image)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_sweeps(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = {"vessel_width": ("vessel profile width [px]", "Vessel width"), "disc_blur": ("Gaussian blur of the disc edges [px]", "Edge blur of discs")}
    colours = {M.BASELINE: "C3", "mixed160": "C0"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharey="row")
    for i, sweep in enumerate(titles):
        for j, k in enumerate(U.SENSOR_COUNTS):
            ax = axes[i, j]
            for label, style in (("high", "-"), ("noiseless", ":")):
                sel = [r for r in rows if r["sweep"] == sweep and r["sensors"] == k and r["label"] == label]
                x = [r["value"] for r in sel]
                for g in GROUPS:
                    d = np.array([r[g]["minus_tikhonov"] for r in sel])
                    ax.errorbar(x, d[:, 0], yerr=[d[:, 0] - d[:, 1], d[:, 2] - d[:, 0]], color=colours[g], ls=style, marker="o",
                                ms=4, capsize=2, label=f"{NAMES[g]}, {'14 dB SNR' if label == 'high' else 'no noise'}")
            ax.axhline(0, color="k", lw=0.8)
            ax.set_title(f"{titles[sweep][1]}, {k} sensors", fontsize=10)
            ax.set_xlabel(titles[sweep][0])
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel("PSNR minus Tikhonov [dB]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Controlled shifts of one image property (20 images per point; 95 % CI)", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main(overwrite="--overwrite" in sys.argv)
