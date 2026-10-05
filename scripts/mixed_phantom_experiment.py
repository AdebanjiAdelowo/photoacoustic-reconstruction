"""Does the unrolled Tikhonov network keep an advantage over Tikhonov with a less restrictive prior?

The networks of scripts/unrolled_evaluation.py were trained on 1 to 3 smooth Gaussian blobs and lose
their lead over Tikhonov on sharp-edged discs. Here the same architecture and training recipe are
used with a broader training distribution, and everything is scored against the blob-trained
networks, which remain the baseline. Nothing existing is retrained, regenerated or overwritten.

Protocol (fixed before any result of this experiment was seen):
  * Training distribution: four families in equal parts (src/shape_phantoms.py): smooth blobs,
    sharp-edged discs, sharp-edged rotated ellipses and sharp-edged rotated rectangles. Sensor counts
    are balanced within each family. Phantom seeds 200000+ (training) and 210000+ (validation).
  * Two mixed training sets, 5 networks each (seeds 0-4), both with the noise-augmented recipe of
    unrolled_evaluation.py (noise level drawn at random per recording and epoch):
      - "mixed40":  40 training and 8 validation images, 200 epochs: the data budget of the
                    baseline, so each family has a quarter of the baseline's blob count;
      - "mixed160": 160 training and 32 validation images, 50 epochs: as many images per family as
                    the baseline has blobs, and the same 1000 optimiser steps.
    The mixed40 images are the first 40 of the mixed160 images.
  * Baseline: the 5 blob-trained, noise-trained networks (experiments/unrolled/noise_seed*.pt).
  * Test sets:
      A. the 400-image blob test set with its noise draws, unchanged (cost of leaving the
         specialised prior), and its sound-speed mismatch data;
      B. the three families of scripts/unrolled_generalisation_check.py, unchanged (20 images each);
      C. new: 100 images (50 per sensor count) from each of the four training families and from a
         fifth family that no network has seen, thin curved vessel-like lines. Phantom seeds
         300000+, all five noise levels, noise seeds independent of training.
  * Tikhonov: on A and B the published weights (selected on the blob training split). On C the
    weight is selected per noise level and sensor count on the mixed160 training split, with the
    grid and criterion of scripts/tikhonov_evaluation.py, so that Tikhonov is tuned on the same
    distribution the mixed networks are trained on.
  * Uncertainty: 95 % bootstrap intervals over test images for the mean over the 5 networks;
    differences are paired image by image.

The main question is whether mixed training keeps a meaningful advantage over Tikhonov on C, in
particular on the unseen family, and what it costs on A.

Outputs: report/mixed_phantom_results.{txt,json}, report/mixed_phantom_per_image.npz,
report/mixed_phantom_advantage.png. Checkpoints go to experiments/unrolled_mixed/ (gitignored).
Requires the outputs of unrolled_evaluation.py and the scripts it depends on.

    python scripts/mixed_phantom_experiment.py
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import scripts.unrolled_evaluation as U  # noqa: E402
import scripts.unrolled_generalisation_check as G  # noqa: E402
from scripts.evaluate_noise_sensitivity import NOISE_LEVELS, NOISE_SEED, add_sensor_noise  # noqa: E402
from scripts.tikhonov_evaluation import MU_GRID, noisy  # noqa: E402
from src.evaluate import psnr, ssim  # noqa: E402
from src.forward_model import DEFAULT_SOUND_SPEED  # noqa: E402
from src.jax_cache import simulation_done  # noqa: E402
from src.run_safety import ensure_writable  # noqa: E402
from src.shape_phantoms import shape_phantom  # noqa: E402
from src.stats import bootstrap_mean_ci, paired_bootstrap_ci  # noqa: E402

TRAIN_FAMILIES = ("blobs", "discs", "ellipses", "rectangles")
UNSEEN_FAMILY = "vessels"
TEST_FAMILIES = TRAIN_FAMILIES + (UNSEEN_FAMILY,)
MIXED_SETS = {"mixed40": {"n_train": 40, "n_val": 8, "epochs": 200},
              "mixed160": {"n_train": 160, "n_val": 32, "epochs": 50}}
SEEDS = U.TRAIN_SEEDS
TRAIN_SEED_OFFSET, VAL_SEED_OFFSET, TEST_SEED_OFFSET = 200_000, 210_000, 300_000
N_TEST_PER_FAMILY = 100
SHAPE_NOISE_SEED = 868_000        # test noise on set C
TIKHONOV_NOISE_SEED = 778_000     # training-split noise for the Tikhonov weight selection on set C
CKPT_DIR = "experiments/unrolled_mixed"
BASELINE = "blob"
NETWORK_GROUPS = (BASELINE,) + tuple(MIXED_SETS)
OUTPUT_NAMES = ("mixed_phantom_results.json", "mixed_phantom_per_image.npz", "mixed_phantom_results.txt",
                "mixed_phantom_advantage.png")
METRICS = (("psnr", psnr), ("ssim", ssim))


def mixed_split(offset, n):
    """n images cycling through the training families; sensor counts balanced within each family."""
    fam = [TRAIN_FAMILIES[i % len(TRAIN_FAMILIES)] for i in range(n)]
    n_sensors = np.array([U.SENSOR_COUNTS[(i // len(TRAIN_FAMILIES)) % 2] for i in range(n)], np.int32)
    seeds = offset + np.arange(n, dtype=np.int64)
    phantoms = np.stack([shape_phantom(f, U.GRID_SIZE, int(s)) for f, s in zip(fam, seeds)])
    return {"phantom": phantoms, "n_sensors": n_sensors, "seed": seeds, "family": np.array(fam)}


def shape_test_set():
    """Set C: N_TEST_PER_FAMILY images per family, sensor counts alternating."""
    fam, seeds = [], []
    for f_index, f in enumerate(TEST_FAMILIES):
        fam += [f] * N_TEST_PER_FAMILY
        seeds += list(TEST_SEED_OFFSET + 1000 * f_index + np.arange(N_TEST_PER_FAMILY))
    n_sensors = np.array([U.SENSOR_COUNTS[i % 2] for i in range(len(fam))], np.int32)
    phantoms = np.stack([shape_phantom(f, U.GRID_SIZE, int(s)) for f, s in zip(fam, seeds)])
    return {"phantom": phantoms, "n_sensors": n_sensors, "seed": np.array(seeds, np.int64), "family": np.array(fam)}


class Setup:
    def __init__(self):
        self.device = U.get_device("cpu")
        self.physics = U.Physics(self.device)
        self.domain, self.medium = U.build_domain_and_medium(U.GRID_SIZE)
        self.sensors = {k: U.sparse_view_sensor_array(k, U.RADIUS, U.CENTRE) for k in U.SENSOR_COUNTS}

    def recordings(self, split):
        return [simulation_done(y) for y in
                U.recordings(split["phantom"], split["n_sensors"], self.domain, self.medium, self.sensors)]


def train_networks(setup, name, t0):
    cfg = MIXED_SETS[name]
    splits = {"train": mixed_split(TRAIN_SEED_OFFSET, cfg["n_train"]), "val": mixed_split(VAL_SEED_OFFSET, cfg["n_val"])}
    clean = None
    for s in SEEDS:
        path = f"{CKPT_DIR}/{name}_seed{s}.pt"
        if os.path.exists(path):
            continue
        if clean is None:
            clean = {n: setup.recordings(d) for n, d in splits.items()}
        h = U.train(setup.physics, clean, splits, "noise", s, path, epochs=cfg["epochs"])
        print(f"[{time.time() - t0:.0f}s] trained {name} seed {s}: best epoch {h['best_epoch']}, "
              f"val {h['best_val_loss']:.6f}", flush=True)


def score(per_image, key, outputs, phantoms):
    for name, img in outputs.items():
        for metric, f in METRICS:
            per_image[key + (name, metric)] = np.array([f(img[i], phantoms[i]) for i in range(len(phantoms))])
    for group in NETWORK_GROUPS:
        members = [n for n in outputs if n.startswith(group + "_seed")]
        if members:
            for metric, _ in METRICS:
                per_image[key + (group, metric)] = np.mean([per_image[key + (n, metric)] for n in members], axis=0)


def group_summary(per_image, key, group, metric, mask):
    seed_means = [per_image[key + (f"{group}_seed{s}", metric)][mask].mean() for s in SEEDS]
    return {"mean_ci95": bootstrap_mean_ci(per_image[key + (group, metric)][mask]),
            "seed_min": float(np.min(seed_means)), "seed_max": float(np.max(seed_means))}


def select_tikhonov_weights(setup, clean_train, split):
    """mu per noise level and sensor count, by mean PSNR on the mixed160 training split."""
    selection = {}
    for label, rel_std in NOISE_LEVELS:
        Y = noisy(clean_train, rel_std, TIKHONOV_NOISE_SEED)
        for k in U.SENSOR_COUNTS:
            m = np.nonzero(split["n_sensors"] == k)[0]
            scores = {}
            for mu in MU_GRID:
                est = setup.physics.solvers[k].solve(np.stack([Y[i] for i in m]), mu).astype(np.float32)
                scores[mu] = float(np.mean([psnr(e.reshape(U.GRID_SIZE, U.GRID_SIZE), split["phantom"][i])
                                            for e, i in zip(est, m)]))
            selection[(label, k)] = max(scores, key=scores.get)
    return selection


def tikhonov(setup, Y, n_sensors, mu_of):
    out = np.zeros((len(Y), U.GRID_SIZE, U.GRID_SIZE), np.float32)
    for k in U.SENSOR_COUNTS:
        m = np.nonzero(n_sensors == k)[0]
        out[m] = setup.physics.solvers[k].solve(np.stack([Y[i] for i in m]), mu_of(k)).reshape(len(m), U.GRID_SIZE, U.GRID_SIZE)
    return out


def main(out_dir="report", overwrite=False, train_only=None):
    t0 = time.time()
    setup = Setup()
    if train_only:
        train_networks(setup, train_only, t0)
        return
    out_paths = [ensure_writable(os.path.join(out_dir, n), device=setup.device, overwrite=overwrite) for n in OUTPUT_NAMES]
    for name in MIXED_SETS:
        train_networks(setup, name, t0)

    models, networks = {}, {}
    for s in SEEDS:
        models[f"{BASELINE}_seed{s}"], _ = U.load_model(f"{U.CKPT_DIR}/noise_seed{s}.pt", setup.device)
        for name in MIXED_SETS:
            models[f"{name}_seed{s}"], ckpt = U.load_model(f"{CKPT_DIR}/{name}_seed{s}.pt", setup.device)
            networks[f"{name}_seed{s}"] = {"epoch": int(ckpt["epoch"]), "val_loss": float(ckpt["val_loss"])}
    mixed_models = {n: m for n, m in models.items() if not n.startswith(BASELINE)}
    per_image = {}
    results = {"protocol": __doc__.split("Outputs:")[0].strip(), "mixed_sets": MIXED_SETS, "networks": networks}

    # --- A: the blob test set and its mismatch data; baseline and Tikhonov scores are the saved ones ---
    test = np.load(U.TEST_PATH)
    phantoms, n_s = test["phantom"], test["n_sensors"]
    saved_unrolled = np.load("report/unrolled_per_image.npz")
    saved_tik, saved_mis = np.load("report/tikhonov_per_image.npz"), np.load("report/model_mismatch_per_image.npz")
    clean = setup.recordings(test)
    results["blob_test"] = []
    for label, rel_std in NOISE_LEVELS:
        key = ("A", label)
        score(per_image, key, U.reconstruct(mixed_models, setup.physics, noisy(clean, rel_std, NOISE_SEED), n_s), phantoms)
        for k in U.SENSOR_COUNTS:
            m = n_s == k
            row = {"label": label, "sensors": k}
            for metric, _ in METRICS:
                base = saved_unrolled[f"{label}__unrolled_noise__{metric}"][m]
                tik = saved_tik[f"{label}__tikhonov_per_level__{metric}"][m]
                e = {"tikhonov": float(tik.mean()), BASELINE: float(base.mean())}
                for name in MIXED_SETS:
                    v = per_image[key + (name, metric)][m]
                    e[name] = {**group_summary(per_image, key, name, metric, m),
                               "minus_tikhonov": paired_bootstrap_ci(v, tik), "minus_blob_trained": paired_bootstrap_ci(v, base)}
                row[metric] = e
            results["blob_test"].append(row)
        print(f"[{time.time() - t0:.0f}s] A: blob test, noise {label} done", flush=True)

    from jwave.geometry import TimeAxis

    from scripts.model_mismatch_evaluation import simulate

    time_axis = TimeAxis.from_medium(setup.medium, cfl=0.3)
    results["blob_test_mismatch"] = []
    for delta in U.MISMATCH_DELTAS:
        _, medium_true = U.build_domain_and_medium(U.GRID_SIZE, sound_speed=DEFAULT_SOUND_SPEED * (1 + delta))
        clean_mis = [simulation_done(simulate(phantoms[i], setup.domain, medium_true, time_axis, setup.sensors[int(n_s[i])]))
                     for i in range(len(phantoms))]
        for label, rel_std in U.MISMATCH_LEVELS:
            rng = np.random.default_rng(NOISE_SEED + int(round(rel_std * 100000)))
            Y = [add_sensor_noise(c, rel_std, rng)[..., 0].reshape(-1) for c in clean_mis]
            key = ("A", f"{delta:+.2f}", label)
            score(per_image, key, U.reconstruct(mixed_models, setup.physics, Y, n_s), phantoms)
            for k in U.SENSOR_COUNTS:
                m = n_s == k
                row = {"delta": delta, "label": label, "sensors": k}
                for metric, _ in METRICS:
                    base = saved_unrolled[f"{delta:+.2f}__{label}__unrolled_noise__{metric}"][m]
                    tik = saved_mis[f"{delta:+.2f}__{label}__tikhonov__{metric}"][m]
                    e = {"tikhonov": float(tik.mean()), BASELINE: float(base.mean())}
                    for name in MIXED_SETS:
                        v = per_image[key + (name, metric)][m]
                        e[name] = {**group_summary(per_image, key, name, metric, m),
                                   "minus_tikhonov": paired_bootstrap_ci(v, tik), "minus_blob_trained": paired_bootstrap_ci(v, base)}
                    row[metric] = e
                results["blob_test_mismatch"].append(row)
        print(f"[{time.time() - t0:.0f}s] A: mismatch {delta:+.0%} done", flush=True)

    # --- B: the families of the generalisation check, unchanged; published Tikhonov weights ---
    with open("report/tikhonov_results.json") as f:
        published_mu = json.load(f)["selection"]
    families, n_g = G.build_families()
    results["generalisation_check"] = []
    for fam, images in families.items():
        images = np.stack(images)
        clean_g = setup.recordings({"phantom": images, "n_sensors": n_g})
        for label, rel_std in G.LEVELS:
            Y = G.noisy_recordings(clean_g, rel_std)
            key = ("B", fam, label)
            outputs = U.reconstruct(models, setup.physics, Y, n_g)
            outputs["tikhonov"] = tikhonov(setup, Y, n_g, lambda k: published_mu[f"{label}|{k}"]["selected_mu"])
            score(per_image, key, outputs, images)
            for k in U.SENSOR_COUNTS:
                m = n_g == k
                results["generalisation_check"].append({
                    "family": fam, "label": label, "sensors": k,
                    **{metric: {name: float(per_image[key + (name, metric)][m].mean()) for name in ("tikhonov",) + NETWORK_GROUPS}
                       for metric, _ in METRICS}})
    print(f"[{time.time() - t0:.0f}s] B: generalisation families done", flush=True)

    # --- C: new shape test set; Tikhonov tuned on the mixed160 training split ---
    train160 = mixed_split(TRAIN_SEED_OFFSET, MIXED_SETS["mixed160"]["n_train"])
    selection = select_tikhonov_weights(setup, setup.recordings(train160), train160)
    results["tikhonov_weights_selected_on_mixed160"] = {f"{l}|{k}": v for (l, k), v in selection.items()}
    results["tikhonov_weights_published_blob_split"] = {key: v["selected_mu"] for key, v in published_mu.items()}
    shape_test = shape_test_set()
    used = set(train160["seed"].tolist()) | set(mixed_split(VAL_SEED_OFFSET, 32)["seed"].tolist())
    assert not used & set(shape_test["seed"].tolist()), "test seeds overlap the mixed training or validation seeds"
    clean_c = setup.recordings(shape_test)
    results["shape_test"] = []
    for label, rel_std in NOISE_LEVELS:
        Y = noisy(clean_c, rel_std, SHAPE_NOISE_SEED)
        key = ("C", label)
        outputs = U.reconstruct(models, setup.physics, Y, shape_test["n_sensors"])
        outputs["tikhonov"] = tikhonov(setup, Y, shape_test["n_sensors"], lambda k: selection[(label, k)])
        score(per_image, key, outputs, shape_test["phantom"])
        for fam in TEST_FAMILIES:
            for k in U.SENSOR_COUNTS:
                m = (shape_test["family"] == fam) & (shape_test["n_sensors"] == k)
                row = {"family": fam, "seen_in_mixed_training": fam != UNSEEN_FAMILY, "label": label, "sensors": k,
                       "n_images": int(m.sum())}
                for metric, _ in METRICS:
                    tik = per_image[key + ("tikhonov", metric)][m]
                    e = {"tikhonov": bootstrap_mean_ci(tik)}
                    for group in NETWORK_GROUPS:
                        v = per_image[key + (group, metric)][m]
                        e[group] = {**group_summary(per_image, key, group, metric, m), "minus_tikhonov": paired_bootstrap_ci(v, tik)}
                        if group != BASELINE:
                            e[group]["minus_blob_trained"] = paired_bootstrap_ci(v, per_image[key + (BASELINE, metric)][m])
                    row[metric] = e
                results["shape_test"].append(row)
        print(f"[{time.time() - t0:.0f}s] C: shape test, noise {label} done", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    with open(out_paths[0], "w") as f:
        json.dump(results, f, indent=2)
    np.savez_compressed(out_paths[1], shape_test_seed=shape_test["seed"], shape_test_n_sensors=shape_test["n_sensors"],
                        shape_test_family=shape_test["family"], **{"__".join(key): v for key, v in per_image.items()})
    write_text(results, out_paths[2])
    plot(results, out_paths[3])
    print(f"[{time.time() - t0:.0f}s] wrote {out_dir}/mixed_phantom_*")


def _ci(t, d):
    return f"{t[0]:+.{d}f} [{t[1]:+.{d}f}, {t[2]:+.{d}f}]"


def write_text(r, path):
    L = ["Mixed-phantom training of the unrolled Tikhonov network (scripts/mixed_phantom_experiment.py)", "", r["protocol"], "",
         "Networks (best epoch, validation loss): " + ", ".join(f"{k} {v['epoch']} {v['val_loss']:.6f}" for k, v in r["networks"].items()),
         "Tikhonov mu selected on the mixed160 training split: "
         + ", ".join(f"{k} {v:.0e}" for k, v in r["tikhonov_weights_selected_on_mixed160"].items()),
         "Tikhonov mu published (blob training split):       "
         + ", ".join(f"{k} {v:.0e}" for k, v in r["tikhonov_weights_published_blob_split"].items()), "",
         "== C. New shape test set (50 images per row). Mean [95 % CI]; 'vs Tik' is the paired difference to Tikhonov."]
    for metric, d in (("psnr", 2), ("ssim", 3)):
        L.append(f"-- {metric.upper()}")
        for row in r["shape_test"]:
            e = row[metric]
            seen = "" if row["seen_in_mixed_training"] else " (unseen by every network)"
            L.append(f"  {row['family']}{seen}, noise {row['label']}, {row['sensors']} sensors: Tikhonov {e['tikhonov'][0]:.{d}f}")
            for g in NETWORK_GROUPS:
                x = e[g]
                extra = f" | vs blob-trained {_ci(x['minus_blob_trained'], d)}" if "minus_blob_trained" in x else ""
                L.append(f"      {g + '-trained':17s} {x['mean_ci95'][0]:.{d}f} (networks {x['seed_min']:.{d}f}-{x['seed_max']:.{d}f}) "
                         f"| vs Tik {_ci(x['minus_tikhonov'], d)}{extra}")
    L += ["", "== A. Blob test set (200 images per row), unchanged. Tikhonov and blob-trained values are the published ones."]
    for metric, d in (("psnr", 2), ("ssim", 3)):
        L.append(f"-- {metric.upper()}")
        for row in r["blob_test"]:
            e = row[metric]
            L.append(f"  noise {row['label']}, {row['sensors']} sensors: Tikhonov {e['tikhonov']:.{d}f} | blob-trained {e[BASELINE]:.{d}f}")
            for g in MIXED_SETS:
                x = e[g]
                L.append(f"      {g}-trained {x['mean_ci95'][0]:.{d}f} (networks {x['seed_min']:.{d}f}-{x['seed_max']:.{d}f}) "
                         f"| vs Tik {_ci(x['minus_tikhonov'], d)} | vs blob-trained {_ci(x['minus_blob_trained'], d)}")
    L += ["", "== A. Blob test set under sound-speed mismatch (200 images per row), PSNR"]
    for row in r["blob_test_mismatch"]:
        e = row["psnr"]
        L.append(f"  c +{row['delta']:.0%}, noise {row['label']}, {row['sensors']} sensors: Tikhonov {e['tikhonov']:.2f} | blob-trained {e[BASELINE]:.2f} | "
                 + " | ".join(f"{g}-trained {e[g]['mean_ci95'][0]:.2f} (vs Tik {_ci(e[g]['minus_tikhonov'], 2)})" for g in MIXED_SETS))
    L += ["", "== B. Families of the generalisation check (10 images per row), PSNR / SSIM, published Tikhonov weights"]
    for row in r["generalisation_check"]:
        L.append(f"  {row['family']}, noise {row['label']}, {row['sensors']} sensors: "
                 + " | ".join(f"{n} {row['psnr'][n]:.1f} / {row['ssim'][n]:.3f}" for n in ("tikhonov",) + NETWORK_GROUPS))
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[:120]))


def plot(r, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ("noiseless", "high")
    fig, axes = plt.subplots(len(labels), 2, figsize=(11, 7), sharey="row")
    colours = {BASELINE: "C3", "mixed40": "C9", "mixed160": "C0"}
    names = {BASELINE: "trained on blobs (baseline)", "mixed40": "trained on 4 families, 40 images",
             "mixed160": "trained on 4 families, 160 images"}
    x = np.arange(len(TEST_FAMILIES))
    width = 0.26
    for i, label in enumerate(labels):
        for j, k in enumerate(U.SENSOR_COUNTS):
            ax = axes[i, j]
            rows = {row["family"]: row["psnr"] for row in r["shape_test"] if row["label"] == label and row["sensors"] == k}
            for g_index, g in enumerate(NETWORK_GROUPS):
                d = np.array([rows[f][g]["minus_tikhonov"] for f in TEST_FAMILIES])
                ax.bar(x + (g_index - 1) * width, d[:, 0], width, color=colours[g], label=names[g],
                       yerr=[d[:, 0] - d[:, 1], d[:, 2] - d[:, 0]], capsize=2, error_kw={"lw": 0.8})
            ax.axhline(0, color="k", lw=0.8)
            ax.set_xticks(x, [f + ("\n(unseen)" if f == UNSEEN_FAMILY else "") for f in TEST_FAMILIES], fontsize=8)
            ax.set_title(f"{k} sensors, noise {label}", fontsize=10)
            ax.grid(alpha=0.3, axis="y")
            if j == 0:
                ax.set_ylabel("PSNR minus Tikhonov [dB]")
    handles, legend = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend, fontsize=8, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Unrolled network against Tikhonov on new test images, by image family (50 images per bar; 95 % CI)", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="report")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--train-only", choices=tuple(MIXED_SETS), default=None,
                        help="train the 5 networks of one mixed set and exit (lets the two sets train in parallel)")
    args = parser.parse_args()
    main(args.out_dir, args.overwrite, args.train_only)
