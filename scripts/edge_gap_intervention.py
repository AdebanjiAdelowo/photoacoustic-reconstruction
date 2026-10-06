"""Intervention: does filling the edge-sharpness gap in the training images repair the failure?

Background. scripts/prior_shift_analysis.py found that the networks trained on four phantom families
(mixed160) lose most of their advantage over Tikhonov on discs whose edges are blurred by 0.5 to 1
pixel, and on thin vessels. The explanation offered there, after seeing the result, was that the
training images have either one-pixel edges or smooth blob profiles and nothing in between. This
experiment tests that explanation by changing the training distribution and nothing else.

Design (written and committed before any network of this experiment was trained):

  * Control: the 5 existing mixed160 networks (experiments/unrolled_mixed/mixed160_seed*.pt).
  * Intervention ("edgefill160"): the same 160 training and 32 validation images, with the same
    seeds, families, shapes, positions and sensor counts, except that half of the images of each
    sharp-edged family (discs, ellipses, rectangles; balanced over both sensor counts) are replaced
    by the same image with its edges blurred by a Gaussian whose standard deviation is drawn
    uniformly from 0.25 to 1.5 pixels. Images are replaced, not added: 60 of the 160 training
    images change, 60 sharp images and the 40 blob images stay as they are.
  * Everything else is identical to mixed160: architecture, noise-augmented training recipe, Adam
    1e-3, batch 8, 50 epochs, seeds 0 to 4, sensor geometry, checkpoint selection, CPU.
  * Tikhonov is untouched (weights selected on the mixed160 training split).

Evaluation, on existing test sets only, at 14 dB SNR unless stated (PSNR advantage over Tikhonov,
mean of 5 networks, per sensor count):

  Primary     the whole disc-blur sweep of prior_shift_analysis.py (0, 0.5, 1, 2, 3, 4 pixels; the
              same 40 images and noise). The test discs are new images, and blur was introduced in
              training for all three sharp families, not for discs alone.
  Secondary   the vessel-width sweep (thin vessels are a transfer prediction, not a training target).
  Cost        the shape test set and the blob test set of mixed_phantom_experiment.py.

Prediction. Filling the gap reduces or removes the dip at 0.5 to 1 pixel of blur, and may improve
thin vessels.

Success criteria, fixed in advance, each evaluated at 16 and at 64 sensors:

  S1 (valley)   the worst advantage over the blurs 0.5 and 1 pixel improves by at least 3 dB over
                mixed160, and the paired improvement at 1 pixel has a 95 % interval above zero.
                (The mixed160 valley is about 5 to 6 dB below its level at 2 pixels of blur, so
                3 dB is more than half of it.)
  S2 (no harm)  the loss relative to mixed160 is at most 2 dB on sharp discs (blur 0) and on each
                sharp family of the shape test set, and at most 1 dB at blurs of 2, 3 and 4 pixels.
  S3 (transfer) the advantage on the two thinnest vessels (widths 1 and 1.5 pixels) improves by at
                least 1 dB on average.

  Verdict: "supported" if S1 and S2 hold at both sensor counts; "partly supported" if S1 holds at
  both but S2 fails, or S1 holds at one sensor count; "not supported" if S1 fails at both. An
  improvement of less than 1 dB in the valley at both sensor counts counts as a refutation of the
  edge-sharpness-gap explanation. S3 is reported separately and does not enter the verdict.

Outputs: report/edge_gap_intervention_results.{txt,json}, report/edge_gap_intervention.png.
Requires the outputs of prior_shift_analysis.py and mixed_phantom_experiment.py.

    python scripts/edge_gap_intervention.py [--train-seeds 0,1,2]
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

import scripts.mixed_phantom_experiment as M  # noqa: E402
import scripts.prior_shift_analysis as P  # noqa: E402
import scripts.unrolled_evaluation as U  # noqa: E402
from scripts.evaluate_noise_sensitivity import NOISE_LEVELS, NOISE_SEED  # noqa: E402
from scripts.tikhonov_evaluation import noisy  # noqa: E402
from src.evaluate import psnr  # noqa: E402
from src.run_safety import ensure_writable  # noqa: E402
from src.shape_phantoms import blur_edges  # noqa: E402
from src.stats import bootstrap_mean_ci, paired_bootstrap_ci  # noqa: E402

CONTROL, INTERVENTION = "mixed160", "edgefill160"
GROUPS = (CONTROL, INTERVENTION)
SHARP_FAMILIES = ("discs", "ellipses", "rectangles")
BLUR_RANGE = (0.25, 1.5)          # pixels
BLUR_SEED_OFFSET = 500_000
EPOCHS = M.MIXED_SETS[CONTROL]["epochs"]
VALLEY_BLURS, SMOOTH_BLURS, THIN_WIDTHS = (0.5, 1.0), (2.0, 3.0, 4.0), (1.0, 1.5)
S1_MIN_GAIN, S2_MAX_LOSS_SHARP, S2_MAX_LOSS_SMOOTH, S3_MIN_GAIN, REFUTATION_GAIN = 3.0, 2.0, 1.0, 1.0, 1.0
OUTPUT_NAMES = ("edge_gap_intervention_results.json", "edge_gap_intervention_results.txt", "edge_gap_intervention.png")


def is_replaced(index, family):
    """Half of each sharp family, balanced over the two sensor counts (see M.mixed_split)."""
    return family in SHARP_FAMILIES and (index // 8) % 2 == 1


def edgefill_split(offset, n):
    """M.mixed_split with half of the sharp-family images replaced by edge-blurred copies."""
    split = M.mixed_split(offset, n)
    phantoms, blur = split["phantom"].copy(), np.zeros(n)
    for i in range(n):
        if is_replaced(i, split["family"][i]):
            blur[i] = np.random.default_rng(BLUR_SEED_OFFSET + int(split["seed"][i])).uniform(*BLUR_RANGE)
            phantoms[i] = blur_edges(split["phantom"][i], blur[i])
    return {**split, "phantom": phantoms, "blur": blur}


def train_networks(setup, seeds, t0):
    cfg = M.MIXED_SETS[CONTROL]
    splits = {"train": edgefill_split(M.TRAIN_SEED_OFFSET, cfg["n_train"]), "val": edgefill_split(M.VAL_SEED_OFFSET, cfg["n_val"])}
    clean = None
    for s in seeds:
        path = f"{M.CKPT_DIR}/{INTERVENTION}_seed{s}.pt"
        if os.path.exists(path):
            continue
        if clean is None:
            clean = {n: setup.recordings(d) for n, d in splits.items()}
        h = U.train(setup.physics, clean, splits, "noise", s, path, epochs=EPOCHS)
        print(f"[{time.time() - t0:.0f}s] trained {INTERVENTION} seed {s}: best epoch {h['best_epoch']}, "
              f"val {h['best_val_loss']:.6f}", flush=True)


def group_psnr(outputs, group, images):
    return np.mean([[psnr(outputs[f"{group}_seed{s}"][i], images[i]) for i in range(len(images))] for s in M.SEEDS], axis=0)


def evaluate_criteria(sweeps, shape_rows):
    """The pre-registered criteria, at 14 dB SNR, per sensor count."""
    def adv(sweep, value, k, group):
        (row,) = [r for r in sweeps if r["sweep"] == sweep and r["value"] == value and r["sensors"] == k and r["label"] == "high"]
        return row[group]["minus_tikhonov"][0], row

    verdicts = {}
    for k in U.SENSOR_COUNTS:
        valley = {g: min(adv("disc_blur", b, k, g)[0] for b in VALLEY_BLURS) for g in GROUPS}
        gain = valley[INTERVENTION] - valley[CONTROL]
        paired_at_1 = adv("disc_blur", 1.0, k, CONTROL)[1]["intervention_minus_control"]
        s1 = gain >= S1_MIN_GAIN and paired_at_1[1] > 0
        loss_sharp = {"disc_blur_0": -adv("disc_blur", 0.0, k, CONTROL)[1]["intervention_minus_control"][0]}
        for row in shape_rows:
            if row["label"] == "high" and row["sensors"] == k and row["family"] in SHARP_FAMILIES:
                loss_sharp[row["family"]] = -row["intervention_minus_control"][0]
        loss_smooth = {f"disc_blur_{b:g}": -adv("disc_blur", b, k, CONTROL)[1]["intervention_minus_control"][0] for b in SMOOTH_BLURS}
        s2 = max(loss_sharp.values()) <= S2_MAX_LOSS_SHARP and max(loss_smooth.values()) <= S2_MAX_LOSS_SMOOTH
        thin_gain = float(np.mean([adv("vessel_width", w, k, INTERVENTION)[0] - adv("vessel_width", w, k, CONTROL)[0] for w in THIN_WIDTHS]))
        verdicts[str(k)] = {"valley_advantage": valley, "valley_gain": gain, "paired_gain_at_blur_1": paired_at_1,
                            "S1_valley": bool(s1), "loss_on_sharp": loss_sharp, "loss_on_smooth": loss_smooth,
                            "S2_no_harm": bool(s2), "thin_vessel_gain": thin_gain, "S3_transfer": bool(thin_gain >= S3_MIN_GAIN)}
    s1 = [v["S1_valley"] for v in verdicts.values()]
    s2 = [v["S2_no_harm"] for v in verdicts.values()]
    if all(s1) and all(s2):
        overall = "supported"
    elif any(s1):
        overall = "partly supported"
    else:
        overall = "not supported"
    refuted = all(v["valley_gain"] < REFUTATION_GAIN for v in verdicts.values())
    return {"by_sensors": verdicts, "verdict": overall, "edge_gap_explanation_refuted": bool(refuted)}


def main(out_dir="report", overwrite=False, train_seeds=None):
    t0 = time.time()
    setup = M.Setup()
    if train_seeds is not None:
        train_networks(setup, train_seeds, t0)
        return
    out_paths = [ensure_writable(os.path.join(out_dir, n), overwrite=overwrite) for n in OUTPUT_NAMES]
    train_networks(setup, M.SEEDS, t0)

    models, networks = {}, {}
    for s in M.SEEDS:
        for g in GROUPS:
            models[f"{g}_seed{s}"], ckpt = U.load_model(f"{M.CKPT_DIR}/{g}_seed{s}.pt", setup.device)
            networks[f"{g}_seed{s}"] = {"epoch": int(ckpt["epoch"]), "val_loss": float(ckpt["val_loss"])}
    intervention_models = {n: m for n, m in models.items() if n.startswith(INTERVENTION)}
    with open("report/mixed_phantom_results.json") as f:
        mu = json.load(f)["tikhonov_weights_selected_on_mixed160"]
    train = edgefill_split(M.TRAIN_SEED_OFFSET, M.MIXED_SETS[CONTROL]["n_train"])
    results = {"protocol": __doc__.split("Outputs:")[0].strip(), "networks": networks,
               "training_set": {"n_images": int(len(train["blur"])), "n_replaced": int((train["blur"] > 0).sum()),
                                "replaced_by_family": {f: int(((train["blur"] > 0) & (train["family"] == f)).sum()) for f in M.TRAIN_FAMILIES},
                                "blur_range_px": list(BLUR_RANGE), "blur_mean_px": float(train["blur"][train["blur"] > 0].mean())}}

    # --- sweeps of prior_shift_analysis.py: the same images and noise, both groups ---
    sweeps, n_s = P.sweep_images()
    rows = []
    for sweep, levels in sweeps.items():
        for value, images in levels.items():
            clean = setup.recordings({"phantom": images, "n_sensors": n_s})
            for label, rel_std in P.SWEEP_LEVELS:
                Y = noisy(clean, rel_std, P.SWEEP_NOISE_SEED)
                outputs = U.reconstruct(models, setup.physics, Y, n_s)
                tik = M.tikhonov(setup, Y, n_s, lambda k: mu[f"{label}|{k}"])
                tik_psnr = np.array([psnr(tik[i], images[i]) for i in range(len(images))])
                net = {g: group_psnr(outputs, g, images) for g in GROUPS}
                for k in U.SENSOR_COUNTS:
                    m = n_s == k
                    rows.append({"sweep": sweep, "value": value, "label": label, "sensors": k, "n_images": int(m.sum()),
                                 "tikhonov_psnr": float(tik_psnr[m].mean()),
                                 **{g: {"psnr": float(net[g][m].mean()), "minus_tikhonov": paired_bootstrap_ci(net[g][m], tik_psnr[m])}
                                    for g in GROUPS},
                                 "intervention_minus_control": paired_bootstrap_ci(net[INTERVENTION][m], net[CONTROL][m])})
            print(f"[{time.time() - t0:.0f}s] sweep {sweep} = {value} done", flush=True)
    results["sweeps"] = rows
    with open("report/prior_shift_results.json") as f:  # the control must reproduce the earlier analysis
        earlier = {(r["sweep"], r["value"], r["label"], r["sensors"]): r[CONTROL]["minus_tikhonov"][0] for r in json.load(f)["sweeps"]}
    results["control_max_abs_difference_from_prior_shift_analysis"] = float(max(
        abs(r[CONTROL]["minus_tikhonov"][0] - earlier[(r["sweep"], r["value"], r["label"], r["sensors"])]) for r in rows))

    # --- cost: shape test set (saved control and Tikhonov scores) and blob test set ---
    saved = np.load("report/mixed_phantom_per_image.npz")
    shape_test = M.shape_test_set()
    assert np.array_equal(saved["shape_test_seed"], shape_test["seed"])
    clean_c = setup.recordings(shape_test)
    results["shape_test"] = []
    for label, rel_std in NOISE_LEVELS:
        out = U.reconstruct(intervention_models, setup.physics, noisy(clean_c, rel_std, M.SHAPE_NOISE_SEED), shape_test["n_sensors"])
        new = group_psnr(out, INTERVENTION, shape_test["phantom"])
        for fam in M.TEST_FAMILIES:
            for k in U.SENSOR_COUNTS:
                m = (shape_test["family"] == fam) & (shape_test["n_sensors"] == k)
                tik, control = saved[f"C__{label}__tikhonov__psnr"][m], saved[f"C__{label}__{CONTROL}__psnr"][m]
                results["shape_test"].append({
                    "family": fam, "label": label, "sensors": k, "tikhonov_psnr": float(tik.mean()),
                    CONTROL: {"psnr": float(control.mean()), "minus_tikhonov": paired_bootstrap_ci(control, tik)},
                    INTERVENTION: {"psnr": float(new[m].mean()), "minus_tikhonov": paired_bootstrap_ci(new[m], tik)},
                    "intervention_minus_control": paired_bootstrap_ci(new[m], control)})
        print(f"[{time.time() - t0:.0f}s] shape test, noise {label} done", flush=True)

    test = np.load(U.TEST_PATH)
    clean_a = setup.recordings(test)
    saved_tik = np.load("report/tikhonov_per_image.npz")
    results["blob_test"] = []
    for label, rel_std in NOISE_LEVELS:
        out = U.reconstruct(intervention_models, setup.physics, noisy(clean_a, rel_std, NOISE_SEED), test["n_sensors"])
        new = group_psnr(out, INTERVENTION, test["phantom"])
        for k in U.SENSOR_COUNTS:
            m = test["n_sensors"] == k
            tik, control = saved_tik[f"{label}__tikhonov_per_level__psnr"][m], saved[f"A__{label}__{CONTROL}__psnr"][m]
            results["blob_test"].append({"label": label, "sensors": k, "tikhonov_psnr": float(tik.mean()),
                                         CONTROL: float(control.mean()), INTERVENTION: bootstrap_mean_ci(new[m]),
                                         "intervention_minus_control": paired_bootstrap_ci(new[m], control)})
    print(f"[{time.time() - t0:.0f}s] blob test done", flush=True)

    results["criteria"] = evaluate_criteria(rows, results["shape_test"])
    os.makedirs(out_dir, exist_ok=True)
    with open(out_paths[0], "w") as f:
        json.dump(results, f, indent=2)
    write_text(results, out_paths[1])
    plot(results, out_paths[2])
    print(f"[{time.time() - t0:.0f}s] wrote {out_dir}/edge_gap_intervention_*")


def _ci(t, d=2):
    return f"{t[0]:+.{d}f} [{t[1]:+.{d}f}, {t[2]:+.{d}f}]"


def write_text(r, path):
    c, ts = r["criteria"], r["training_set"]
    L = ["Edge-sharpness intervention (scripts/edge_gap_intervention.py)", "", r["protocol"], "",
         f"Training set: {ts['n_replaced']} of {ts['n_images']} images replaced by blurred copies ({ts['replaced_by_family']}), "
         f"mean blur {ts['blur_mean_px']:.2f} px",
         "Networks (best epoch, validation loss): " + ", ".join(f"{k} {v['epoch']} {v['val_loss']:.6f}" for k, v in r["networks"].items()),
         f"Control check: largest difference of the mixed160 sweep values from prior_shift_analysis.py: "
         f"{r['control_max_abs_difference_from_prior_shift_analysis']:.2e} dB", "",
         f"== VERDICT: {c['verdict']}; edge-gap explanation refuted: {c['edge_gap_explanation_refuted']}"]
    for k, v in c["by_sensors"].items():
        L += [f"  {k} sensors: S1 valley {v['S1_valley']} (worst advantage at blur 0.5 to 1 px: {CONTROL} {v['valley_advantage'][CONTROL]:+.2f}, "
              f"{INTERVENTION} {v['valley_advantage'][INTERVENTION]:+.2f}, gain {v['valley_gain']:+.2f} dB; paired gain at 1 px {_ci(v['paired_gain_at_blur_1'])})",
              f"              S2 no harm {v['S2_no_harm']} (loss on sharp: " + ", ".join(f"{n} {x:+.2f}" for n, x in v["loss_on_sharp"].items())
              + "; loss on smooth: " + ", ".join(f"{n} {x:+.2f}" for n, x in v["loss_on_smooth"].items()) + ")",
              f"              S3 transfer {v['S3_transfer']} (mean gain on vessel widths 1 and 1.5 px: {v['thin_vessel_gain']:+.2f} dB)"]
    L += ["", "== Sweeps (20 images per row): PSNR advantage over Tikhonov, and intervention minus control"]
    for row in r["sweeps"]:
        L.append(f"  {row['sweep']} {row['value']:g}, noise {row['label']:9s} {row['sensors']:2d} sensors: Tikhonov {row['tikhonov_psnr']:.2f} | "
                 f"{CONTROL} {_ci(row[CONTROL]['minus_tikhonov'])} | {INTERVENTION} {_ci(row[INTERVENTION]['minus_tikhonov'])} | "
                 f"difference {_ci(row['intervention_minus_control'])}")
    L += ["", "== Shape test set (50 images per row), PSNR advantage over Tikhonov"]
    for row in r["shape_test"]:
        L.append(f"  {row['family']:10s} noise {row['label']:9s} {row['sensors']:2d} sensors: {CONTROL} {row[CONTROL]['minus_tikhonov'][0]:+.2f} | "
                 f"{INTERVENTION} {row[INTERVENTION]['minus_tikhonov'][0]:+.2f} | difference {_ci(row['intervention_minus_control'])}")
    L += ["", "== Blob test set (200 images per row), PSNR"]
    for row in r["blob_test"]:
        L.append(f"  noise {row['label']:9s} {row['sensors']:2d} sensors: Tikhonov {row['tikhonov_psnr']:.2f} | {CONTROL} {row[CONTROL]:.2f} | "
                 f"{INTERVENTION} {row[INTERVENTION][0]:.2f} | difference {_ci(row['intervention_minus_control'])}")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[L.index(f"== VERDICT: {c['verdict']}; edge-gap explanation refuted: {c['edge_gap_explanation_refuted']}"):][:70]))


def plot(r, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = {"disc_blur": ("Gaussian blur of the disc edges [px]", "Edge blur of discs (primary)"),
              "vessel_width": ("vessel profile width [px]", "Vessel width (transfer)")}
    colours = {CONTROL: "C0", INTERVENTION: "C2"}
    names = {CONTROL: "4 families (control)", INTERVENTION: "4 families, half of the sharp images edge-blurred"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharey="row")
    for i, sweep in enumerate(titles):
        for j, k in enumerate(U.SENSOR_COUNTS):
            ax = axes[i, j]
            for label, style in (("high", "-"), ("noiseless", ":")):
                sel = [row for row in r["sweeps"] if row["sweep"] == sweep and row["sensors"] == k and row["label"] == label]
                x = [row["value"] for row in sel]
                for g in GROUPS:
                    d = np.array([row[g]["minus_tikhonov"] for row in sel])
                    ax.errorbar(x, d[:, 0], yerr=[d[:, 0] - d[:, 1], d[:, 2] - d[:, 0]], color=colours[g], ls=style, marker="o", ms=4,
                                capsize=2, label=f"{names[g]}, {'14 dB SNR' if label == 'high' else 'no noise'}")
            if sweep == "disc_blur":
                ax.axvspan(BLUR_RANGE[0], BLUR_RANGE[1], color="C2", alpha=0.08, lw=0)
            ax.axhline(0, color="k", lw=0.8)
            ax.set_title(f"{titles[sweep][1]}, {k} sensors", fontsize=10)
            ax.set_xlabel(titles[sweep][0])
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel("PSNR minus Tikhonov [dB]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Training with edges of intermediate sharpness (shaded: blur range used in training); 20 images per point, 95 % CI",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="report")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--train-seeds", default=None, help="comma-separated seeds: train those networks and exit")
    args = parser.parse_args()
    main(args.out_dir, args.overwrite, None if args.train_seeds is None else [int(s) for s in args.train_seeds.split(",")])
