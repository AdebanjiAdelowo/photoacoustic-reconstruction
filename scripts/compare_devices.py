"""Compare two arms of the expanded evaluation that differ only in the device the U-Net ran on.

Inputs are the expanded_eval_per_image.npz files of two runs of scripts/expanded_evaluation.py on the
same machine and the same data (normally arm A = CPU and arm B = CUDA, written by
scripts/remote_experiment.py). The published results in report/expanded_eval_results.json are read
for context only and never written.

For every noise level, sensor count and metric the comparison reports: the mean per trained network
(seed) in each arm, the mean and standard deviation over the networks, the U-Net gain over
calibrated time reversal, the difference B - A per network and for the network average (with a
paired bootstrap interval over test images), and where the published mean, interval and
between-network range lie. Time reversal must be identical in both arms; this is checked.

Agreement is statistical, not bitwise: two devices give two different sets of trained networks, in
the same way that two seeds do.

    python scripts/compare_devices.py --a results/<RUN_ID>/report/cpu --b results/<RUN_ID>/report/cuda \
        --out-dir results/<RUN_ID>/report/comparison
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.run_safety import ensure_writable  # noqa: E402
from src.stats import paired_bootstrap_ci  # noqa: E402

PER_IMAGE = "expanded_eval_per_image.npz"
HISTORICAL_JSON = os.path.join(ROOT, "report", "expanded_eval_results.json")
METRICS = (("psnr", 2), ("ssim", 3))


def _levels_and_seeds(npz):
    levels, seeds = [], []
    for key in npz.files:
        parts = key.split("__")
        if len(parts) != 3:
            continue
        if parts[0] not in levels:
            levels.append(parts[0])
        if parts[1].startswith("seed") and parts[1] not in seeds:
            seeds.append(parts[1])
    return levels, sorted(seeds)


def compare(dir_a, dir_b, label_a="cpu", label_b="cuda", historical_json=HISTORICAL_JSON):
    a, b = np.load(os.path.join(dir_a, PER_IMAGE)), np.load(os.path.join(dir_b, PER_IMAGE))
    if not (np.array_equal(a["seed"], b["seed"]) and np.array_equal(a["n_sensors"], b["n_sensors"])):
        raise ValueError("the two arms were not evaluated on the same test images")
    levels, seeds = _levels_and_seeds(a)
    if (levels, seeds) != _levels_and_seeds(b):
        raise ValueError("the two arms do not contain the same noise levels and networks")
    n_sensors = a["n_sensors"]

    historical = {}
    if historical_json and os.path.exists(historical_json):
        with open(historical_json) as f:
            historical = {lvl["label"]: lvl["by_sensors"] for lvl in json.load(f)["levels"]}

    physics_identical = all(np.array_equal(a[f"{lvl}__{m}__{met}"], b[f"{lvl}__{m}__{met}"])
                            for lvl in levels for m in ("tr_raw", "tr_cal") for met, _ in METRICS)
    result = {"arm_a": label_a, "arm_b": label_b, "networks": seeds, "n_images": int(len(n_sensors)),
              "time_reversal_identical_in_both_arms": bool(physics_identical), "levels": []}
    for lvl in levels:
        level = {"label": lvl, "by_sensors": {}}
        for k in sorted(set(n_sensors.tolist())):
            m = n_sensors == k
            entry = {"n_images": int(m.sum())}
            for metric, _ in METRICS:
                get = lambda arm, name: arm[f"{lvl}__{name}__{metric}"][m]  # noqa: E731
                arms = {}
                for label, arm in ((label_a, a), (label_b, b)):
                    per_seed = np.array([get(arm, s).mean() for s in seeds])
                    gain = paired_bootstrap_ci(get(arm, "unet_mean_over_seeds"), get(arm, "tr_cal"))
                    arms[label] = {
                        "per_seed_means": per_seed.tolist(),
                        "mean_over_seeds": float(per_seed.mean()),
                        "sd_over_seeds": float(per_seed.std(ddof=1)) if len(seeds) > 1 else None,
                        "tr_cal_mean": float(get(arm, "tr_cal").mean()),
                        "unet_minus_tr_cal": dict(zip(("mean", "ci95_lo", "ci95_hi"), gain)),
                    }
                diff = paired_bootstrap_ci(get(b, "unet_mean_over_seeds"), get(a, "unet_mean_over_seeds"))
                per_seed_diff = [float(get(b, s).mean() - get(a, s).mean()) for s in seeds]
                e = {**arms, "b_minus_a": {
                    "per_seed_mean_difference": per_seed_diff,
                    "max_abs_per_seed_mean_difference": float(np.max(np.abs(per_seed_diff))),
                    "mean_over_seeds_difference": dict(zip(("mean", "ci95_lo", "ci95_hi"), diff)),
                    "gain_over_tr_cal_difference": arms[label_b]["unet_minus_tr_cal"]["mean"]
                                                   - arms[label_a]["unet_minus_tr_cal"]["mean"],
                }}
                h = historical.get(lvl, {}).get(str(k), {}).get(metric)
                if h:
                    u = h["unet"]
                    spread = u["across_training_seeds"]
                    ctx = {"published_mean": u["mean"], "published_ci95": u["ci95"],
                           "published_seed_min": spread["min"], "published_seed_max": spread["max"],
                           "published_seed_sd": spread["sd"],
                           "published_gain_over_tr_cal": h["unet_minus_tr_cal"]["mean"]}
                    for label in (label_a, label_b):
                        mean = arms[label]["mean_over_seeds"]
                        ctx[f"{label}_minus_published_mean"] = mean - u["mean"]
                        ctx[f"{label}_mean_within_published_seed_range"] = bool(spread["min"] <= mean <= spread["max"])
                        ctx[f"{label}_mean_within_published_ci95"] = bool(u["ci95"][0] <= mean <= u["ci95"][1])
                    ctx["device_difference_within_published_seed_sd"] = bool(
                        abs(e["b_minus_a"]["mean_over_seeds_difference"]["mean"]) <= spread["sd"])
                    e["published_context"] = ctx
                entry[metric] = e
            level["by_sensors"][str(k)] = entry
        result["levels"].append(level)
    return result


def text_report(r):
    a, b = r["arm_a"], r["arm_b"]
    L = [f"Device comparison: arm A = {a}, arm B = {b}; {r['n_images']} test images, networks {', '.join(r['networks'])}",
         f"Time reversal identical in both arms: {r['time_reversal_identical_in_both_arms']}",
         "Agreement between devices is statistical, not bitwise. Timing is reported separately.", ""]
    for lvl in r["levels"]:
        L.append(f"== noise {lvl['label']}")
        for k, entry in lvl["by_sensors"].items():
            for metric, d in METRICS:
                e = entry[metric]
                diff = e["b_minus_a"]["mean_over_seeds_difference"]
                L.append(f"  {k} sensors {metric.upper()}:")
                for label in (a, b):
                    x = e[label]
                    L.append(f"    {label:5s} per network " + " ".join(f"{v:.{d}f}" for v in x["per_seed_means"])
                             + f" | mean {x['mean_over_seeds']:.{d}f}"
                             + (f" sd {x['sd_over_seeds']:.{d}f}" if x["sd_over_seeds"] is not None else "")
                             + f" | U-Net - cal TR {x['unet_minus_tr_cal']['mean']:+.{d}f} "
                               f"[{x['unet_minus_tr_cal']['ci95_lo']:+.{d}f}, {x['unet_minus_tr_cal']['ci95_hi']:+.{d}f}]")
                L.append(f"    {b} - {a}: per network " + " ".join(f"{v:+.{d}f}" for v in e["b_minus_a"]["per_seed_mean_difference"])
                         + f" | network average {diff['mean']:+.{d}f} [{diff['ci95_lo']:+.{d}f}, {diff['ci95_hi']:+.{d}f}]")
                c = e.get("published_context")
                if c:
                    L.append(f"    published (other machine, CPU): mean {c['published_mean']:.{d}f} "
                             f"[{c['published_ci95'][0]:.{d}f}, {c['published_ci95'][1]:.{d}f}], networks "
                             f"{c['published_seed_min']:.{d}f} to {c['published_seed_max']:.{d}f}, sd {c['published_seed_sd']:.{d}f}; "
                             f"{a} {c[f'{a}_minus_published_mean']:+.{d}f}, {b} {c[f'{b}_minus_published_mean']:+.{d}f}; "
                             f"device difference within one published network sd: {c['device_difference_within_published_seed_sd']}")
    return "\n".join(L) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--a", required=True, help="report directory of arm A (CPU)")
    parser.add_argument("--b", required=True, help="report directory of arm B (CUDA)")
    parser.add_argument("--label-a", default="cpu")
    parser.add_argument("--label-b", default="cuda")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    write(compare(args.a, args.b, args.label_a, args.label_b), args.out_dir, args.overwrite)


def write(result, out_dir, overwrite=False):
    paths = [ensure_writable(os.path.join(out_dir, name), overwrite=overwrite)
             for name in ("device_comparison.json", "device_comparison.txt")]
    os.makedirs(out_dir, exist_ok=True)
    with open(paths[0], "w") as f:
        json.dump(result, f, indent=2)
    text = text_report(result)
    with open(paths[1], "w") as f:
        f.write(text)
    print(text)
    return paths


if __name__ == "__main__":
    main()
