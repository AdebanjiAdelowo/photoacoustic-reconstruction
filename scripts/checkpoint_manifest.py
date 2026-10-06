"""Provenance of the ten checkpoints behind the edge-sharpness intervention, and their verification.

The five control networks (experiments/unrolled_mixed/mixed160_seed*.pt) and the five intervention
networks (edgefill160_seed*.pt) are gitignored, so the committed results rest on files that exist
only where they were trained. This script records what identifies them and checks that a set of
files, for instance one restored from an archive, is that set.

    python scripts/checkpoint_manifest.py                     # write report/checkpoint_manifest.json
    python scripts/checkpoint_manifest.py --verify [--checkpoint-dir DIR]
    python scripts/checkpoint_manifest.py --archive PATH.tar.gz   # archive + report/checkpoint_archive.json

Writing the manifest reads the checkpoints and never writes to them. Besides hashes and the stored
metadata it runs three integrity checks that tie each file to committed numbers:

  1. the stored epoch and validation loss equal those in the committed results JSON;
  2. the validation loss recomputed from the weights (same validation images and noise draws as in
     training) equals the stored one;
  3. a small part of the committed evaluation is recomputed: per network on the first 20 images of
     the shape test set at 14 dB SNR (control only; per-network scores of the intervention were not
     committed), and per group on two sweep points of report/edge_gap_intervention_results.json.

The training environment was not recorded when the networks were trained. The manifest states what
follows from the committed code and marks everything else as unknown; the environment in which the
manifest was written is recorded separately and is not a statement about training.
"""
import argparse
import datetime
import importlib.metadata
import json
import os
import platform
import sys
import tarfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.run_safety import sha256_file  # noqa: E402

CKPT_DIR = "experiments/unrolled_mixed"
MANIFEST_PATH = "report/checkpoint_manifest.json"
ARCHIVE_RECORD_PATH = "report/checkpoint_archive.json"
SEEDS = (0, 1, 2, 3, 4)
GROUPS = {
    "mixed160": {
        "role": "control",
        "training_script": "scripts/mixed_phantom_experiment.py",
        "training_split": "mixed_split(TRAIN_SEED_OFFSET=200000, 160) and mixed_split(VAL_SEED_OFFSET=210000, 32)",
        "protocol_commit": None,
        "protocol_note": "the script and its results were committed together; there is no earlier protocol commit",
        "result_commit": "8e975dd",
        "results_with_training_record": "report/mixed_phantom_results.json",
        "committed_result_files": ["report/mixed_phantom_results.json", "report/mixed_phantom_results.txt",
                                   "report/mixed_phantom_per_image.npz", "report/prior_shift_results.json",
                                   "report/edge_gap_intervention_results.json"],
    },
    "edgefill160": {
        "role": "intervention",
        "training_script": "scripts/edge_gap_intervention.py",
        "training_split": "edgefill_split(200000, 160) and edgefill_split(210000, 32): the control images with "
                          "60 of 160 (12 of 32) sharp-family images replaced by edge-blurred copies",
        "protocol_commit": "d19c949",
        "protocol_note": "design and criteria committed before training",
        "result_commit": "c1f1b3e",
        "results_with_training_record": "report/edge_gap_intervention_results.json",
        "committed_result_files": ["report/edge_gap_intervention_results.json", "report/edge_gap_intervention_results.txt",
                                   "report/edge_gap_intervention.png"],
    },
}
PACKAGES = ("torch", "numpy", "scipy", "jax", "jaxlib", "jwave", "jaxdf", "scikit-image")
SUBSET_SHAPE_IMAGES = 20
SUBSET_SWEEP_POINTS = (("disc_blur", 1.0), ("vessel_width", 1.0))


def checkpoint_names():
    return [f"{group}_seed{seed}.pt" for group in GROUPS for seed in SEEDS]


def file_record(path):
    stat = os.stat(path)
    return {"sha256": sha256_file(path), "bytes": stat.st_size,
            "file_modified_utc": datetime.datetime.fromtimestamp(stat.st_mtime, datetime.timezone.utc).isoformat(timespec="seconds")}


def verify(manifest, checkpoint_dir):
    """Compare the files in `checkpoint_dir` with the manifest. Returns a list of problems (empty if none)."""
    problems = []
    for entry in manifest["checkpoints"]:
        path = os.path.join(checkpoint_dir, entry["filename"])
        if not os.path.exists(path):
            problems.append(f"{entry['filename']}: missing")
        elif os.path.getsize(path) != entry["bytes"]:
            problems.append(f"{entry['filename']}: {os.path.getsize(path)} bytes, expected {entry['bytes']}")
        elif sha256_file(path) != entry["sha256"]:
            problems.append(f"{entry['filename']}: SHA-256 differs from the manifest")
    return problems


def current_environment():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
            "packages": versions}


def _validation_loss(U, physics, model, split, clean):
    """The validation loss of U.train for the noise-trained variant: every image at every noise level."""
    import torch

    rng = np.random.default_rng(U.TRAIN_NOISE_SEED + 1)
    Y, idx = [], []
    for r in U.REL_STDS:
        Y += U.add_relative_noise(clean, [r] * len(clean), rng)
        idx += list(range(len(clean)))
    idx = np.array(idx)
    b = physics.backproject(Y, split["n_sensors"][idx])
    target = torch.from_numpy(split["phantom"]).unsqueeze(1).float()[idx]
    with torch.no_grad():
        out = model(b, physics.geometry_index(split["n_sensors"][idx]), physics.operators, U.GRID_SIZE)
        return float(torch.nn.functional.mse_loss(out, target))


def build_manifest(checkpoint_dir=CKPT_DIR):
    os.chdir(ROOT)
    import scripts.edge_gap_intervention as E
    M, P, U = E.M, E.P, E.U
    from scripts.evaluate_noise_sensitivity import NOISE_LEVELS
    from scripts.tikhonov_evaluation import noisy
    from src.evaluate import psnr
    from src.run_safety import git_state

    before = {name: file_record(os.path.join(checkpoint_dir, name)) for name in checkpoint_names()}
    setup = M.Setup()
    committed = {g: json.load(open(GROUPS[g]["results_with_training_record"]))["networks"] for g in GROUPS}
    val_splits = {"mixed160": M.mixed_split(M.VAL_SEED_OFFSET, 32), "edgefill160": E.edgefill_split(M.VAL_SEED_OFFSET, 32)}
    val_clean = {g: setup.recordings(s) for g, s in val_splits.items()}

    models, entries = {}, []
    for group, info in GROUPS.items():
        for seed in SEEDS:
            name = f"{group}_seed{seed}.pt"
            model, ckpt = U.load_model(os.path.join(checkpoint_dir, name), setup.device)
            models[f"{group}_seed{seed}"] = model
            stored = {k: v for k, v in ckpt.items() if k != "model_state"}
            in_results = committed[group][f"{group}_seed{seed}"]
            recomputed = _validation_loss(U, setup.physics, model, val_splits[group], val_clean[group])
            entries.append({
                "filename": name, "path": f"{CKPT_DIR}/{name}", "group": group, "role": info["role"], "seed": seed,
                **before[name],
                "loads_with": "scripts.unrolled_evaluation.load_model (torch.load, weights_only=True)",
                "stored_metadata": stored,
                "best_epoch_zero_indexed": stored["epoch"],
                "state_dict": {k: list(v.shape) for k, v in ckpt["model_state"].items()},
                "n_parameters": int(sum(v.numel() for v in ckpt["model_state"].values())),
                "link_to_committed_results": {
                    "results_file": info["results_with_training_record"],
                    "stored_epoch_and_val_loss_equal_committed": bool(
                        stored["epoch"] == in_results["epoch"] and stored["val_loss"] == in_results["val_loss"]),
                    "val_loss_recomputed_from_weights": recomputed,
                    "val_loss_recomputed_minus_stored": recomputed - stored["val_loss"],
                },
            })

    # --- part of the committed evaluation, recomputed ---
    high = dict(NOISE_LEVELS)["high"]
    shape = M.shape_test_set()
    subset = {k: v[:SUBSET_SHAPE_IMAGES] for k, v in shape.items()}
    saved = np.load("report/mixed_phantom_per_image.npz")
    out = U.reconstruct({n: m for n, m in models.items() if n.startswith("mixed160")}, setup.physics,
                        noisy(setup.recordings(subset), high, M.SHAPE_NOISE_SEED), subset["n_sensors"])
    for entry in entries:
        if entry["group"] == "mixed160":
            mine = np.array([psnr(out[f"mixed160_seed{entry['seed']}"][i], subset["phantom"][i]) for i in range(SUBSET_SHAPE_IMAGES)])
            ref = saved[f"C__high__mixed160_seed{entry['seed']}__psnr"][:SUBSET_SHAPE_IMAGES]
            entry["link_to_committed_results"]["per_network_psnr_max_abs_difference_db"] = float(np.abs(mine - ref).max())
            entry["link_to_committed_results"]["per_network_psnr_compared_with"] = (
                f"report/mixed_phantom_per_image.npz, first {SUBSET_SHAPE_IMAGES} images of the shape test set, 14 dB SNR")
        else:
            entry["link_to_committed_results"]["per_network_psnr_max_abs_difference_db"] = None
            entry["link_to_committed_results"]["per_network_psnr_compared_with"] = "not available: no per-network scores were committed"

    with open("report/edge_gap_intervention_results.json") as f:
        rows = {(r["sweep"], r["value"], r["label"], r["sensors"]): r for r in json.load(f)["sweeps"]}
    sweeps, n_s = P.sweep_images()
    group_checks = []
    for sweep, value in SUBSET_SWEEP_POINTS:
        images = sweeps[sweep][value]
        out = U.reconstruct(models, setup.physics, noisy(setup.recordings({"phantom": images, "n_sensors": n_s}), high,
                                                         P.SWEEP_NOISE_SEED), n_s)
        for k in U.SENSOR_COUNTS:
            m = np.nonzero(n_s == k)[0]
            for group in GROUPS:
                mine = float(np.mean([[psnr(out[f"{group}_seed{s}"][i], images[i]) for i in m] for s in SEEDS]))
                ref = rows[(sweep, value, "high", k)][group]["psnr"]
                group_checks.append({"sweep": sweep, "value": value, "sensors": int(k), "group": group,
                                     "committed_mean_psnr_db": ref, "recomputed_mean_psnr_db": mine,
                                     "difference_db": mine - ref})

    after = {name: file_record(os.path.join(checkpoint_dir, name)) for name in checkpoint_names()}
    assert after == before, "a checkpoint changed while the manifest was being written"
    cfg = M.MIXED_SETS["mixed160"]
    return {
        "purpose": "Identity and provenance of the ten checkpoints behind the edge-sharpness intervention "
                   "(scripts/edge_gap_intervention.py). The files are gitignored; verify a restored copy with "
                   "python scripts/checkpoint_manifest.py --verify --checkpoint-dir DIR.",
        "written_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "written_at_git_state": git_state(),
        "architecture": {"class": "src.unrolled.UnrolledTikhonov", "geometries": list(U.SENSOR_COUNTS),
                         "iterations": U.ITERATIONS, "denoiser_features": 32, "denoiser_layers": 5,
                         "init_log10_mu": -2.0, "grid_size": U.GRID_SIZE},
        "training": {"variant": "noise (noise level drawn per recording and epoch from the five evaluation levels)",
                     "n_train": cfg["n_train"], "n_val": cfg["n_val"], "epochs": cfg["epochs"], "batch_size": U.BATCH_SIZE,
                     "optimiser": "Adam", "learning_rate": U.LR, "optimiser_steps": cfg["epochs"] * cfg["n_train"] // U.BATCH_SIZE,
                     "checkpoint_rule": "lowest validation loss; saved whenever it improves",
                     "seeds": list(SEEDS)},
        "groups": GROUPS,
        "historical_training_environment": {
            "recorded_at_training_time": False,
            "device": {"value": "cpu", "basis": "the committed training code builds its setup with get_device('cpu'); "
                                                "no run log was kept"},
            "host": {"value": "unknown", "note": "believed to be the local Apple Silicon machine on which the files "
                                                 "were found; inferred, not recorded"},
            "python": "unknown", "torch": "unknown", "other_packages": "unknown",
            "pinned_in_requirements_txt_at_result_commit": "torch==2.14.0, numpy==2.4.6, scipy==1.17.1, jax==0.4.38, "
                                                           "jaxlib==0.4.38, jwave==0.2.1, jaxdf==0.2.8 (intended "
                                                           "versions; that they were the installed ones was not recorded)",
            "training_time": "unknown; file_modified_utc of each checkpoint is the file-system time of its last write, "
                             "which is when its best epoch was saved",
        },
        "environment_at_verification": current_environment(),
        "checkpoints": entries,
        "integrity_check": {
            "note": "Recomputed in the environment at verification from the files hashed above. Not a new experiment: "
                    "every number is compared with one already committed.",
            "all_ten_load": True,
            "max_abs_val_loss_recomputed_minus_stored": max(abs(e["link_to_committed_results"]["val_loss_recomputed_minus_stored"]) for e in entries),
            "control_per_network_max_abs_psnr_difference_db": max(
                e["link_to_committed_results"]["per_network_psnr_max_abs_difference_db"] for e in entries if e["group"] == "mixed160"),
            "group_mean_checks": group_checks,
            "group_mean_max_abs_difference_db": max(abs(c["difference_db"]) for c in group_checks),
        },
    }


def write_archive(archive_path, manifest_path=MANIFEST_PATH, checkpoint_dir=CKPT_DIR, record_path=ARCHIVE_RECORD_PATH):
    """A tar.gz of the ten checkpoints and the manifest, and a small tracked record of its hash and contents."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    problems = verify(manifest, checkpoint_dir)
    if problems:
        raise SystemExit("the checkpoints do not match the manifest:\n  " + "\n  ".join(problems))
    for path in (archive_path, record_path):
        if os.path.exists(path):
            raise SystemExit(f"{path} exists; remove it first if it is to be replaced")
    members = [(os.path.join(checkpoint_dir, e["filename"]), f"{CKPT_DIR}/{e['filename']}") for e in manifest["checkpoints"]]
    members.append((manifest_path, MANIFEST_PATH))
    with tarfile.open(archive_path, "w:gz") as tar:
        for source, name in members:
            tar.add(source, arcname=name)
    record = {
        "archive_filename": os.path.basename(archive_path),
        "sha256": sha256_file(archive_path), "bytes": os.path.getsize(archive_path),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "members": [{"name": name, "sha256": sha256_file(source), "bytes": os.path.getsize(source)} for source, name in members],
        "restore": "tar -xzf ARCHIVE -C REPOSITORY_ROOT (never over existing checkpoints), then "
                   "python scripts/checkpoint_manifest.py --verify",
        "note": "The archive is not committed and its location is not recorded here. The member hashes of the "
                "checkpoints are those of report/checkpoint_manifest.json.",
    }
    with open(record_path, "w") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="compare checkpoint files with the committed manifest")
    parser.add_argument("--checkpoint-dir", default=os.path.join(ROOT, CKPT_DIR))
    parser.add_argument("--archive", default=None, help="write this tar.gz and report/checkpoint_archive.json")
    args = parser.parse_args()
    manifest_file = os.path.join(ROOT, MANIFEST_PATH)
    if args.verify:
        with open(manifest_file) as f:
            found = verify(json.load(f), args.checkpoint_dir)
        print("\n".join(found) if found else f"all {len(checkpoint_names())} checkpoints match {MANIFEST_PATH}")
        sys.exit(1 if found else 0)
    if args.archive:
        r = write_archive(args.archive, manifest_file, args.checkpoint_dir, os.path.join(ROOT, ARCHIVE_RECORD_PATH))
        print(f"wrote {args.archive}: {r['bytes']} bytes, sha256 {r['sha256']}")
        sys.exit(0)
    if os.path.exists(manifest_file):
        raise SystemExit(f"{MANIFEST_PATH} exists; it is a record and is not regenerated in place")
    result = build_manifest(args.checkpoint_dir)
    with open(manifest_file, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    check = result["integrity_check"]
    print(f"wrote {MANIFEST_PATH}: validation loss within {check['max_abs_val_loss_recomputed_minus_stored']:.2e}, "
          f"control per-network PSNR within {check['control_per_network_max_abs_psnr_difference_db']:.2e} dB, "
          f"group means within {check['group_mean_max_abs_difference_db']:.2e} dB")
