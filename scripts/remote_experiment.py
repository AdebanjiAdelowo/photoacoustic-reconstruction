"""Gated CPU-versus-CUDA experiment on one machine, following the expanded-evaluation protocol.

Nothing here changes the protocol of scripts/expanded_evaluation.py: the same splits, 400 test
images, noise levels, U-Net recipe (60 epochs, Adam 1e-3, batch 8, best-validation checkpoint) and
metrics, with training seeds 0 to 4. What is added is a second arm and isolation of every output.

  data      generated once with j-Wave on CPU into <run-dir>/data and shared by both arms; the
            time-reversal reconstructions at every noise level are cached there as well
  arm A     U-Net trained and applied on CPU, deterministic, seeds 0-4
  arm B     U-Net trained and applied on CUDA, deterministic settings of src/device.py, seeds 0-4
  reports   <run-dir>/report/{cpu,cuda}/ (expanded evaluation), .../mvp_seed0/ (the 8-image
            evaluation of the seed-0 network of that arm, which is a newly trained network and not
            the original MPS checkpoint), <run-dir>/report/comparison/ (scripts/compare_devices.py)
  records   <run-dir>/provenance.json, <run-dir>/timing.json, a .meta.json beside every checkpoint

The Tikhonov baseline, its diagnostics and the model-mismatch check are not part of this run: they
are CPU-only (JAX and SciPy), numerically sensitive and independent of the U-Net device. Run them
separately if needed.

    python scripts/remote_experiment.py --run-dir results/<RUN_ID> --plan
    python scripts/remote_experiment.py --run-dir results/<RUN_ID> --estimate results/<RUN_ID>/smoke/smoke_verdict.json
    python scripts/remote_experiment.py --run-dir results/<RUN_ID> --confirm-full

Without --confirm-full the experiment does not start.
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.device import enforce_jax_cpu  # noqa: E402

enforce_jax_cpu()  # before anything imports jax

SEEDS = (0, 1, 2, 3, 4)
ARMS = ("cpu", "cuda")
MVP_NOTE = ("newly trained network (seed 0, device {arm}, remote run {run_id}); this is NOT the original "
            "MPS checkpoint behind report/mvp_results.txt")
# workload of the protocol, used for the plan and the runtime estimate
N_SPLIT_IMAGES = 40 + 8 + 8
N_TEST_IMAGES = 400
N_NOISE_LEVELS = 5
EPOCHS, STEPS_PER_EPOCH = 60, 5


def plan(run_dir, arms=ARMS, seeds=SEEDS):
    run_dir = os.path.abspath(run_dir)
    return {
        "run_dir": run_dir,
        "arms": list(arms),
        "seeds": list(seeds),
        "data_dir": os.path.join(run_dir, "data"),
        "tr_cache": os.path.join(run_dir, "data", "tr_by_noise_level.npz"),
        "checkpoint_dirs": {arm: os.path.join(run_dir, "checkpoints", arm) for arm in arms},
        "report_dirs": {arm: os.path.join(run_dir, "report", arm) for arm in arms},
        "mvp_dirs": {arm: os.path.join(run_dir, "report", arm, "mvp_seed0") for arm in arms},
        "comparison_dir": os.path.join(run_dir, "report", "comparison"),
        "physics": "j-Wave/JAX on CPU, generated once and shared by both arms",
        "excluded": ["tikhonov_evaluation.py", "tikhonov_diagnostics.py", "model_mismatch_evaluation.py"],
        "workload": {
            "forward_simulations": N_SPLIT_IMAGES + 2 * N_TEST_IMAGES,
            "time_reversals": N_SPLIT_IMAGES + N_TEST_IMAGES + N_NOISE_LEVELS * N_TEST_IMAGES,
            "optimiser_steps_per_arm": len(seeds) * EPOCHS * STEPS_PER_EPOCH,
            "metric_images_per_arm": (2 + len(seeds)) * N_NOISE_LEVELS * N_TEST_IMAGES,
        },
    }


def estimate(smoke_verdict, arms=ARMS, seeds=SEEDS):
    """Rough runtime of the full run from the components the smoke measured on this machine."""
    t = smoke_verdict["timing"]
    w = plan(".", arms, seeds)["workload"]
    forward = min(t["forward_seconds_per_call"])        # steady state: the fastest call excludes compilation
    reversal = min(t["time_reversal_seconds_per_call"])
    step_device = t["train_later_steps_mean_seconds"]
    cpu_ref = t.get("cpu_reference_training")
    step_cpu = cpu_ref["later_steps_mean_seconds"] if cpu_ref else step_device
    est = {
        "physics_seconds": w["forward_simulations"] * forward + w["time_reversals"] * reversal,
        "training_seconds_cpu_arm": w["optimiser_steps_per_arm"] * step_cpu,
        "training_seconds_cuda_arm": w["optimiser_steps_per_arm"] * step_device,
        "metrics_seconds": len(arms) * w["metric_images_per_arm"] * t["metrics_seconds_per_image"],
        "basis": "smoke timings; training steps were measured at batch 4 and the protocol uses batch 8, "
                 "validation passes, checkpoint writes, inference and JAX compilation are not included",
    }
    est["total_seconds"] = (est["physics_seconds"] + est["training_seconds_cpu_arm"]
                            + est["training_seconds_cuda_arm"] + est["metrics_seconds"])
    est["physics_share"] = est["physics_seconds"] / est["total_seconds"]
    return est


def run_full(run_dir, arms=ARMS, seeds=SEEDS, overwrite=False):
    """The experiment itself. Only reached with --confirm-full."""
    p = plan(run_dir, arms, seeds)  # resolved before the imports below, one of which changes directory

    import numpy as np

    import scripts.expanded_evaluation as expanded
    from scripts import compare_devices, evaluate_mvp, generate_training_data
    from scripts.train import get_device, train
    from src.device import assert_jax_cpu, device_metadata
    from src.run_safety import (dataset_fingerprint, ensure_writable, git_state, is_historical, make_run_dirs,
                                write_json)

    if is_historical(p["run_dir"]):
        raise SystemExit(f"{p['run_dir']} is a historical location; use results/<RUN_ID>")
    jax_meta = assert_jax_cpu()
    devices = {arm: get_device(arm) for arm in arms}  # fails here, before any work, if CUDA is missing
    make_run_dirs(p["run_dir"])
    timing = {"data": {}, "training": {}, "evaluation": {}}
    t0 = time.perf_counter()

    # --- physics, once ---
    data_dir = p["data_dir"]
    generate_training_data.main(data_dir=data_dir, overwrite=overwrite, timing=timing["data"])
    test_path = os.path.join(data_dir, os.path.basename(expanded.TEST_PATH))
    ensure_writable(test_path, overwrite=overwrite)
    expanded.generate_test_set(test_path, data_dir, timing=timing["data"])
    timing["data"]["note"] = "j-Wave on CPU; includes JAX compilation"

    # --- training, both arms on the same data ---
    for arm in arms:
        os.makedirs(p["checkpoint_dirs"][arm], exist_ok=True)
        timing["training"][arm] = {}
        for s in seeds:
            path = os.path.join(p["checkpoint_dirs"][arm], f"unet_seed{s}.pt")
            history = train(devices[arm], seed=s, checkpoint_path=path, verbose=False, data_dir=data_dir,
                            overwrite=overwrite, metadata_path=path.replace(".pt", ".meta.json"),
                            requested_device=arm)
            timing["training"][arm][f"seed{s}"] = {**{k: v for k, v in history["timing"].items() if k != "epoch_seconds"},
                                                   "cuda_memory": history["cuda_memory"],
                                                   "best_epoch": history["best_epoch"],
                                                   "best_val_loss": history["best_val_loss"]}
            print(f"[{time.perf_counter() - t0:.0f}s] trained {arm} seed {s}")

    # --- evaluation: the expanded protocol per arm, time reversal shared through the cache ---
    for arm in arms:
        timing["evaluation"][arm] = expanded.main(
            device_choice=arm, data_dir=data_dir, ckpt_dir=p["checkpoint_dirs"][arm], out_dir=p["report_dirs"][arm],
            reference_checkpoint=None, tr_cache=p["tr_cache"],
            timing_path=os.path.join(p["report_dirs"][arm], "expanded_eval_timing.json"), overwrite=overwrite)
        evaluate_mvp.main(arm, os.path.join(p["checkpoint_dirs"][arm], "unet_seed0.pt"), data_dir, p["mvp_dirs"][arm],
                          note=MVP_NOTE.format(arm=arm, run_id=os.path.basename(p["run_dir"])), overwrite=overwrite)

    comparison = None
    if len(arms) == 2:
        comparison = compare_devices.compare(p["report_dirs"][arms[0]], p["report_dirs"][arms[1]], arms[0], arms[1])
        compare_devices.write(comparison, p["comparison_dir"], overwrite)

    timing["end_to_end_seconds"] = time.perf_counter() - t0
    timing["interpretation"] = ("U-Net training and inference times compare the two arms on this machine. "
                                "End-to-end time also contains the CPU physics, which is the same work for both.")
    write_json(os.path.join(p["run_dir"], "timing.json"), timing, overwrite=overwrite)
    write_json(os.path.join(p["run_dir"], "provenance.json"), {
        **git_state(), **jax_meta, "numpy_version": np.__version__, "plan": p,
        "devices": {arm: device_metadata(arm, devices[arm]) for arm in arms},
        "dataset": dataset_fingerprint([os.path.join(data_dir, f"{n}.npz") for n in ("train", "val", "test", "test_expanded")]),
        "protocol": "scripts/expanded_evaluation.py, unchanged; reference checkpoint not scored",
        "time_reversal_identical_in_both_arms": comparison["time_reversal_identical_in_both_arms"] if comparison else None,
    }, overwrite=overwrite)
    print(f"FULL EXPERIMENT COMPLETE in {timing['end_to_end_seconds']:.0f} s: {p['run_dir']}")
    return timing


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-dir", required=True, help="results/<RUN_ID>")
    parser.add_argument("--plan", action="store_true", help="print what would run and exit")
    parser.add_argument("--estimate", metavar="SMOKE_VERDICT_JSON", default=None,
                        help="print a runtime estimate from a smoke verdict and exit")
    parser.add_argument("--confirm-full", action="store_true", help="required to start the experiment")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.estimate:
        with open(args.estimate) as f:
            est = estimate(json.load(f))
        print(json.dumps(est, indent=2))
        print(f"Estimated full run: about {est['total_seconds'] / 60:.0f} min, "
              f"{100 * est['physics_share']:.0f} % of it CPU physics. This is an estimate, not a measurement.")
        return 0
    print(json.dumps(plan(args.run_dir), indent=2))
    if args.plan or not args.confirm_full:
        print("FULL EXPERIMENT NOT STARTED (pass --confirm-full to run it).")
        return 0 if args.plan else 2
    run_full(args.run_dir, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    sys.exit(main())
