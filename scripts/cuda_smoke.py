"""CUDA smoke test: the real physics on CPU, the real U-Net on the requested device.

This is a verification script, not an experiment; none of its numbers is a project result. It checks
that the U-Net trains and predicts on CUDA, that a checkpoint written there loads on CPU and gives
the same output, and that the j-Wave/JAX path is untouched and still on CPU.

Protocol:
  1. 4 phantoms (seeds 900000-900003, disjoint from every split), 2 with 16 sensors and 2 with 64;
  2. j-Wave forward simulation and time reversal on CPU, exactly as in generate_training_data.py;
     recordings must be (297, n, 1) and reconstructions (64, 64);
  3. the project's U-Net, 20 Adam steps (lr 1e-3, MSE) on the 4 images, on the requested device;
     losses and gradients must be finite and the loss must end below where it started;
  4. inference on the device: output (4, 1, 64, 64), finite, finite PSNR/SSIM computed on CPU arrays;
  5. checkpoint saved from the device, reloaded on CPU, CPU output compared with the device output;
  6. synchronised timings and peak CUDA allocator memory;
  7. the same 20 steps on CPU from the same initial weights, for a same-machine timing reference.

    python scripts/cuda_smoke.py --out-dir results/<RUN_ID>/smoke            # requires CUDA
    python scripts/cuda_smoke.py --out-dir DIR --device cpu --allow-non-cuda  # local dry run
    python scripts/cuda_smoke.py --validate results/<RUN_ID>/smoke/smoke_verdict.json

The verdict is written to <out-dir>/smoke_verdict.json; the exit status is non-zero unless every
check passed.
"""
import argparse
import copy
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.device import enforce_jax_cpu  # noqa: E402

enforce_jax_cpu()  # before anything imports jax

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from src.device import (DEVICE_CHOICES, DeviceUnavailableError, configure_determinism, cpu_state_dict,  # noqa: E402
                        cuda_peak_memory, device_metadata, jax_metadata, reset_cuda_peak_memory, resolve_device, synchronize,
                        to_numpy)
from src.run_safety import ensure_writable, git_state, write_json  # noqa: E402

SCHEMA_VERSION = 1
VERDICT_NAME = "smoke_verdict.json"
CHECKPOINT_NAME = "smoke_checkpoint.pt"
SMOKE_SEED_OFFSET = 900_000
SENSOR_PLAN = (16, 16, 64, 64)
EXPECTED_NT = 297
TRAIN_STEPS = 20
# CPU and CUDA evaluate the same float32 network with different convolution kernels and summation
# order. Each layer contributes rounding of order 1e-7 relative; through the 18 convolutions and 14
# instance normalisations of this U-Net that stays well below 1e-4 on outputs of order 1, while a
# wrong weight, layout or precision would show up at 1e-2 or more. 1e-4 absolute corresponds to an
# 80 dB PSNR floor, far above any reconstruction in this project.
OUTPUT_TOLERANCE = 1e-4


def simulate_inputs():
    """The physics part, on CPU through the unchanged project code. Returns arrays, checks, timings."""
    from scripts.generate_training_data import CENTRE, GRID_SIZE, RADIUS
    from src.baselines import time_reversal_reconstruction
    from src.forward_model import build_domain_and_medium, simulate_sensor_data, sparse_view_sensor_array
    from src.phantoms import random_blob_phantom

    domain, medium = build_domain_and_medium(GRID_SIZE)
    phantoms, recons, checks = [], [], []
    forward_s, reversal_s = [], []
    for i, n_sensors in enumerate(SENSOR_PLAN):
        phantom = random_blob_phantom(size=GRID_SIZE, seed=SMOKE_SEED_OFFSET + i, n_blobs=2)
        sensor_pos = sparse_view_sensor_array(n_sensors, RADIUS, CENTRE)
        t_a = time.perf_counter()
        recording, time_axis = simulate_sensor_data(phantom, domain, medium, sensor_pos)
        t_b = time.perf_counter()
        recon = time_reversal_reconstruction(recording, sensor_pos, domain, medium, time_axis)
        t_c = time.perf_counter()
        forward_s.append(t_b - t_a)
        reversal_s.append(t_c - t_b)
        checks.append(check(f"recording_shape_image{i}", recording.shape == (EXPECTED_NT, n_sensors, 1),
                            f"{tuple(recording.shape)}, expected ({EXPECTED_NT}, {n_sensors}, 1)"))
        checks.append(check(f"recording_finite_image{i}", bool(np.isfinite(recording).all()), ""))
        checks.append(check(f"reconstruction_shape_image{i}",
                            recon.shape == (GRID_SIZE, GRID_SIZE) and recon.dtype == np.float32,
                            f"{tuple(recon.shape)} {recon.dtype}, expected (64, 64) float32"))
        checks.append(check(f"reconstruction_finite_image{i}", bool(np.isfinite(recon).all()), ""))
        phantoms.append(phantom)
        recons.append(recon)
    timing = {"forward_seconds_per_call": forward_s, "time_reversal_seconds_per_call": reversal_s,
              "note": "j-Wave on CPU; the first call for each sensor count includes JAX compilation"}
    return np.stack(phantoms), np.stack(recons), checks, timing


def check(name, passed, detail=""):
    return {"name": name, "passed": bool(passed), "detail": str(detail)}


def train_steps(model, x, y, device, steps):
    """`steps` full-batch Adam steps. Returns losses, gradient finiteness and per-step seconds."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    losses, grads_finite, seconds = [], True, []
    model.train()
    for _ in range(steps):
        synchronize(device)
        t_a = time.perf_counter()
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        grads_finite = grads_finite and all(
            p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        opt.step()
        losses.append(loss.item())
        synchronize(device)
        seconds.append(time.perf_counter() - t_a)
    return losses, grads_finite, seconds


def run_smoke(device_choice="cuda", out_dir=None, require_cuda=True, steps=TRAIN_STEPS, overwrite=False):
    from src.evaluate import psnr, ssim
    from src.reconstruction_net import ReconstructionUNet

    if out_dir is None:
        raise ValueError("an output directory is required, e.g. results/<RUN_ID>/smoke")
    device = resolve_device(device_choice)  # explicit cuda/mps fail here if unavailable
    verdict_path = ensure_writable(os.path.join(out_dir, VERDICT_NAME), device=device, overwrite=overwrite)
    checkpoint_path = ensure_writable(os.path.join(out_dir, CHECKPOINT_NAME), device=device, overwrite=overwrite)
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.perf_counter()

    determinism = configure_determinism(device) if device.type in ("cpu", "cuda") else {"device_type": device.type}
    jax_meta = jax_metadata()
    checks = [check("jax_cpu_only", jax_meta["jax_cpu_only"], f"devices {jax_meta['jax_devices']}"),
              check("device_is_cuda", device.type == "cuda" or not require_cuda, f"resolved {device}")]

    phantoms, recons, physics_checks, physics_timing = simulate_inputs()
    checks += physics_checks
    jax_after = jax_metadata()
    checks.append(check("jax_cpu_only_after_simulation", jax_after["jax_cpu_only"], f"devices {jax_after['jax_devices']}"))

    # --- U-Net on the device ---
    torch.manual_seed(0)
    model = ReconstructionUNet(base_features=16)
    initial_state = copy.deepcopy(model.state_dict())
    x_cpu = torch.from_numpy(recons).unsqueeze(1).float()
    y_cpu = torch.from_numpy(phantoms).unsqueeze(1).float()
    model = model.to(device)
    x, y = x_cpu.to(device), y_cpu.to(device)
    checks.append(check("parameters_on_device", all(p.device.type == device.type for p in model.parameters()),
                        f"{sorted({str(p.device) for p in model.parameters()})}"))
    checks.append(check("batch_on_device", x.device.type == device.type and y.device.type == device.type,
                        f"x {x.device}, y {y.device}"))
    checks.append(check("float32_everywhere", x.dtype == torch.float32 and
                        all(p.dtype == torch.float32 for p in model.parameters()), ""))

    reset_cuda_peak_memory(device)
    losses, grads_finite, step_s = train_steps(model, x, y, device, steps)
    memory_training = cuda_peak_memory(device)
    checks.append(check("losses_finite", bool(np.isfinite(losses).all()) and len(losses) == steps,
                        f"first {losses[0]:.6g}, last {losses[-1]:.6g}"))
    checks.append(check("gradients_finite", grads_finite, ""))
    checks.append(check("loss_decreased", losses[-1] < losses[0], f"{losses[0]:.6g} -> {losses[-1]:.6g}"))

    model.eval()
    reset_cuda_peak_memory(device)
    inference_s = []
    with torch.no_grad():
        for _ in range(2):  # the first call is warm-up
            synchronize(device)
            t_a = time.perf_counter()
            out = model(x)
            synchronize(device)
            inference_s.append(time.perf_counter() - t_a)
    memory_inference = cuda_peak_memory(device)
    checks.append(check("output_on_device", out.device.type == device.type, str(out.device)))
    checks.append(check("output_shape", tuple(out.shape) == (len(SENSOR_PLAN), 1, 64, 64), str(tuple(out.shape))))
    out_np = to_numpy(out)
    checks.append(check("output_finite", bool(np.isfinite(out_np).all()), ""))

    t_a = time.perf_counter()
    psnrs = [psnr(out_np[i, 0], phantoms[i]) for i in range(len(phantoms))]
    ssims = [ssim(out_np[i, 0], phantoms[i]) for i in range(len(phantoms))]
    metrics_s = time.perf_counter() - t_a
    checks.append(check("psnr_finite", bool(np.isfinite(psnrs).all()), f"{[round(v, 3) for v in psnrs]}"))
    checks.append(check("ssim_finite", bool(np.isfinite(ssims).all()), f"{[round(v, 4) for v in ssims]}"))

    # --- checkpoint written from the device, read back on CPU ---
    torch.save({"model_state": cpu_state_dict(model), "epoch": steps - 1, "val_loss": losses[-1], "seed": 0},
               checkpoint_path)
    stored = torch.load(checkpoint_path, weights_only=True)  # no map_location: the file must not need one
    checks.append(check("checkpoint_is_device_neutral",
                        all(v.device.type == "cpu" for v in stored["model_state"].values()),
                        f"{sorted({str(v.device) for v in stored['model_state'].values()})}"))
    reloaded = ReconstructionUNet(base_features=16)
    reloaded.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True)["model_state"])
    reloaded.eval()
    with torch.no_grad():
        out_cpu = to_numpy(reloaded(x_cpu))
    max_diff = float(np.abs(out_cpu - out_np).max())
    checks.append(check("cpu_reload_matches_device_output", max_diff <= OUTPUT_TOLERANCE,
                        f"max |difference| {max_diff:.3e}, tolerance {OUTPUT_TOLERANCE:.0e}"))

    # --- the same steps on CPU from the same initial weights: same-machine timing reference ---
    cpu_reference = None
    if device.type != "cpu":
        cpu_model = ReconstructionUNet(base_features=16)
        cpu_model.load_state_dict(initial_state)
        cpu_losses, _, cpu_step_s = train_steps(cpu_model, x_cpu, y_cpu, torch.device("cpu"), steps)
        cpu_reference = {
            "first_step_seconds": cpu_step_s[0],
            "later_steps_mean_seconds": float(np.mean(cpu_step_s[1:])),
            "final_loss": cpu_losses[-1],
            "final_loss_difference_device_minus_cpu": losses[-1] - cpu_losses[-1],
            "note": "informational; trajectories on different devices are not expected to agree bitwise",
        }

    passed = all(c["passed"] for c in checks)
    verdict = {
        "schema_version": SCHEMA_VERSION,
        "passed": passed,
        "require_cuda": bool(require_cuda),
        "failed_checks": [c["name"] for c in checks if not c["passed"]],
        "checks": checks,
        "protocol": {"phantom_seeds": [SMOKE_SEED_OFFSET + i for i in range(len(SENSOR_PLAN))],
                     "sensor_counts": list(SENSOR_PLAN), "train_steps": steps, "batch_size": len(SENSOR_PLAN),
                     "optimizer": "Adam", "lr": 1e-3, "loss": "MSELoss", "output_tolerance": OUTPUT_TOLERANCE},
        "losses": losses,
        "psnr": psnrs,
        "ssim": ssims,
        "max_abs_difference_cpu_reload_vs_device": max_diff,
        "timing": {
            **physics_timing,
            "train_first_step_seconds": step_s[0],
            "train_later_steps_mean_seconds": float(np.mean(step_s[1:])) if len(step_s) > 1 else None,
            "train_step_seconds": step_s,
            "inference_warmup_seconds": inference_s[0],
            "inference_seconds": inference_s[1],
            "metrics_seconds_per_image": metrics_s / len(phantoms),
            "total_seconds": time.perf_counter() - t_start,
            "synchronized": device.type in ("cuda", "mps"),
            "cpu_reference_training": cpu_reference,
        },
        "cuda_memory": {"training": memory_training, "inference": memory_inference,
                        "note": "PyTorch CUDA allocator peaks; not comparable with process RSS or MPS figures"},
        "provenance": {**git_state(), **device_metadata(device_choice, device), **jax_after,
                       "determinism": determinism, "numpy_version": np.__version__,
                       "parameter_count": sum(p.numel() for p in model.parameters()),
                       "model": {"class": "ReconstructionUNet", "base_features": 16}},
    }
    write_json(verdict_path, verdict, device=device, overwrite=overwrite)
    return verdict


def validate_verdict(verdict, require_cuda=True):
    """Problems with a smoke verdict, as a list of strings; empty means the smoke is accepted."""
    problems = []
    if verdict.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"unexpected schema_version {verdict.get('schema_version')}")
    if verdict.get("passed") is not True:
        problems.append(f"smoke did not pass; failed checks: {verdict.get('failed_checks')}")
    if not verdict.get("checks") or not all(c.get("passed") for c in verdict["checks"]):
        problems.append("at least one check is missing or failed")
    prov = verdict.get("provenance", {})
    if prov.get("jax_cpu_only") is not True:
        problems.append("JAX was not CPU-only")
    if require_cuda:
        if verdict.get("require_cuda") is not True:
            problems.append("the smoke was run with --allow-non-cuda")
        if not str(prov.get("resolved_device", "")).startswith("cuda"):
            problems.append(f"resolved device was {prov.get('resolved_device')}, not CUDA")
        mem = (verdict.get("cuda_memory") or {}).get("training")
        if not mem or mem.get("peak_allocated_bytes", 0) <= 0:
            problems.append("no CUDA memory was recorded during training")
    diff = verdict.get("max_abs_difference_cpu_reload_vs_device")
    if diff is None or not diff <= OUTPUT_TOLERANCE:
        problems.append(f"CPU reload differs from the device output by {diff}")
    return problems


def summary_lines(verdict):
    prov, t = verdict["provenance"], verdict["timing"]
    mem = verdict["cuda_memory"]["training"]
    lines = [
        f"smoke passed: {verdict['passed']}  ({sum(c['passed'] for c in verdict['checks'])}/{len(verdict['checks'])} checks)",
        f"device: {prov['resolved_device']}  GPU: {prov['gpu_name']}  torch {prov['torch_version']} (CUDA {prov['torch_cuda_version']})",
        f"JAX {prov['jax_version']} devices {prov['jax_devices']}  CPU-only: {prov['jax_cpu_only']}",
        f"loss {verdict['losses'][0]:.5f} -> {verdict['losses'][-1]:.5f} in {verdict['protocol']['train_steps']} steps",
        f"CPU reload vs device output: max |difference| {verdict['max_abs_difference_cpu_reload_vs_device']:.2e}",
        f"U-Net step: first {t['train_first_step_seconds']:.4f} s, later mean {t['train_later_steps_mean_seconds']:.4f} s",
    ]
    if t.get("cpu_reference_training"):
        lines.append(f"same steps on this machine's CPU: later mean {t['cpu_reference_training']['later_steps_mean_seconds']:.4f} s")
    if mem:
        lines.append(f"CUDA peak during training: allocated {mem['peak_allocated_bytes'] / 2**20:.1f} MiB, "
                     f"reserved {mem['peak_reserved_bytes'] / 2**20:.1f} MiB")
    if verdict["failed_checks"]:
        lines.append(f"FAILED: {verdict['failed_checks']}")
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="cuda")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--allow-non-cuda", action="store_true",
                        help="local dry run on CPU/MPS; such a verdict is not accepted as a CUDA smoke")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate", metavar="VERDICT_JSON", default=None,
                        help="check an existing verdict instead of running the smoke")
    args = parser.parse_args(argv)

    if args.validate:
        with open(args.validate) as f:
            verdict = json.load(f)
        problems = validate_verdict(verdict, require_cuda=not args.allow_non_cuda)
        print("\n".join(summary_lines(verdict)))
        for p in problems:
            print(f"PROBLEM: {p}")
        print("SMOKE VERDICT ACCEPTED" if not problems else "SMOKE VERDICT REJECTED")
        return 1 if problems else 0

    try:
        verdict = run_smoke(args.device, args.out_dir, require_cuda=not args.allow_non_cuda, overwrite=args.overwrite)
    except DeviceUnavailableError as exc:
        print(f"error: {exc}")
        return 1
    print("\n".join(summary_lines(verdict)))
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
