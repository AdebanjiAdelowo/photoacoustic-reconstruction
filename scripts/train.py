"""Stage 6 — train the U-Net reconstruction-refinement model.

Run: python scripts/train.py [--overfit-check] [--epochs N] [--seed S]
                             [--device auto|cpu|mps|cuda] [--checkpoint PATH] [--data-dir DIR]
                             [--metadata PATH] [--overwrite]

Defaults reproduce the original run (seed 0, MPS on Apple Silicon, experiments/unet_checkpoint.pt).
--device auto picks CUDA, then MPS, then CPU; an explicit accelerator that is unavailable is an error.
--device cpu with torch deterministic algorithms makes a run exactly repeatable for a given seed;
--device cuda applies the deterministic CUDA settings of src/device.py. An existing checkpoint is
never replaced unless --overwrite is given, and CUDA runs may not write to experiments/, data/ or
report/ (see REMOTE_GPU.md).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.device import (DEVICE_CHOICES, DeviceUnavailableError, configure_determinism, cpu_state_dict, cuda_peak_memory,
                        device_metadata, jax_metadata, reset_cuda_peak_memory, resolve_device, synchronize)
from src.reconstruction_net import ReconstructionUNet
from src.run_safety import dataset_fingerprint, ensure_new_checkpoint, ensure_writable, git_state, write_json

SEED = 0
CHECKPOINT_PATH = "experiments/unet_checkpoint.pt"


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(choice="auto"):
    """Resolve --device and apply the determinism settings: always for CUDA, and for an explicit
    `cpu` (as before). MPS, and a CPU reached through `auto`, are left as they always were."""
    device = resolve_device(choice)
    if choice == "cpu" or device.type == "cuda":
        configure_determinism(device)
    return device


def load_split(name, data_dir="data"):
    d = np.load(os.path.join(data_dir, f"{name}.npz"))
    x = torch.from_numpy(d["recon"]).unsqueeze(1).float()      # (N, 1, H, W) input
    y = torch.from_numpy(d["phantom"]).unsqueeze(1).float()    # (N, 1, H, W) target
    return x, y


def overfit_check(device, n_examples=4, steps=200, data_dir="data"):
    """Sanity gate: can the training pipeline actually learn, on a tiny handful of examples,
    before spending time on a full run? If loss doesn't drop substantially, something upstream
    (data loading, loss, optimizer wiring) is broken and must be fixed before proceeding.
    """
    set_seed(SEED)
    x, y = load_split("train", data_dir)
    x, y = x[:n_examples].to(device), y[:n_examples].to(device)

    model = ReconstructionUNet(base_features=16).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    losses = []
    for step in range(steps):
        opt.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    print(f"Overfit check: loss[0]={losses[0]:.6f} -> loss[-1]={losses[-1]:.6f} "
          f"(ratio: {losses[-1] / losses[0]:.4f})")
    return losses


def training_metadata(model, device, requested_device, seed, config, data_dir, history):
    """Provenance of one training run (written next to the checkpoint when requested)."""
    return {
        **git_state(),
        **device_metadata(requested_device, device),
        **jax_metadata(),
        "determinism": {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "amp": False,
            "channels_last": False,
        },
        "seed": seed,
        "model": {"class": type(model).__name__, "base_features": 16,
                  "parameter_count": sum(p.numel() for p in model.parameters()),
                  "dtype": str(next(model.parameters()).dtype)},
        "training": config,
        "dataset": dataset_fingerprint([os.path.join(data_dir, f"{n}.npz") for n in ("train", "val")]),
        "best_epoch": history["best_epoch"],
        "best_val_loss": history["best_val_loss"],
        "timing": history["timing"],
        "cuda_memory": history["cuda_memory"],
    }


def train(device, epochs=60, batch_size=8, lr=1e-3, seed=SEED, checkpoint_path=CHECKPOINT_PATH,
          verbose=True, data_dir="data", overwrite=False, allow_historical=False, metadata_path=None,
          requested_device=None):
    ensure_new_checkpoint(checkpoint_path, device=device, overwrite=overwrite, allow_historical=allow_historical)
    if metadata_path is not None:
        ensure_writable(metadata_path, device=device, overwrite=overwrite)
    set_seed(seed)
    x_train, y_train = load_split("train", data_dir)
    x_val, y_val = load_split("val", data_dir)
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)

    model = ReconstructionUNet(base_features=16).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n = x_train.shape[0]
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}
    best_epoch = None
    epoch_seconds = []

    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

    reset_cuda_peak_memory(device)
    synchronize(device)
    t_start = time.perf_counter()
    for epoch in range(epochs):
        t_epoch = time.perf_counter()
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = x_train[idx], y_train[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * xb.shape[0]
        epoch_loss /= n

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val)
            val_loss = loss_fn(val_pred, y_val).item()

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            # the state dict is moved to CPU so the file loads on any machine without a device tag
            torch.save({"model_state": cpu_state_dict(model), "epoch": epoch,
                        "val_loss": val_loss, "seed": seed}, checkpoint_path)

        synchronize(device)
        epoch_seconds.append(time.perf_counter() - t_epoch)

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            print(f"epoch {epoch + 1}/{epochs}  train_loss={epoch_loss:.6f}  val_loss={val_loss:.6f}")

    synchronize(device)
    # wall clock for the whole loop (training steps, validation and checkpoint writes); the first
    # epoch is reported separately because it includes accelerator warm-up
    history["timing"] = {
        "total_seconds": time.perf_counter() - t_start,
        "first_epoch_seconds": epoch_seconds[0] if epoch_seconds else None,
        "later_epochs_mean_seconds": float(np.mean(epoch_seconds[1:])) if len(epoch_seconds) > 1 else None,
        "epoch_seconds": epoch_seconds,
        "synchronized": device.type in ("cuda", "mps"),
        "includes": "optimiser steps, validation pass and checkpoint writes",
    }
    history["cuda_memory"] = cuda_peak_memory(device)
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val_loss

    if metadata_path is not None:
        config = {"epochs": epochs, "batch_size": batch_size, "lr": lr, "optimizer": "Adam",
                  "loss": "MSELoss", "checkpoint_selection": "lowest validation loss",
                  "n_train": int(n), "n_val": int(x_val.shape[0])}
        meta = training_metadata(model, device, requested_device or str(device), seed, config, data_dir, history)
        meta["checkpoint"] = os.path.basename(str(checkpoint_path))
        write_json(metadata_path, meta, device=device, overwrite=overwrite)

    if verbose:
        print(f"Best val_loss: {best_val_loss:.6f} (checkpoint saved to {checkpoint_path})")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overfit-check", action="store_true")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--metadata", default=None, help="write training provenance to this JSON file")
    parser.add_argument("--overwrite", action="store_true", help="allow replacing an existing checkpoint")
    parser.add_argument("--allow-historical-overwrite", action="store_true",
                        help="let a CUDA/remote run write under experiments/ (not recommended)")
    args = parser.parse_args()

    try:
        device = get_device(args.device)
    except DeviceUnavailableError as exc:
        sys.exit(f"error: {exc}")
    print(f"Using device: {device}")

    if args.overfit_check:
        losses = overfit_check(device, data_dir=args.data_dir)
        ratio = losses[-1] / losses[0]
        if ratio < 0.1:
            print("OVERFIT CHECK PASSED (loss dropped by >90% on a tiny fixed batch)")
        else:
            print("OVERFIT CHECK FAILED — loss did not drop substantially. Do not proceed to full training.")
            sys.exit(1)
    else:
        train(device, epochs=args.epochs, seed=args.seed, checkpoint_path=args.checkpoint,
              data_dir=args.data_dir, overwrite=args.overwrite,
              allow_historical=args.allow_historical_overwrite, metadata_path=args.metadata,
              requested_device=args.device)
