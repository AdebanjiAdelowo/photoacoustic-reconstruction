"""Stage 6 — train the U-Net reconstruction-refinement model.

Run: python scripts/train.py [--overfit-check] [--epochs N]
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reconstruction_net import ReconstructionUNet

SEED = 0
CHECKPOINT_PATH = "experiments/unet_checkpoint.pt"


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_split(name):
    d = np.load(f"data/{name}.npz")
    x = torch.from_numpy(d["recon"]).unsqueeze(1).float()      # (N, 1, H, W) input
    y = torch.from_numpy(d["phantom"]).unsqueeze(1).float()    # (N, 1, H, W) target
    return x, y


def overfit_check(device, n_examples=4, steps=200):
    """Sanity gate: can the training pipeline actually learn, on a tiny handful of examples,
    before spending time on a full run? If loss doesn't drop substantially, something upstream
    (data loading, loss, optimizer wiring) is broken and must be fixed before proceeding.
    """
    set_seed(SEED)
    x, y = load_split("train")
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


def train(device, epochs=60, batch_size=8, lr=1e-3):
    set_seed(SEED)
    x_train, y_train = load_split("train")
    x_val, y_val = load_split("val")
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)

    model = ReconstructionUNet(base_features=16).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n = x_train.shape[0]
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    os.makedirs("experiments", exist_ok=True)

    for epoch in range(epochs):
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
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "val_loss": val_loss}, CHECKPOINT_PATH)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"epoch {epoch + 1}/{epochs}  train_loss={epoch_loss:.6f}  val_loss={val_loss:.6f}")

    print(f"Best val_loss: {best_val_loss:.6f} (checkpoint saved to {CHECKPOINT_PATH})")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overfit-check", action="store_true")
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    if args.overfit_check:
        losses = overfit_check(device)
        ratio = losses[-1] / losses[0]
        if ratio < 0.1:
            print("OVERFIT CHECK PASSED (loss dropped by >90% on a tiny fixed batch)")
        else:
            print("OVERFIT CHECK FAILED — loss did not drop substantially. Do not proceed to full training.")
            sys.exit(1)
    else:
        train(device, epochs=args.epochs)
