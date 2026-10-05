"""Do the unrolled networks keep their advantage on phantoms unlike their training data?

A diagnostic added after the main evaluation (scripts/unrolled_evaluation.py) was seen; it is not
part of that protocol, and it is small (20 images per family, 10 per sensor count). The networks
were trained on images of 1 to 3 Gaussian blobs. They are applied here, unchanged, to
  * new images from that family (as a control),
  * images of 8 Gaussian blobs,
  * images of 2 or 3 sharp-edged discs,
without noise and at the "high" noise level (relative std 0.20), and compared with Tikhonov using the
weights already selected on the training split for those noise levels (report/tikhonov_results.json).
Phantom seeds and noise seeds are fixed and disjoint from every other split.

Output: report/unrolled_generalisation.txt.    python scripts/unrolled_generalisation_check.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import scripts.unrolled_evaluation as U  # noqa: E402
from src.evaluate import psnr, ssim  # noqa: E402
from src.phantoms import random_blob_phantom  # noqa: E402

N_PER_FAMILY = 20
SEED = 424242
LEVELS = (("noiseless", 0.0), ("high", 0.2))
OUT = "report/unrolled_generalisation.txt"


def disc_phantom(rng, n_discs, size=U.GRID_SIZE):
    X, Y = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    img = np.zeros((size, size))
    for _ in range(n_discs):
        cx, cy, r, a = rng.uniform(20, 44), rng.uniform(20, 44), rng.uniform(3, 7), rng.uniform(0.5, 1.0)
        img = np.maximum(img, a * ((X - cx) ** 2 + (Y - cy) ** 2 <= r * r))
    return (img / img.max()).astype(np.float32)


def main():
    device = U.get_device("cpu")
    physics = U.Physics(device)
    domain, medium = U.build_domain_and_medium(U.GRID_SIZE)
    sensors = {k: U.sparse_view_sensor_array(k, U.RADIUS, U.CENTRE) for k in U.SENSOR_COUNTS}
    models = {f"{v}_seed{s}": U.load_model(f"{U.CKPT_DIR}/{v}_seed{s}.pt", device)[0]
              for v in U.VARIANTS for s in U.TRAIN_SEEDS}
    with open("report/tikhonov_results.json") as f:
        selection = json.load(f)["selection"]
    rng = np.random.default_rng(SEED)
    families = {
        "1 to 3 Gaussian blobs (training family)": [random_blob_phantom(U.GRID_SIZE, 700_000 + i, int(rng.integers(1, 4)))
                                                    for i in range(N_PER_FAMILY)],
        "8 Gaussian blobs": [random_blob_phantom(U.GRID_SIZE, 710_000 + i, 8) for i in range(N_PER_FAMILY)],
        "2 or 3 sharp-edged discs": [disc_phantom(rng, int(rng.integers(2, 4))) for _ in range(N_PER_FAMILY)],
    }
    n_s = np.array(list(U.SENSOR_COUNTS) * (N_PER_FAMILY // 2))
    L = ["Generalisation check for the unrolled Tikhonov networks (scripts/unrolled_generalisation_check.py)", "",
         __doc__.split("Output:")[0].strip(), "",
         "Mean over 10 images per row; unrolled values are also averaged over the 5 networks. PSNR in dB / SSIM.", ""]
    for name, phantoms in families.items():
        phantoms = np.stack(phantoms)
        clean = U.recordings(phantoms, n_s, domain, medium, sensors)
        L.append(name)
        for label, rel_std in LEVELS:
            Y = U.add_relative_noise(clean, [rel_std] * len(clean), np.random.default_rng(SEED + int(rel_std * 100)))
            out = U.reconstruct(models, physics, Y, n_s)
            for k in U.SENSOR_COUNTS:
                m = np.nonzero(n_s == k)[0]
                tik = physics.solvers[k].solve(np.stack([Y[i] for i in m]), selection[f"{label}|{k}"]["selected_mu"])
                tik = tik.reshape(len(m), U.GRID_SIZE, U.GRID_SIZE).astype(np.float32)
                cells = [f"Tikhonov {np.mean([psnr(tik[j], phantoms[i]) for j, i in enumerate(m)]):.1f} / "
                         f"{np.mean([ssim(tik[j], phantoms[i]) for j, i in enumerate(m)]):.3f}"]
                for v in U.VARIANTS:
                    p = np.mean([[psnr(out[f"{v}_seed{s}"][i], phantoms[i]) for i in m] for s in U.TRAIN_SEEDS])
                    q = np.mean([[ssim(out[f"{v}_seed{s}"][i], phantoms[i]) for i in m] for s in U.TRAIN_SEEDS])
                    cells.append(f"unrolled, {v}-trained {p:.1f} / {q:.3f}")
                L.append(f"  noise {label:9s} {k:2d} sensors: " + " | ".join(cells))
        L.append("")
    with open(OUT, "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
