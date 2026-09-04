"""Stage 5 — training-data generation.

Scope decision (documented, not silent): rather than training two separate models (one per MVP
sparsity setting), each generated example is assigned one of the two MVP sparsity settings at
random (fixed seed for reproducibility), and a single U-Net is trained across both. This keeps the
MVP to one model while still directly supporting evaluation at both defined sparsity settings, and
additionally tests whether a single refinement network generalises across sparsity levels it saw
during training — a reasonable, low-risk scope choice for the MVP, recorded here rather than in
ARCHITECTURE.md since it does not change the project's formulation, only the training-data
protocol.

Output: data/{split}.npz containing arrays `phantom` (N, size, size), `recon` (N, size, size)
[time-reversal baseline output], `n_sensors` (N,) [which sparsity setting each example used],
`seed` (N,) [phantom seed, for traceability]. Not committed to git (data/*.npz is gitignored).
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baselines import time_reversal_reconstruction
from src.forward_model import build_domain_and_medium, simulate_sensor_data, sparse_view_sensor_array
from src.phantoms import random_blob_phantom

GRID_SIZE = 64
CENTRE = (GRID_SIZE // 2, GRID_SIZE // 2)
RADIUS = 24
SPARSITY_SETTINGS = [16, 64]  # the two MVP sparsity levels, from configs/mvp.yaml

SPLITS = {
    "train": 40,
    "val": 8,
    "test": 8,
}
# Deterministic, non-overlapping seed ranges per split — prevents leakage between splits by
# construction, not just by convention.
SPLIT_SEED_OFFSETS = {"train": 0, "val": 10_000, "test": 20_000}


def generate_split(split_name: str, n_examples: int):
    rng = np.random.default_rng(SPLIT_SEED_OFFSETS[split_name])
    domain, medium = build_domain_and_medium(GRID_SIZE)

    phantoms = np.zeros((n_examples, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    recons = np.zeros((n_examples, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    n_sensors_used = np.zeros(n_examples, dtype=np.int32)
    seeds_used = np.zeros(n_examples, dtype=np.int64)

    t0 = time.time()
    for i in range(n_examples):
        phantom_seed = int(SPLIT_SEED_OFFSETS[split_name] + i)
        n_blobs = int(rng.integers(1, 4))
        n_sensors = int(rng.choice(SPARSITY_SETTINGS))

        phantom = random_blob_phantom(size=GRID_SIZE, seed=phantom_seed, n_blobs=n_blobs)
        sensor_pos = sparse_view_sensor_array(n_sensors, RADIUS, CENTRE)
        recording, time_axis = simulate_sensor_data(phantom, domain, medium, sensor_pos)
        recon = time_reversal_reconstruction(recording, sensor_pos, domain, medium, time_axis)

        phantoms[i] = phantom
        recons[i] = recon
        n_sensors_used[i] = n_sensors
        seeds_used[i] = phantom_seed

        if (i + 1) % 10 == 0 or i == n_examples - 1:
            print(f"  [{split_name}] {i + 1}/{n_examples} ({time.time() - t0:.1f}s elapsed)")

    return phantoms, recons, n_sensors_used, seeds_used


def main():
    import os
    os.makedirs("data", exist_ok=True)

    for split_name, n_examples in SPLITS.items():
        print(f"Generating split '{split_name}' ({n_examples} examples)...")
        phantoms, recons, n_sensors_used, seeds_used = generate_split(split_name, n_examples)
        out_path = f"data/{split_name}.npz"
        np.savez(out_path, phantom=phantoms, recon=recons,
                  n_sensors=n_sensors_used, seed=seeds_used)
        print(f"  saved {out_path}: phantom {phantoms.shape}, recon {recons.shape}")

    # Leakage check: verify no seed appears in more than one split.
    all_seeds = []
    for split_name in SPLITS:
        d = np.load(f"data/{split_name}.npz")
        all_seeds.append(set(d["seed"].tolist()))
    for i, s1 in enumerate(list(SPLITS.keys())):
        for s2 in list(SPLITS.keys())[i + 1:]:
            overlap = all_seeds[list(SPLITS.keys()).index(s1)] & all_seeds[list(SPLITS.keys()).index(s2)]
            assert not overlap, f"Seed overlap between {s1} and {s2}: {overlap}"
    print("No seed overlap between splits — verified.")


if __name__ == "__main__":
    main()
