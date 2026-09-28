"""Diagnostics behind two statements in the README's Tikhonov and model-mismatch discussion.

1. Conditioning of the forward matrices data/tikhonov_A_{16,64}.npy: singular values, condition
   number, and how many singular values fall below 1e-6 of the largest.
2. Oracle weight under sound-speed mismatch (a DIAGNOSTIC, not a result): for noiseless test data
   simulated with c_true = 1500 * (1 + delta) and reconstructed with the nominal matrix, the best
   mean PSNR any weight on the grid of scripts/tikhonov_evaluation.py achieves, choosing the weight on
   the same test images. This uses test ground truth and must never be reported as performance; it
   only shows how much of the noiseless collapse is due to the weight chosen on matched data.
   First 100 test images per sensor count (by index in data/test_expanded.npz).

    python scripts/tikhonov_diagnostics.py   ->   report/tikhonov_diagnostics.txt
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from jwave.geometry import TimeAxis  # noqa: E402

from scripts.expanded_evaluation import SENSOR_COUNTS, TEST_PATH  # noqa: E402
from scripts.generate_training_data import CENTRE, GRID_SIZE, RADIUS  # noqa: E402
from scripts.model_mismatch_evaluation import simulate  # noqa: E402
from scripts.tikhonov_evaluation import MU_GRID  # noqa: E402
from src.evaluate import psnr  # noqa: E402
from src.forward_model import DEFAULT_SOUND_SPEED, build_domain_and_medium, sparse_view_sensor_array  # noqa: E402
from src.tikhonov import Tikhonov  # noqa: E402

N_DIAG = 100
DELTAS = (0.01, 0.02)


def main():
    lines = ["Tikhonov diagnostics (scripts/tikhonov_diagnostics.py)", ""]
    solvers = {k: Tikhonov(np.load(f"data/tikhonov_A_{k}.npy")) for k in SENSOR_COUNTS}
    for k, tk in solvers.items():
        s = np.linalg.svd(tk.A, compute_uv=False)
        lines.append(f"{k} sensors: A {tk.A.shape[0]} x {tk.A.shape[1]}; singular values {s[0]:.4g} to {s[-1]:.4g}; "
                     f"condition number {s[0] / s[-1]:.3g}; {(s < 1e-6 * s[0]).sum()} of {len(s)} below 1e-6 of "
                     f"the largest; numerical rank (numpy default tolerance) {(s > s[0] * max(tk.A.shape) * np.finfo(float).eps).sum()}")
    lines += ["", f"Oracle-weight diagnostic (weight chosen on the SAME test images; not a valid result), noiseless, "
                  f"first {N_DIAG} test images per sensor count:"]
    test = np.load(TEST_PATH)
    domain, medium_nom = build_domain_and_medium(GRID_SIZE)
    time_axis = TimeAxis.from_medium(medium_nom, cfl=0.3)
    sensors = {k: sparse_view_sensor_array(k, RADIUS, CENTRE) for k in SENSOR_COUNTS}
    for delta in DELTAS:
        _, medium_true = build_domain_and_medium(GRID_SIZE, sound_speed=DEFAULT_SOUND_SPEED * (1 + delta))
        for k in SENSOR_COUNTS:
            idx = np.nonzero(test["n_sensors"] == k)[0][:N_DIAG]
            Y = np.stack([simulate(test["phantom"][i], domain, medium_true, time_axis, sensors[k])[..., 0].reshape(-1)
                          for i in idx])
            scores = {}
            for mu in MU_GRID:
                est = solvers[k].solve(Y, mu).reshape(len(idx), GRID_SIZE, GRID_SIZE).astype(np.float32)
                scores[mu] = float(np.mean([psnr(e, test["phantom"][i]) for e, i in zip(est, idx)]))
            best = max(scores, key=scores.get)
            lines.append(f"  delta +{delta:.0%}, {k} sensors: best mean PSNR {scores[best]:.2f} dB at mu {best:.0e}; "
                         + ", ".join(f"{mu:.0e} {v:.2f}" for mu, v in scores.items()))
    with open("report/tikhonov_diagnostics.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
