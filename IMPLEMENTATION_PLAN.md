# Implementation Plan — PLANNED → Working MVP

This is the sequencing document for turning `ARCHITECTURE.md`'s plan into real, running code. Each
step lists what it depends on, what "done" looks like, and what to check before moving to the next
step. Nothing in this file is a result — it is a checklist.

**MVP scope, deliberately kept small (do not exceed this until it works):**
one phantom family (Shepp-Logan) + one working forward model (j-Wave) + one classical
reconstruction (time-reversal) + one learned reconstruction (U-Net refinement) + two sparsity
settings + PSNR/SSIM. No uncertainty quantification, no multiple phantom families, no unrolled
iterative reconstruction, no Tikhonov baseline yet — those come after the MVP works (Tikhonov is
part of the "strong version," not MVP, per ARCHITECTURE.md §6's E1/E2 split; adding it to the MVP
checklist below only as step 5b, clearly marked optional-for-MVP).

## 1. Environment / dependencies

- Install: `numpy`, `scipy`, `matplotlib`, `jwave`, `torch`, `scikit-image`, `pytest`.
- **Verification gate:** before writing any project code, run `jwave`'s own quick-start/example
  from its official documentation (ucl-bug.github.io/jwave) as a smoke test. If that doesn't run
  cleanly, stop and resolve the environment issue before proceeding — everything downstream depends
  on this working.
- No GPU/Metal required for the MVP's small 2D grids; CPU-only JAX is expected to be sufficient.

## 2. Phantom generation (`src/phantoms.py`)

- Implement `shepp_logan_phantom(size)` only — defer the DRIVE-vessel-mask variant (explicitly
  optional in ARCHITECTURE.md §4) until after the MVP.
- **Done when:** a single call produces a 2D array of the expected shape and value range, visually
  inspected via `matplotlib` to confirm it looks like a phantom, not noise.
- Depends on: step 1 (numpy/matplotlib only — does not depend on jwave).

## 3. Forward acoustic model (`src/forward_model.py`)

- Wrap j-Wave's homogeneous-medium initial-value-problem setup (constant sound speed) with the
  phantom from step 2 as $p_0$. Follow j-Wave's official example notebook structure rather than the
  illustrative pseudocode in ARCHITECTURE.md.
- **Verification gate:** before adding sensors, confirm the raw wave simulation itself looks
  physically sane (energy propagates outward from the phantom, doesn't blow up numerically) via a
  simple animation/snapshot plot. This is a real correctness check, not optional.
- Depends on: step 1's smoke test passing, step 2's phantom.

## 4. Measurement geometry (`src/forward_model.py::sparse_view_sensor_array`)

- Circular sensor array around the domain, parameterised by sensor count $K$.
- Implement the two MVP sparsity levels from `configs/mvp.yaml` (currently drafted as
  $K \in \{16, 64\}$ — revisit these exact numbers once step 3's grid size is finalised, since
  sensor count should scale sensibly with domain size).
- Depends on: step 3 (needs a working forward model to record from).

## 5. Classical reconstruction baseline(s) (`src/baselines.py`)

- **5a — Time-reversal (required for MVP):** implement via j-Wave's documented reconstruction
  workflow. This is the MVP's only classical baseline.
- **5b — Tikhonov (optional, defer past MVP):** gradient-descent reconstruction through j-Wave's
  differentiable forward model. Real, nontrivial additional code (needs an optimizer loop, e.g.
  `optax` or plain JAX gradient steps) — do not attempt until 5a and the learned model (step 7) are
  both working, so the project has *a* working comparison before adding complexity.
- Depends on: step 4 (needs sensor data to reconstruct from).

## 6. Training-data generation (`scripts/generate_training_data.py`, new)

- Generate $N$ phantom instances by varying Shepp-Logan ellipse parameters (position, size,
  contrast) within a fixed, documented range.
- For each: run the forward model (step 3) + sensor recording (step 4) + time-reversal baseline
  (step 5a) → save (degraded reconstruction, ground-truth phantom) pairs.
- **Scope control:** keep the grid small (e.g. 64×64 or 128×128) and time-stepping coarse enough
  that generating the full dataset takes minutes, not hours, on a laptop CPU — if it doesn't, that's
  a signal to shrink the grid before generating more data, not a signal to acquire more compute.
- Depends on: steps 2–5a all individually verified working — do not generate a dataset from an
  unverified pipeline.

## 7. Learned reconstruction model (`src/reconstruction_net.py`)

- U-Net refining the time-reversal estimate, architecture adapted from `abdominal-ct-segmentation`'s
  design (changed from a segmentation output — softmax/Dice — to a regression output — linear,
  MSE/L1 loss).
- Train on step 6's pairs. Keep network capacity modest for the MVP's small synthetic dataset to
  avoid overfitting; consider simple augmentation (rotation/flip of phantoms) if the dataset is
  small.
- Depends on: step 6.

## 8. Evaluation (`src/evaluate.py`)

- PSNR/SSIM via `skimage.metrics` (not hand-rolled — see ARCHITECTURE.md §7's correction).
- **Done when:** time-reversal and U-Net reconstructions both have PSNR/SSIM numbers against ground
  truth, for both MVP sparsity levels.
- Depends on: step 7 (needs a trained model to evaluate) and step 5a (needs the baseline to compare
  against).

## 9. Reproducibility

- Fix random seeds (numpy, jax, torch) for phantom generation, network initialisation, and training,
  once the above steps exist — retrofit seeding rather than bolting it on at the very end.
- Document exact package versions actually used once the environment is confirmed working (update
  `requirements.txt` from "planned" to pinned).
- Document forward-model physical parameters actually used (sound speed, grid spacing, sensor
  geometry) in `configs/mvp.yaml`, replacing the current draft/placeholder values with the ones
  actually run.

## 10. Tests (`tests/`)

Minimal, fast sanity tests — not full physics validation:
- `test_phantoms.py`: correct output shape/value range from `shepp_logan_phantom`.
- `test_evaluate.py`: `psnr`/`ssim` return the expected values on identical images (inf / 1.0) and
  on two known-different images.
- `test_forward_model.py`: a trivial sanity check (e.g. a point-like phantom produces a
  roughly-expanding wavefront) — qualitative, not a strict numerical assertion.

## 11. Figures/results to eventually produce (once real, not before)

- Example forward-simulated sensor data visualisation.
- Reconstruction comparison panel: ground truth | time-reversal | U-Net, at both MVP sparsity
  levels.
- A small PSNR/SSIM summary table for the two methods at the two sparsity levels (a full
  sparsity-sweep curve is strong-version scope, E2 in ARCHITECTURE.md §6 — not MVP).

---

## Technical risks, ranked by how early they'd block progress

1. **j-Wave environment setup (step 1)** — highest risk simply because it's first; the smoke-test
   gate exists specifically to surface this immediately rather than after other code is written.
2. **API drift** — this document and ARCHITECTURE.md describe the *workflow*, verified real;
   specific function/class names were not independently verified against current j-Wave source and
   should not be trusted — always check the current official example notebook.
3. **Forward-model correctness (step 3's verification gate)** — a wave simulation that "runs" but is
   numerically wrong (e.g. wrong sign, unstable time step) would silently corrupt everything
   downstream; the qualitative sanity-plot check exists to catch this before building on top of it.
4. **Tikhonov-via-autodiff (step 5b)** — real new code (an optimisation loop), deliberately deferred
   past the MVP rather than risking it blocking the first working comparison.
5. **Dataset generation cost (step 6)** — mitigated by the explicit small-grid scope control; if
   this becomes a bottleneck, shrink the problem rather than escalate compute.
