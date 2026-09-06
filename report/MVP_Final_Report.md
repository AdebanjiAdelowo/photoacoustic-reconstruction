---
title: "Photoacoustic Reconstruction MVP Report"
subtitle: "PLANNED to MVP COMPLETE — Implementation, Verification, and Results"
author: "Adebanji Adelowo"
date: "4 September 2026"
---

## Implementation status

Every stage in `IMPLEMENTATION_PLAN.md` up through the MVP is real, executed, and verified — not
scaffolded. `src/phantoms.py`, `forward_model.py`, `baselines.py`, `reconstruction_net.py`,
`evaluate.py` all contain working implementations (no `NotImplementedError` remaining). The full
pipeline runs end-to-end: **phantom → forward simulation → sparse measurements → time-reversal →
learned reconstruction → PSNR/SSIM evaluation.**

## Verification evidence

- **Stage 1:** dedicated conda env (`photoacoustic`, Python 3.11), `jwave`/`jax`/`torch`/
  `scikit-image` installed; API directly introspected via `inspect` (not trusted from memory or
  the earlier web search) — this surfaced real facts the original architecture got wrong or left
  unspecified (PML boundary, not periodic; `simulate_wave_propagation`'s actual return shape).
  Smoke test passed.
- **Stage 2:** phantom generator — 5/5 tests pass, visually inspected. Design corrected from
  "Shepp-Logan" to seeded Gaussian blobs (physically appropriate reasoning documented).
- **Stage 3:** forward model — 8/8 tests pass. One real test failure diagnosed (sensor-array
  radius too small, sensor sat in phantom's Gaussian tail) and fixed, not hidden. Independent
  physical validation: the sensor trace shows a textbook bipolar photoacoustic pulse, which
  emerged from the physics rather than being designed in.
- **Stage 4:** time-reversal baseline — 12/12 tests pass. One real API misunderstanding (`on_grid`
  returns the full trajectory, not just the final step) diagnosed and fixed. 0.929 correlation
  with ground truth on a real reconstruction.
- **Stage 5:** training-data pipeline — 56 examples generated in 12.5s, zero seed overlap across
  splits (verified programmatically, not just by convention).
- **Stage 6:** U-Net — overfit sanity check passed (loss dropped 99.8% on a fixed batch) *before*
  the full run, per the required gate. Full training: val loss 0.0118 → 0.0005 over 60 epochs
  (5.2s).
- **Stage 7:** evaluation — 15/15 tests pass overall. Real metrics computed on the held-out test
  set.

## Results (real, measured)

| sparsity | TR PSNR | TR SSIM | Learned PSNR | Learned SSIM |
|---|---|---|---|---|
| 16 sensors | 18.97 dB | 0.706 | **30.73 dB** | 0.500 |
| 64 sensors | 22.43 dB | 0.735 | **33.08 dB** | 0.517 |

## Failures/limitations — reported honestly, not hidden

**PSNR and SSIM disagree on which method wins.** The learned model wins decisively on PSNR and,
unambiguously, visually (streak artifacts almost entirely removed — `report/mvp_comparison.png`).
It *loses* on whole-image SSIM. This was investigated rather than picking the favorable metric:
within the true structure, learned SSIM = 0.936 vs. 0.188 for time-reversal (confirms the visual
win); in the background (>85% of image area), the U-Net leaves a small residual "haze" (mean
0.0099 vs. near-zero for both ground truth and time-reversal) that SSIM's local-variance
sensitivity penalizes heavily, inverting the whole-image average. This is a real, diagnosed, known
failure mode of MSE-trained restoration networks — not fixed in this pass (that would be
strong-version scope), documented in `report/mvp_results.txt` and `report/mvp_ssim_diagnosis.png`.

Two minor process issues, both disclosed and fixed forward-only (no history rewrite): a
`.gitignore` gap let `data/*.npz` get committed once before being caught and untracked; a test's
sensor radius needed correcting for the same underlying geometric reason as the Stage 3 fix.

## Repository changes

25 files changed, +1196/−138 lines across 9 commits (`7270cd6`..`7f37d50`) — full stage-by-stage
detail in `IMPLEMENTATION_LOG.md`. New: `IMPLEMENTATION_LOG.md`,
`scripts/{smoke_test,generate_training_data,train,evaluate_mvp}.py`,
`tests/test_{forward_model,baselines,evaluate}.py`, `report/{dev_*,mvp_*}` figures and results.
Modified: `ARCHITECTURE.md` (2 corrections found during implementation), `README.md`,
`requirements.txt` (pinned), all `src/*.py` (from stubs to real implementations).

## MVP verdict

**MVP COMPLETE.**

---

*Prepared 4 September 2026, documenting the implementation pass that moved this project from
PLANNED to MVP COMPLETE. See `IMPLEMENTATION_LOG.md` in this repository for the full
stage-by-stage verification record.*
