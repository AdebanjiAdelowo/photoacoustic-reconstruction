# Photoacoustic Reconstruction — Model-Based vs. Learned Inversion Under Sparse-View Sensing

**Status: MVP COMPLETE.** The full pipeline (phantom → forward simulation → sparse measurements →
time-reversal → learned reconstruction → PSNR/SSIM evaluation) is implemented and verified with
real execution — see [`IMPLEMENTATION_LOG.md`](IMPLEMENTATION_LOG.md) for the stage-by-stage
verification record and [`report/mvp_results.txt`](report/mvp_results.txt) for real, measured
results. **Not started:** Tikhonov baseline, uncertainty quantification, multiple phantom families,
full sparsity sweep — all explicitly out of MVP scope, deferred to a future "strong version" pass.

## What this project is

A comparative study of classical model-based reconstruction (time-reversal; Tikhonov regularisation
planned but not yet implemented) against a learned reconstruction network for photoacoustic
(optoacoustic) tomography, simulated on synthetic phantoms under a sparse-view sensor array. See
[`ARCHITECTURE.md`](ARCHITECTURE.md) for the technical plan and [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)
for the build sequence that was actually followed.

## Real MVP result (measured, not projected)

| sparsity | TR PSNR | TR SSIM | Learned PSNR | Learned SSIM |
|---|---|---|---|---|
| 16 sensors | 18.97 dB | 0.706 | **30.73 dB** | 0.500 |
| 64 sensors | 22.43 dB | 0.735 | **33.08 dB** | 0.517 |

The learned model wins decisively on PSNR and, more importantly, visually (see
[`report/mvp_comparison.png`](report/mvp_comparison.png) — streak artefacts are almost entirely
removed). It loses on whole-image SSIM, which was investigated rather than hidden: within the true
structure the learned model's SSIM is 0.936 vs. 0.188 for time-reversal (dramatically better), but
it introduces a small, diagnosed background "haze" that SSIM's local-variance sensitivity penalises
heavily since background dominates the image by area. Full diagnosis in
[`report/mvp_results.txt`](report/mvp_results.txt) and
[`report/mvp_ssim_diagnosis.png`](report/mvp_ssim_diagnosis.png).

## Why this project

Originated from `01_Applications/PhD/Germany/TUM/00_Strategy/TUM_2026_Project_Portfolio_Strategy.md`
(Section 7, Project ①) as the single highest cross-application-value project identified across
seven TUM/Helmholtz PhD applications — it is Direct evidence for the CBI/Jüstel computational-
imaging position (optoacoustic reconstruction is literally that group's stated project), Strong
evidence for Heckel's data-centric image-reconstruction lab, and Moderate evidence for Quaini
(PDE-governed simulation) and the Helmholtz/IBMI data-science position. **Not yet added to any
application CV** — that decision is deferred to the portfolio-verdict step, per instruction.

## Environment

Conda env `photoacoustic`, Python 3.11. Verified working dependency versions pinned in
`requirements.txt`. To reproduce: `conda create -n photoacoustic python=3.11 && conda activate
photoacoustic && pip install -r requirements.txt`.

## Repository layout

```
photoacoustic-reconstruction/
├── README.md               this file
├── ARCHITECTURE.md          technical plan — updated with corrections found during implementation
├── IMPLEMENTATION_PLAN.md   the dependency-ordered build sequence that was followed
├── IMPLEMENTATION_LOG.md    stage-by-stage verification record: what was run, what passed/failed,
│                             how failures were diagnosed and fixed, real measured results
├── requirements.txt         pinned, verified working dependency versions
├── src/                     phantoms.py, forward_model.py, baselines.py, reconstruction_net.py,
│                             evaluate.py — all implemented, not skeletons
├── scripts/                 smoke_test.py, generate_training_data.py, train.py, evaluate_mvp.py
├── configs/                 mvp.yaml
├── data/                    generated train/val/test .npz splits — gitignored, not committed
├── experiments/              trained checkpoint — gitignored, not committed
├── tests/                   15 tests, all passing
└── report/                  real figures and results: dev-stage verification plots +
                              mvp_comparison.png, mvp_ssim_diagnosis.png, mvp_results.txt
```

## What's next

Strong version (Tikhonov baseline, full sparsity sweep, multiple phantom families) and the
background-haze failure mode are both natural next steps — neither was started in this pass, per
its explicit stop condition. See `IMPLEMENTATION_LOG.md`'s final entry for the full portfolio
verdict and recommended next action.
