# Photoacoustic Reconstruction — Model-Based vs. Learned Inversion Under Sparse-View Sensing

**Status: PLANNED — architecture and repository scaffolding complete; implementation not yet
started.** No forward model, baseline, learned model, experiment, or figure in this repository is
implemented or run yet. Nothing here should be cited as completed work.

## What this project is

A comparative study of classical model-based reconstruction (time-reversal, Tikhonov
regularisation) against a learned reconstruction network for photoacoustic (optoacoustic)
tomography, simulated entirely on synthetic phantoms under a sparse-view sensor array. See
[`ARCHITECTURE.md`](ARCHITECTURE.md) for the technical plan and [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)
for the exact, dependency-ordered build sequence.

## Why this project

Originated from `01_Applications/PhD/Germany/TUM/00_Strategy/TUM_2026_Project_Portfolio_Strategy.md`
(Section 7, Project ①) as the single highest cross-application-value project identified across
seven TUM/Helmholtz PhD applications — it is Direct evidence for the CBI/Jüstel computational-
imaging position (optoacoustic reconstruction is literally that group's stated project), Strong
evidence for Heckel's data-centric image-reconstruction lab, and Moderate evidence for Quaini
(PDE-governed simulation) and the Helmholtz/IBMI data-science position.

## Planned scope (see ARCHITECTURE.md and IMPLEMENTATION_PLAN.md for full detail)

- **MVP:** forward model (`j-Wave`, verified fit — see ARCHITECTURE.md §5) + time-reversal baseline
  + one U-Net refinement model, one phantom family, two sparsity levels.
- **Strong version:** add Tikhonov baseline (via gradient descent through the differentiable forward
  model), full sparsity sweep, multiple phantom families, artefact analysis, written report.
- **Optional research extension:** MC-dropout/ensemble uncertainty quantification.

## Repository layout

```
photoacoustic-reconstruction/
├── README.md            this file
├── ARCHITECTURE.md       technical plan (problem, formulation, methods, evaluation) — verified
│                          library choice as of Sep 2026 architecture review
├── IMPLEMENTATION_PLAN.md  dependency-ordered build sequence, MVP scope, ranked technical risks
├── requirements.txt      planned dependencies (not yet installed/pinned against a working env)
├── src/                  module skeletons — signatures and docstrings only, no implementation
├── configs/              planned experiment configuration (draft, unvalidated)
├── data/                 synthetic phantom generation will live here — no data yet
├── experiments/          empty — experiment scripts land here once src/ is implemented
├── scripts/              empty — CLI entry points land here
├── tests/                empty — unit tests land here alongside implementation
└── report/               empty — the written technical report lands here once results exist
```

## Explicitly not yet true

- No dependency has been installed or version-pinned against a working environment.
- No forward simulation has been run.
- No baseline has been computed.
- No network has been trained.
- No figure, metric, or number in this repository is real. Any numbers appearing in
  `ARCHITECTURE.md` are targets/plans, never results, and are labelled as such.
