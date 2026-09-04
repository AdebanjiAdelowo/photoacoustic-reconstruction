# Photoacoustic Reconstruction — Model-Based vs. Learned Inversion Under Sparse-View Sensing

**Status: PLANNED — architecture and repository scaffolding complete; implementation not yet
started.** No forward model, baseline, learned model, experiment, or figure in this repository is
implemented or run yet. Nothing here should be cited as completed work.

## What this project is

A comparative study of classical model-based reconstruction (filtered back-projection, Tikhonov
regularisation) against a learned reconstruction network for photoacoustic (optoacoustic)
tomography, simulated entirely on synthetic phantoms under a sparse-view sensor array. See
[`ARCHITECTURE.md`](ARCHITECTURE.md) for the full technical plan.

## Why this project

Originated from `01_Applications/PhD/Germany/TUM/00_Strategy/TUM_2026_Project_Portfolio_Strategy.md`
(Section 7, Project ①) as the single highest cross-application-value project identified across
seven TUM/Helmholtz PhD applications — it is Direct evidence for the CBI/Jüstel computational-
imaging position (optoacoustic reconstruction is literally that group's stated project), Strong
evidence for Heckel's data-centric image-reconstruction lab, and Moderate evidence for Quaini
(PDE-governed simulation) and the Helmholtz/IBMI data-science position.

## Planned scope (see ARCHITECTURE.md for full detail)

- **MVP:** forward model (via `j-Wave` or `k-wave-python`) + filtered back-projection baseline + one
  U-Net refinement model, one phantom family, two sparsity levels.
- **Strong version:** add Tikhonov baseline, full sparsity sweep, multiple phantom families,
  artefact analysis, written report.
- **Optional research extension:** MC-dropout/ensemble uncertainty quantification.

## Repository layout

```
photoacoustic-reconstruction/
├── README.md            this file
├── ARCHITECTURE.md       full technical plan (problem, formulation, methods, evaluation)
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
