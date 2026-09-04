# Implementation Log

Format per stage: **Stage → implementation → verification performed → result → issues → decision.**
Status tags used throughout: **VERIFIED**, **PARTIALLY VERIFIED**, **FAILED**, **NOT YET ATTEMPTED**.

Project status: **IN PROGRESS** (was PLANNED as of the last architecture-review pass).

---

## Stage 1 — Environment & j-Wave verification — **VERIFIED**

**Implementation:** created a dedicated conda environment (`photoacoustic`, Python 3.11), installed
`numpy`, `scipy`, `matplotlib`, `scikit-image`, `torch`, `pytest`, then `jwave` (which pulled in
`jax`, `jaxlib`, `jaxdf` as dependencies). No errors during install.

**Verification performed:**
- Confirmed all imports succeed and printed exact installed versions (recorded in
  `requirements.txt`, now pinned rather than placeholder).
- Confirmed `jax.devices()` returns a CPU device — no GPU/Metal required or used.
- **Directly introspected the installed `jwave` API** via `inspect.signature`/`dir()` rather than
  trusting memory or the earlier web-search-based architecture review. This surfaced real facts not
  previously known:
  - `Medium` uses a **PML absorbing boundary** (`pml_size` parameter, default 20.0), not a periodic
    boundary — the physically correct choice for open-domain photoacoustic simulation, and worth
    recording explicitly since `ARCHITECTURE.md` did not previously specify the boundary treatment.
  - `simulate_wave_propagation(medium, time_axis, p0=..., sensors=...)` is a direct one-call forward
    solver — no separate "wrapper" logic needed beyond constructing `Domain`/`Medium`/`TimeAxis`/
    `FourierSeries`.
  - `TimeAxis.from_medium(medium, cfl=0.3)` derives the time step from a CFL condition automatically
    — no manual `dt` calculation needed.
  - `geometry.points_on_circle(n, radius, centre)` exists and is exactly what's needed for the
    sparse-view circular sensor array (§5 of `ARCHITECTURE.md`).
  - The docstring states this implementation "is equivalent to the `kspaceFirstOrderND` function in
    the k-Wave Toolbox" — independent confirmation this is the same class of solver the field's
    standard toolbox uses.
- Wrote `scripts/smoke_test.py`: builds a 64×64 domain, a homogeneous 1500 m/s medium, a Gaussian
  blob as p0, a 16-sensor circular array, and calls `simulate_wave_propagation`.

**Result:** ran successfully. Recording shape `(297, 16, 1)` (time × sensors × field-component),
finite everywhere, non-zero (max ≈ 0.148, min ≈ −0.074), 1.5s wall-clock on CPU.

**Issues encountered and resolved:** first run failed an assertion because I assumed the sensor
dimension was the *last* array axis; the actual (empirically observed) layout is
`(Nt, n_sensors, 1)`. Fixed the assertion, reran, passed. This is recorded here rather than hidden
— exactly the kind of API-detail mismatch `ARCHITECTURE.md`'s own "do not trust unverified API
names" caveat anticipated.

**Decision:** environment and library choice confirmed working. Proceed to Stage 2.

---

## Stage 2 — Phantom generation — **VERIFIED**

**Design correction (implementation evidence, not assumed in advance):** the original plan
(`ARCHITECTURE.md` §4) said "Shepp-Logan-style" phantoms. On implementation, this was corrected:
Shepp-Logan is a CT X-ray-attenuation phantom (a stylised anatomical cross-section), not a
physically appropriate model for a photoacoustic initial-pressure source (a localised optical
absorber). Implemented `random_blob_phantom(size, seed, n_blobs)` instead — a small number of
seeded Gaussian blobs, normalised to [0, 1]. `ARCHITECTURE.md` §4 updated to match (see diff in
this commit).

**Implementation:** `src/phantoms.py::random_blob_phantom`. Documented coordinate convention
(numpy `'ij'` indexing, origin at `[0,0]`) and value convention (dimensionless amplitude in
`[0,1]`, not calibrated to real units) directly in the module docstring.

**Verification performed:**
- `tests/test_phantoms.py` — 5 tests (shape/dtype, value range, reproducibility given a fixed
  seed, distinctness across seeds, distinctness across `n_blobs`). All 5 **PASSED**.
- Generated and visually inspected 3 sample phantoms (seeds 0, 1, 2) at 128×128 — saved to
  `report/dev_phantom_samples.png` and viewed directly. Confirmed: plausible localised-blob
  structures, correct value range (black background = 0, bright blobs up to 1.0), visually
  distinct across seeds.

**Result:** phantom generator is deterministic, reproducible, and produces physically sensible
initial-pressure-style images.

**Issues encountered:** none beyond the design correction above.

**Decision:** proceed to Stage 3 (forward model), using this phantom generator.

---

## Stage 3 — Forward acoustic model — **VERIFIED**

**Implementation:** `src/forward_model.py` — `build_domain_and_medium`, `sparse_view_sensor_array`,
`simulate_sensor_data`, built on the directly-introspected `j-Wave` API from Stage 1. Documented in
the module docstring: grid/spacing, PSTD discretisation, CFL-derived time step, homogeneous sound
speed, **PML absorbing boundary** (not periodic — confirmed from `jwave.geometry.Medium`'s real
signature), p0/u0 initial-value convention, circular sensor geometry, and the verified
`(Nt, n_sensors, 1)` output layout. Defined explicitly what "sparse-view" means in this project:
fewer sensors on the same fixed-radius circular array (angular under-sampling), not a reduced
radius or limited-angle arc.

**Verification performed:**
- Ran the forward model on a real phantom (from Stage 2, seed 0, 64×64 grid, 32 sensors at
  radius 24) — recording shape `(297, 32, 1)`, finite, max |signal| ≈ 0.216.
- **Physical plausibility check, visual:** plotted the sensor-0 pressure trace over time
  (`report/dev_forward_model_check.png`) and inspected it directly. Result: a smooth bipolar pulse
  (rises to a positive peak, crosses zero, dips to a negative trough, decays to zero) — this is the
  textbook photoacoustic transient-pressure signature for a smooth absorber. This shape was not
  designed or assumed in advance; it emerged from the physics, which is meaningful independent
  evidence the simulation is doing something physically correct, not just numerically stable.
- Wrote `tests/test_forward_model.py` (3 tests: shape/finiteness, non-trivial signal, near-zero
  signal before wave arrival).

**Issues encountered and resolved (real failure, documented rather than hidden):** the first test
run of `test_signal_starts_near_zero_before_wave_arrival` **FAILED** — one sensor (index 1, at
radius=12 on a 32×32 test grid) showed amplitude 0.036 at t=0, not near-zero. Diagnosed by printing
sensor positions and the phantom's bright-region bounding box: that sensor position fell within the
non-negligible Gaussian tail of the phantom itself (Gaussians have infinite support), i.e. the
sensor was geometrically placed too close to/inside the source region for this small test's
radius — a test-configuration issue, not a forward-model bug. Fixed by increasing the test's sensor
radius from 12 to 15 (verified empirically this clears the phantom's tail: max t=0 amplitude dropped
from 0.036 to 0.00039). Confirmed the MVP-scale configuration (grid 64, radius 24) does not have
this issue, since blob placement range and sensor radius are proportionally further apart at that
scale.

**Result:** forward model verified working, physically plausible, and correctly separates
source and sensor regions at MVP scale. All 8 tests (5 phantom + 3 forward-model) **PASS**.

**Decision:** proceed to Stage 4/5 (sensor geometry is already implemented as part of this stage;
next is the classical time-reversal reconstruction baseline).

---

## Stage 4 — Classical reconstruction baseline (time-reversal) — **VERIFIED**

**API investigation before implementation** (per instruction — do not trust API names from
memory): `jwave`'s top-level namespace has no dedicated "time reversal" function. Introspected
`jwave.geometry.Sources` directly (`inspect.getsource`) — confirmed it takes
`(positions, signals, dt, domain)` and injects `signals[:, n]` at grid `positions` at time step
`n`. Standard time-reversal is therefore: reverse the recorded signals in time, inject them as
`Sources` into a second `simulate_wave_propagation` call with **zero initial pressure**, and read
out the field at the *final* time step.

**Empirical discovery during implementation:** calling `simulate_wave_propagation(..., sensors=
None)` does **not** return only the final-time field, as first assumed — it returns a
`FourierSeries` object whose `.on_grid` property is the **entire field trajectory**, shape
`(Nt, grid, grid, 1)`. Corrected the implementation to explicitly index `[-1]` after discovering
this (an `np.asarray()` call on the un-indexed object initially produced a nonsensical shape `()`,
which surfaced the misunderstanding immediately).

**Implementation:** `src/baselines.py::time_reversal_reconstruction(recording, sensor_positions,
domain, medium, time_axis)` — note the function signature does not accept the phantom/ground truth
at all, a structural guarantee against ground-truth leakage (verified by
`test_no_ground_truth_leakage_by_construction`).

**Verification performed:**
- Ran on a real two-blob phantom (64×64, 32 sensors, radius 24). **Correlation with ground truth:
  0.929.**
- Visual inspection (`report/dev_timereversal_check.png`): both blobs recovered in approximately
  correct position and relative brightness, with faint circular/arc streak artefacts around them.
  This is the well-known, expected signature of time-reversal reconstruction under discrete/
  sparse sensor sampling in the photoacoustic-tomography literature — not designed or assumed in
  advance, and independent evidence the pipeline is physically correct rather than merely
  numerically stable.
- `tests/test_baselines.py` — 4 tests: no-leakage-by-construction, shape/finiteness, correlation
  with ground truth (>0.5 threshold), and — the central research-relevant check — that a sparser
  16-sensor array does not reconstruct *better* than a 64-sensor array (within numerical
  tolerance). **All 4 PASS.**

**Result:** the full pipeline phantom → forward simulation → sparse measurements → time-reversal
now works end-to-end, with a real, physically-sensible reconstruction and a confirmed sparsity
effect (the core research question the MVP exists to test). All 12 tests across all stages
**PASS**.

**Issues encountered:** the `on_grid` shape misunderstanding above — resolved by direct empirical
inspection before finalising the implementation, not by guessing.

**Decision:** proceed to Stage 5 (training-data pipeline) and Stage 6 (learned model). Tikhonov
(originally planned as a second baseline) remains explicitly deferred past the MVP.

---

## Stage 5 — Training-data pipeline — **VERIFIED**

**Scope decision (documented, not silent):** rather than training two separate models (one per
MVP sparsity setting), each generated example is randomly (but seeded/reproducibly) assigned one
of the two sparsity settings (16 or 64 sensors), and a single U-Net is trained across both —
recorded in `scripts/generate_training_data.py`'s own docstring, not in `ARCHITECTURE.md`, since
it is a training-data protocol choice, not a change to the project's formulation.

**Implementation:** `scripts/generate_training_data.py`. Deterministic, non-overlapping seed
ranges per split (`train`: seeds 0–39, `val`: 10000–10007, `test`: 20000–20007) — leakage
prevented by construction, and additionally verified programmatically (the script asserts no seed
appears in more than one split's saved `.npz`).

**Verification performed:**
- Ran for real: 40 train + 8 val + 8 test = 56 examples generated in **12.5s** total (well within
  the "minutes, not hours" scope-control target in `IMPLEMENTATION_PLAN.md`).
- Confirmed sparsity settings are reasonably balanced by chance (16 vs. 64 roughly 50/50 in the
  printed counts).
- **Programmatic leakage check passed**: no seed shared across splits.
- Visually inspected 4 train examples (`report/dev_dataset_samples.png`): dense (64-sensor)
  reconstructions are visibly cleaner than sparse (16-sensor) ones, which show pronounced
  streak/starburst artefacts around the true blob positions — confirms the dataset presents the
  U-Net with a genuine, visually obvious task (artefact removal), not a trivial one.

**Result:** `data/train.npz`, `data/val.npz`, `data/test.npz` created (not committed — gitignored
per `IMPLEMENTATION_PLAN.md` Stage 7's "avoid committing large generated data" instruction).

**Issues encountered and resolved:** first run failed with `ModuleNotFoundError: No module named
'src'` — running a script directly (`python scripts/foo.py`) does not add the repo root to
`sys.path` the way `pytest` does. Fixed with an explicit `sys.path.insert` based on the script's
own file location.

**Decision:** proceed to Stage 6 (learned model), training on this dataset.

---
