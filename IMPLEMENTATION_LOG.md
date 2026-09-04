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
