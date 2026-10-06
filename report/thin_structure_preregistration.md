# Thin-structure diagnostic study: pre-registration (phase 1)

This document fixes the design, hypotheses, thresholds, analysis, order of evidence and stopping
rule of the study before any image of its test set is reconstructed. It, the configuration
`configs/thin_structure_study.json`, the generator `src/thin_structures.py` and the runner
`scripts/thin_structure_study.py` were committed before the evaluation. The runner reads every
number from the configuration, refuses to run unless these files are committed and unmodified, and
writes the commit into its results. Where this text and the configuration disagree, the
configuration is authoritative and the disagreement is an error to be reported.

## 1. Question

After the intermediate edge-sharpness gap was repaired, what explains the remaining failure of the
learned reconstruction on extremely thin structures: structural thickness, edge profile, or curved
geometry?

Phase 1 answers this for the ten existing networks by evaluation only. No network is trained.

## 2. Prior evidence

All numbers are committed results at 14 dB SNR (`report/edge_gap_intervention_results.txt`).

- The edge-sharpness intervention (`scripts/edge_gap_intervention.py`, design commit `d19c949`,
  results commit `c1f1b3e`) was supported by its pre-set criteria. Replacing half of the sharp
  training images by edge-blurred copies removed the dip of the mixed-family networks on discs
  blurred by 0.5 to 1 pixel (worst advantage over Tikhonov from +2.7 to +8.7 dB with 16 sensors and
  from +1.5 to +7.0 dB with 64).
- It did not remove the failure on the thinnest vessel-like lines. With 64 sensors, on lines of
  Gaussian width 1 pixel: Tikhonov 36.94 dB, control networks 30.98 dB, intervention networks
  32.97 dB, so the intervention networks remain 3.96 dB behind Tikhonov. At width 1.5 pixels they
  are 1.89 dB ahead. On discs with the same range of edge sharpness they are 7 to 8.5 dB ahead.
- In that vessel sweep one number sets the thickness of the line and the sharpness of its edges, and
  every line is curved. The sweep cannot say which of the three properties the failure follows.
- Between 16 and 64 sensors, Tikhonov gains 8.9 dB on the 1 pixel lines and the intervention
  networks 0.9 dB (on different images: that sweep did not reuse images across sensor counts). A
  change of the advantage over Tikhonov can therefore come from either method.

## 3. Phantoms

`src/thin_structures.py`, `line_phantom(size, seed, core_width, edge_sigma, geometry)`, on the
64 x 64 grid of every other experiment.

**Cross-section.** A flat core of full width w convolved with a Gaussian of standard deviation s
(both in pixels), as a function of the distance d from the centreline, scaled to 1 on the
centreline:

    p(d; w, s) = [erf((w/2 - d) / (sqrt(2) s)) + erf((w/2 + d) / (sqrt(2) s))] / [2 erf(w / (2 sqrt(2) s))]

with two exact limits: for s = 0, p = 1 where d <= w/2 and 0 elsewhere (a sharp-edged bar, the edge
type of the sharp training families); for w = 0, p = exp(-d^2 / (2 s^2)) (the Gaussian line of the
vessel family, of width s).

**Geometry.** Two classes, built from the same random draws for a given seed.

- *Curved*: the centreline of the existing vessel family (`src/shape_phantoms.py`): 1 or 2 lines per
  image, end points uniform in the central region, a sinusoidal bend of up to 0.08 x 64 pixels,
  amplitude 0.5 to 1. With w = 0 the phantom is identical, bit for bit, to
  `vessel_phantom(size, seed, s)`, which a test checks.
- *Straight*: the same end points, amplitudes and number of lines with the bend set to zero. It
  removes curvature and changes nothing else, so location, orientation, chord length and amplitude
  distributions are those of the vessel family, and each straight image is paired with a curved one.

The sharp-edged bar asked for as a third control is the s = 0 column of the straight class. Rings
and other closed shapes are not used: a closed outline adds a further difference and resembles the
boundary of the training discs.

The distance d is taken to the 200 sampled points of the centreline, as in the vessel family, so the
ends of every line are rounded. The image is scaled to peak 1.

## 4. Factor grid

Core width w in {0, 1, 2, 4, 6} pixels, edge width s in {0, 0.5, 1, 1.5, 2} pixels, both
geometries, without the empty cell w = 0, s = 0: 24 profiles x 2 geometries = 48 cells. The
profiles w = 0, s in {1, 1.5, 2} are the 1, 1.5 and 2 pixel points of the earlier vessel sweep.

The table gives, for each profile, the width at half maximum (FWHM), the width over which the
continuous profile is at least 0.95 (plateau), its largest slope, and the mean largest gradient
measured on the straight phantoms of the test set (computed from the phantoms alone; nothing was
reconstructed). All in pixels or per pixel.

| w | s | FWHM | plateau | largest slope | measured gradient |
|---|---|---|---|---|---|
| 0 | 0.5 | 1.18 | 0.32 | 1.21 | 0.61 |
| 0 | 1 | 2.35 | 0.64 | 0.61 | 0.49 |
| 0 | 1.5 | 3.53 | 0.96 | 0.40 | 0.37 |
| 0 | 2 | 4.71 | 1.28 | 0.30 | 0.29 |
| 1 | 0 | 1.00 | 1.00 | grid-limited | 0.70 |
| 1 | 0.5 | 1.39 | 0.38 | 1.04 | 0.60 |
| 1 | 1 | 2.45 | 0.67 | 0.58 | 0.48 |
| 1 | 1.5 | 3.60 | 0.98 | 0.40 | 0.36 |
| 1 | 2 | 4.76 | 1.29 | 0.30 | 0.28 |
| 2 | 0 | 2.00 | 2.00 | grid-limited | 0.71 |
| 2 | 0.5 | 2.06 | 0.65 | 0.84 | 0.57 |
| 2 | 1 | 2.77 | 0.76 | 0.52 | 0.44 |
| 2 | 1.5 | 3.80 | 1.04 | 0.38 | 0.35 |
| 2 | 2 | 4.91 | 1.34 | 0.29 | 0.28 |
| 4 | 0 | 4.00 | 4.00 | grid-limited | 0.71 |
| 4 | 0.5 | 4.00 | 2.36 | 0.80 | 0.57 |
| 4 | 1 | 4.11 | 1.31 | 0.42 | 0.38 |
| 4 | 1.5 | 4.68 | 1.31 | 0.32 | 0.30 |
| 4 | 2 | 5.54 | 1.52 | 0.26 | 0.25 |
| 6 | 0 | 6.00 | 6.00 | grid-limited | 0.71 |
| 6 | 0.5 | 6.00 | 4.36 | 0.80 | 0.58 |
| 6 | 1 | 6.01 | 2.76 | 0.40 | 0.36 |
| 6 | 1.5 | 6.17 | 1.96 | 0.28 | 0.27 |
| 6 | 2 | 6.67 | 1.90 | 0.23 | 0.22 |

**Limits of the separation, stated before the evaluation.** This parameterisation reduces the
confounding of width and edge sharpness in the earlier vessel sweep. It does not remove it everywhere
on a 64 x 64 grid. Width and edge are cleanly distinguishable only in part of the grid, approximately
w >= 4 with s <= 1. The thin profiles, including the one-pixel regime in which the failure was
observed, are not an orthogonal width and edge experiment, and no result of this study is to be
described as one.

1. *Thin cores do not separate the two parameters.* For w below about 2 s the profile is close to a
   Gaussian of variance s^2 + w^2/12 and has no flat part. The profiles (1, 1), (1, 1.5) and (1, 2)
   are within 5 % of (0, 1), (0, 1.5) and (0, 2) in FWHM and are treated as near-replicates of
   them, not as independent cells. Width and edge are separate properties of the image only for
   w >= 4 with s <= 1 (FWHM within 0.15 pixels of w), and approximately for w = 6 up to s = 2.
2. *At fixed s the largest slope still depends on w.* It is 0.607/s for a Gaussian line (w = 0) and
   0.399/s for an isolated blurred edge (w >= 4), because a thin profile is scaled up to peak 1.
   H1 holds s fixed, as the edge parameter of the design, so its primary contrast compares a slope
   of 0.61 at (0, 1) with 0.40 at (6, 1). This remaining difference in steepness is part of the
   confounding that the design does not remove. How much the advantage at w = 6 changes between
   s = 0.5 (slope 0.80) and s = 1 (slope 0.40) is reported with H1 as a descriptive check of whether
   steepness alone could produce an effect of that size; it does not enter the verdict.
3. *The grid cannot show an edge much sharper than one pixel.* The measured gradient is 0.71 for a
   sharp edge and 0.57 to 0.61 for s = 0.5, although the continuous slopes differ by more. The
   profile (0, 0.5) is narrower than a pixel (FWHM 1.18): its sampled values depend on where the
   centreline falls between grid points.
4. *A sharp bar of width 1 or 2 pixels at an arbitrary angle is a staircase of pixels,* and all bars
   have rounded ends, unlike the square ends of the training rectangles.
5. *The curvature tested is that of the vessel family:* a bend of at most about 5 pixels, no
   branching and no closed curves. A curved line is somewhat longer than its straight partner.
6. *Some lines are short.* 9 of the 50 images contain a line whose chord is shorter than 6 pixels
   (minimum 3.1), which at larger s looks like a blob. They are kept, because the vessel family
   contains them; every hypothesis is also reported on the 41 images without them (section 8).

If these limits make phase 1 inconclusive, that is the result. No further experiment is designed to
obtain a cleaner answer (section 10).

## 5. Test set and evaluation

- **Images:** 50 images per cell, phantom seeds 600000 to 600049, used by no other split or sweep
  in the repository. The same 50 seeds are used in every cell, so cells are paired by centreline and
  amplitude, and the same phantoms are evaluated with both sensor counts.
- **Regime:** 14 dB SNR (relative noise 0.2 of the recording's RMS, the project's noise model), with
  16 and with 64 sensors. Noise seed 979000. The standard-normal draw depends only on the image
  index and the sensor count, so one image carries the same draw in every cell. Noiseless and
  low-noise behaviour is not part of this study.
- **Methods:** Tikhonov with the weight 0.01 at both sensor counts (selected earlier on the mixed
  training split at this noise level, `report/mixed_phantom_results.json`; not re-tuned on lines);
  the five control networks `mixed160_seed0..4`; the five intervention networks
  `edgefill160_seed0..4`. The checkpoints are verified against `report/checkpoint_manifest.json`
  before and after the evaluation.
- **Size:** 48 cells x 50 images x 2 sensor counts = 4800 forward simulations and 52800
  reconstructions.

## 6. Metrics

For every cell and sensor count, with the factors recorded in every row:

- PSNR and SSIM of Tikhonov, the control group and the intervention group;
- control minus Tikhonov, intervention minus Tikhonov, intervention minus control (PSNR);
- each of these for every network separately, the mean over the five networks and their standard
  deviation;
- 95 % bootstrap intervals over the 50 images (2000 resamples, seed 0, paired where two quantities
  share images) for the group means. Variation between networks is reported beside the intervals and
  is not folded into them.

Hypotheses are stated on the **advantage** A = PSNR of the mean of a group's five networks minus
PSNR of Tikhonov, because the failure to be explained is defined that way. Every contrast is also
split into the change of the networks' PSNR and the change of Tikhonov's PSNR, and the report must
say which of the two moved. Absolute PSNR is not compared across widths as a measure of difficulty:
a thin structure fills few pixels and scores high with any method.

A contrast **meets a threshold** when its estimate is at least the threshold and the lower end of its
95 % interval is above zero. If fewer than four of the five networks meet it individually, this is
stated beside the verdict and does not change it.

## 7. Hypotheses and thresholds

Thresholds, fixed here:

- **3 dB, a material effect.** The criterion of the edge-sharpness intervention; three quarters of
  the 3.96 dB deficit to be explained; several times the spread between networks (0.06 to 0.85 dB in
  the two earlier sweeps, measured when the intervention was audited) and the image intervals of the
  vessel sweep (half-width at most 0.35 dB with 20 images).
- **1 dB, a negligible effect and the geometry tolerance.** The refutation line of the intervention;
  about the largest spread between networks seen so far; a quarter of the deficit.
- **2 dB, the flag for an interaction.**

All hypotheses concern the intervention networks at 14 dB SNR unless stated, and each is evaluated
at 16 and at 64 sensors separately. The classification (section 9) is drawn at 64 sensors, where the
failure was observed; 16 sensors is reported with the same rules. Profiles are written (w, s).

**P0, precondition: the failure is present.** On curved lines of profile (0, 1) the upper end of the
95 % interval of A is below zero. If P0 fails with 64 sensors, the phenomenon is not present in this
test set, no classification is drawn, and the study stops.

**H1, thinness.** The purpose is to estimate the effect of structural width while the edge width is
held fixed. Every H1 contrast compares two profiles with the same s.

- Primary, on straight lines: E1 = A(6, 1) - A(0, 1).

H1 is *supported* if E1 meets 3 dB, *rejected* if E1 is below 1 dB, and *indeterminate* otherwise.
The verdict uses this contrast alone and does not involve any change of s.

- Secondary, on straight lines, each with the same three-way reading and reported beside the
  primary verdict without changing it: A(6, 0.5) - A(1, 0.5) and A(6, 1.5) - A(0, 1.5).

The first secondary contrast uses w = 1 as its thin profile because (0, 0.5) is narrower than a
pixel. The second repeats the comparison at the 1.5 pixel point of the earlier sweep, where the
networks were already ahead of Tikhonov, and shows whether the width effect fades as the line
thickens. All three contrasts are also computed on curved lines. The advantage at s = 1 over
w = 0, 1, 2, 4, 6 is reported as a curve without a verdict.

**H2, edge profile (positive control).** At a thick core the earlier edge-sharpness result should
reappear. On straight bars with w = 6, let the valley of a group be its lower advantage at s = 0.5
and s = 1. H2 is *supported* if all three hold, and *not reproduced* otherwise:

- the control networks dip: A_control(6, 2) minus the control valley is at least 3 dB;
- the intervention removes most of it: intervention valley minus control valley is at least 3 dB;
- the paired gain of intervention over control at (6, 1) has a 95 % interval above zero.

If H2 is not reproduced, the edge axis of this test set does not behave like the disc sweep, and
every statement about edge profile is marked as not validated.

**H3, geometry.** The signed contrast is G(w, s) = A(straight) - A(curved) at the same profile,
paired by image, so that a positive G means that curved lines are worse. It is computed on the thin
soft profiles (0, 1), (0, 1.5), (1, 0.5) and (1, 1), with (0, 1) as the primary one. The geometry
effect is

- *negligible* if the 95 % interval of G(0, 1) lies inside (-1, +1) dB and |G| is below 1 dB at all
  four profiles;
- *curved materially worse* if G(0, 1) is at least +3 dB with an interval above zero;
- *curved materially better* if G(0, 1) is at most -3 dB with an interval below zero. This is an
  unexpected effect in the opposite direction. It is reported as such and is not evidence that
  curvature causes the vessel failure;
- *modest or indeterminate* otherwise.

The tolerance of 1 dB is kept: it is the refutation line of the intervention, about the largest
spread between networks seen so far, and a quarter of the deficit to be explained.

**H4, sharp thin bar.** If thinness itself is the problem, a bar with the familiar sharp edge also
degrades once it is thinner than anything in training. The training rectangles and ellipses are 3.8
to 10.2 pixels wide and the discs 6.4 to 14.1, so bars of width 4 and 6 are of trained width and
bars of width 1 and 2 are not. On straight bars with s = 0:

- E4 = mean of A(4, 0) and A(6, 0), minus A(1, 0).

H4 is *supported* if E4 meets 3 dB, *rejected* if it is below 1 dB, *indeterminate* otherwise. The
verdict uses the intervention networks; the control networks, which saw twice as many sharp images,
are reported beside it, as is the bar of width 2.

## 8. Interactions and robustness

Two interactions are computed as differences of differences of A, with paired bootstrap intervals:

- width x edge: [A(6, 1) - A(1, 1)] - [A(6, 0) - A(1, 0)] on straight lines, whether the cost of
  thinness depends on the edge;
- width x geometry: [A(6, 1) - A(0, 1)] on curved lines minus the same on straight lines, whether
  the cost of thinness is larger on curved lines (positive) or on straight ones.

An interaction is *flagged* when its magnitude is at least 2 dB and its interval excludes zero. No
other test is run on the 48 cells: they are reported in full as width x edge tables and maps for
each geometry and sensor count (Tikhonov PSNR, intervention PSNR and their difference), not only as
one-dimensional curves, and only the contrasts above carry verdicts.

Every hypothesis is recomputed on the 41 images in which each line has a chord of at least 6 pixels.
This is a robustness report. A verdict that changes on this subset is stated as fragile.

## 9. Order of evidence and summary classification

The report of the results presents the evidence in this order, and conclusions are weighted in the
same order:

1. the primary pre-registered contrasts: P0 and H1 to H4, each with its interval, its values for
   the individual networks, and its split into network and Tikhonov changes;
2. the width x edge and width x geometry interactions;
3. the complete descriptive response surface over all 48 cells;
4. a summary classification.

The classification is a summary of items 1 and 2. It never overrides a primary contrast or a strong
interaction that contradicts it: where they disagree, the contrasts and interactions stand and the
classification is reported as not applicable. It is assigned by the runner for the 64-sensor
results, after P0, by the first rule that applies.

- **Thinness-dominated:** H1 supported on straight and on curved lines, geometry negligible, H4
  supported, and no interaction flagged.
- **Geometry-dominated:** curved materially worse, H1 not supported on straight lines (width alone
  does not explain the deficit), and the width x edge interaction not flagged. A width x geometry
  interaction is expected in this pattern and does not count against it.
- **Edge-profile-dominated:** H1 not supported on straight lines, curved not materially worse, the
  advantage of the intervention networks on straight bars with w = 6 varies by at least 3 dB over
  s in {0.5, 1, 1.5, 2} (the deficit follows the edge profile at a fixed structural width), and the
  width x edge interaction not flagged.
- **Dominance not established:** one of the three patterns above holds except that a flagged
  interaction complicates it, or exactly one of the three factors below is present. The factor is
  named.
- **Mixed:** at least two of these factors are present: H1 supported on straight lines; curved
  materially worse; the 3 dB edge range just defined. The factors are named.
- **Inconclusive:** none is present.

A single winner is not forced: "mixed", "dominance not established" and "inconclusive" are acceptable
outcomes. A geometry effect in the opposite direction (curved materially better) never counts as a
geometry factor; it is noted beside the classification. Flagged interactions and a failed H2 are
also named beside it.

## 10. Stopping rule

Phase 1 consists only of the evaluation of the existing checkpoints on the test set defined here,
run once.

After phase 1, work stops. The results are reviewed before anything else is done. At most one
additional training intervention, addressing the dominant factor that phase 1 identifies, may then
be proposed; it is not started automatically and needs its own pre-registration. If phase 1 is
inconclusive, including because of the discretisation limits of section 4, no further experiment is
designed in order to obtain a clean story.

## 11. Limitations

- Phase 1 varies the test images, not the training set. It can show which property of an image the
  existing networks fail on; it cannot show which change to training would repair it.
- Everything is synthetic, on a 64 x 64 grid, with data generated by the same operator that Tikhonov
  and the networks invert, at one noise level.
- Tikhonov's weight was tuned on the mixed training shapes, not on lines.
- The networks were still improving at their last epochs, and there are five per group.
- The limits of the width and edge separation are those of section 4: the design reduces the
  earlier confounding and does not eliminate it, least of all in the one-pixel regime.

## 12. Out of scope

The learned data-consistency weights and the gap to Tikhonov at low noise with 64 sensors are a
separate question, recorded in `report/future_question_data_consistency_weights.md`. It is not
investigated here and no result of this study is to be read as evidence about it.
