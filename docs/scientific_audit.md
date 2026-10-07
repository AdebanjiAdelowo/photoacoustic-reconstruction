# Scientific Audit — Sparse-View Photoacoustic Reconstruction

## Verdict
The repository studies a meaningful linear inverse problem: recover initial pressure from sparse circular acoustic measurements. The model-based Tikhonov baseline, learned image prior and unrolled model-based network are all legitimate approaches, but they answer different prior/regularisation questions.

The strongest limitation is distribution shift: learned gains on the training phantom family do not consistently transfer to thin vessel-like structures. This is a scientifically valuable negative result, not a bug to hide.

## Forward model
For the fixed homogeneous medium used here, acoustic propagation from initial pressure to sensor recordings is linear. The repository explicitly assembles the discrete forward matrix for the Tikhonov experiments, making the discrete adjoint exactly its matrix transpose.

This is especially important because autodifferentiating through the selected j-Wave path did not satisfy the measured adjoint identity closely enough; refusing to use that gradient is the correct decision.

## Tikhonov
The direct solution of
[
\min_p \|Ap-y\|^2+\lambda\|p\|^2
]
is a well-defined convex baseline for (lambda>0). Selecting regularisation on training/validation data rather than the test split is essential and is protected by protocol tests.

A noise-matched/oracle-style Tikhonov result should be labelled as such when it receives information not supplied to a learned model; it is then an informative upper/control baseline, not necessarily an operationally fair competitor.

## Learned reconstruction
A U-Net refinement of time reversal learns a distribution-specific image prior. High in-distribution PSNR does not establish correctness of the inverse physics or robustness to new morphology.

The vessel/thin-structure experiments demonstrate this directly.

## Unrolled Tikhonov network
The unrolled model alternates learned residual correction with exact linear data-consistency solves. This is methodologically stronger than a purely image-domain network when the forward model is trusted, but the learned denoiser remains a prior and can still fail under prior shift.

## Model mismatch
Training/evaluating with a nominal homogeneous sound speed while generating data with perturbed sound speed is an appropriate way to test operator mismatch. A method that enforces the wrong (A) more strongly can lose its advantage; that is not paradoxical.

## Prior-shift interpretation
The experiments support the statement that thin structures are a major failure mode for the learned prior in the tested setting. They do **not** establish a universal causal law about curvature, edge sharpness or vessel morphology. Where hypotheses were not reproduced, they should remain rejected/unresolved.

## Classification
- Wrong inverse problem: **no**.
- Tikhonov: **appropriate model-based baseline**.
- U-Net: **appropriate learned-prior experiment, distribution dependent**.
- Unrolled method: **appropriate model-based learned reconstruction**.
- Thin-structure failure: **genuine generalisation limitation**.
- Physical/clinical validation: **not established by synthetic phantom experiments**.

No new implementation defect was identified in this audit pass.
