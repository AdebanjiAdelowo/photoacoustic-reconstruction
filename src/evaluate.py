"""Evaluation: PSNR/SSIM against ground truth, swept over sensor count / sparsity.

Delegates to scikit-image's metrics rather than reimplementing PSNR/SSIM from scratch — evaluation
code must be trustworthy for the comparison to mean anything (ARCHITECTURE.md Section 7). Not
implemented — no metric has ever been computed in this repository.
"""


def psnr(reconstruction, ground_truth):
    """Peak signal-to-noise ratio via skimage.metrics.peak_signal_noise_ratio.

    Planned. Not implemented.
    """
    raise NotImplementedError("Planned — see ARCHITECTURE.md Section 7")


def ssim(reconstruction, ground_truth):
    """Structural similarity index via skimage.metrics.structural_similarity.

    Planned. Not implemented.
    """
    raise NotImplementedError("Planned — see ARCHITECTURE.md Section 7")


def sparsity_sweep(methods: dict, phantom, sensor_counts: list):
    """Run all methods across a sweep of sensor counts and collect metrics.

    Planned. Not implemented — will produce the PSNR/SSIM-vs-sparsity curve described in
    ARCHITECTURE.md Section 8 once the pipeline exists. Note: the MVP (see IMPLEMENTATION_PLAN.md)
    only requires two sparsity levels, not a full sweep — this function's full sweep use is
    strong-version scope (E2 in ARCHITECTURE.md Section 6), not MVP scope.
    """
    raise NotImplementedError("Planned — see ARCHITECTURE.md Sections 6-8")
