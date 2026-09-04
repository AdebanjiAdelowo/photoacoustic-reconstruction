"""Evaluation: PSNR/SSIM against ground truth, swept over sensor count / sparsity.

See ARCHITECTURE.md Sections 6-7 for the experiment and evaluation plan. Not implemented — no
metric has ever been computed in this repository.
"""


def psnr(reconstruction, ground_truth):
    """Peak signal-to-noise ratio. Planned. Not implemented."""
    raise NotImplementedError("Planned — see ARCHITECTURE.md Section 7")


def ssim(reconstruction, ground_truth):
    """Structural similarity index. Planned. Not implemented."""
    raise NotImplementedError("Planned — see ARCHITECTURE.md Section 7")


def sparsity_sweep(methods: dict, phantom, sensor_counts: list):
    """Run all methods across a sweep of sensor counts and collect metrics.

    Planned. Not implemented — will produce the PSNR/SSIM-vs-sparsity curve described in
    ARCHITECTURE.md Section 8 once the pipeline exists.
    """
    raise NotImplementedError("Planned — see ARCHITECTURE.md Sections 6-8")
