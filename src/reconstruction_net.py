"""Learned reconstruction: a U-Net refining a classical (FBP/Tikhonov) estimate.

Architecture reuses design experience from ../../abdominal-ct-segmentation, adapted from a
segmentation target to a regression/reconstruction target (ARCHITECTURE.md Section 5). Not
implemented — no model has been defined, instantiated, or trained.
"""


class ReconstructionUNet:
    """Planned U-Net for refining a classical reconstruction into a learned one.

    Not implemented.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Planned — see ARCHITECTURE.md Section 5")


def train(model, train_loader, val_loader, epochs: int):
    """Training loop. Planned. Not implemented."""
    raise NotImplementedError("Planned — see ARCHITECTURE.md Section 6")
