"""Learned reconstruction: a small U-Net refining the time-reversal estimate.

Deliberately small for the MVP (per IMPLEMENTATION_PLAN.md Stage 7 — "do not optimize architecture
complexity, the objective is to test the research question, not maximize network size"). Input:
the time-reversal reconstruction (1, H, W). Output: a refined reconstruction (1, H, W), same shape,
regression (not classification/segmentation) — the one real architectural change from the
`abdominal-ct-segmentation` design this reuses (that project's final layer is a sigmoid/Dice-loss
segmentation head; this one is a linear regression head with MSE loss).
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ReconstructionUNet(nn.Module):
    """Small 3-level U-Net, 1 -> 1 channel regression.

    Base feature count and depth are intentionally small (16 base features, 3 levels) — this is a
    proof-of-concept refinement network for a 64x64 MVP, not a production model.
    """

    def __init__(self, base_features: int = 16):
        super().__init__()
        f = base_features
        self.enc1 = ConvBlock(1, f)
        self.enc2 = ConvBlock(f, f * 2)
        self.enc3 = ConvBlock(f * 2, f * 4)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(f * 4, f * 8)

        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2)
        self.dec3 = ConvBlock(f * 8, f * 4)
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
        self.dec2 = ConvBlock(f * 4, f * 2)
        self.up1 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
        self.dec1 = ConvBlock(f * 2, f)

        self.out_conv = nn.Conv2d(f, 1, 1)  # linear output — regression, not classification

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)
