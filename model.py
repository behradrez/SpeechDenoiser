from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DenoiserCNN(nn.Module):
    """Simple convolutional network that predicts clean log-magnitude spectra."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(1, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 32),
            ConvBlock(32, 16),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, noisy_log_mag: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noisy_log_mag: (batch, 1, freq_bins, time_frames)
        Returns:
            clean_log_mag: (batch, 1, freq_bins, time_frames)
        """
        return self.net(noisy_log_mag)
