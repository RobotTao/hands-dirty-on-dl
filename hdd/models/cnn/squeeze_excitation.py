"""Squeeze-and-Excitation Module"""

import torch
import torch.nn as nn


class SqueezeExcitation(nn.Module):
    """Pluggable Squeeze-Excitation Module."""

    def __init__(self, channel: int, reduction_ratio: int) -> None:
        """Init.

        Args:
            channel: Number of channels.
            reduction_ratio: Reduction ratio for the internal FC layer.
        """
        super().__init__()
        fc_channel = channel // reduction_ratio
        if fc_channel * reduction_ratio != channel:
            raise ValueError("Channel is not divisible by reduction ratio.")
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            # 使用1x1卷积来模拟Linear
            nn.Conv2d(channel, fc_channel, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(fc_channel, channel, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X * self.se(X)
