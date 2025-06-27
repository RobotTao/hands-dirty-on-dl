"""Modified squeezenet implementation from
https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py"""

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class _Conv2d(nn.Module):
    """Basic Inception 2D convolution block."""

    def __init__(
        self, in_channel: int, out_channel: int, add_norm: bool, **kwargs
    ) -> None:
        """Init.

        Args:
            in_channel: input channel to the Conv2d class.
            out_channel: output channel to the Conv2d class.
            add_norm: Whether to use normalization layer or not.
            kwargs: Other parameters to the Conv2d class.
        """
        super().__init__()
        self.conv2d = nn.Conv2d(in_channel, out_channel, **kwargs)
        self.norm = None
        if add_norm:
            self.norm = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X: Tensor) -> Tensor:
        X = self.conv2d(X)
        if self.norm:
            X = self.norm(X)
        X = self.relu(X)
        return X


class Fire(nn.Module):
    """SqueezeNet Fire Module."""

    def __init__(
        self,
        inplanes: int,
        squeeze_planes: int,
        expand1x1_planes: int,
        expand3x3_planes: int,
        add_norm: bool,
    ) -> None:
        super().__init__()
        self.squeeze = _Conv2d(inplanes, squeeze_planes, add_norm, kernel_size=1)
        self.expand1x1 = _Conv2d(
            squeeze_planes, expand1x1_planes, add_norm, kernel_size=1
        )
        self.expand3x3 = _Conv2d(
            squeeze_planes, expand3x3_planes, add_norm, kernel_size=3, padding=1
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: shape (B,C,H,W)."""
        x = self.squeeze(x)
        return torch.cat([self.expand1x1(x), self.expand3x3(x)], dim=1)


def residual_connection(x: Tensor, fn: nn.Module) -> Tensor:
    """Helper function to create residual connection."""
    # return nn.functional.relu(x + fn(x))
    return x + fn(x)


class SqueezeNet(nn.Module):
    """SqueezeNet with residual connection."""

    def __init__(self, num_classes: int, add_norm: bool, dropout: float) -> None:
        super().__init__()
        self.conv1 = _Conv2d(3, 64, add_norm, kernel_size=3, stride=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.fire2 = Fire(64, 16, 64, 64, add_norm)
        self.fire3 = Fire(128, 16, 64, 64, add_norm)
        self.fire4 = Fire(128, 32, 128, 128, add_norm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.fire5 = Fire(256, 32, 128, 128, add_norm)
        self.fire6 = Fire(256, 48, 192, 192, add_norm)
        self.fire7 = Fire(384, 48, 192, 192, add_norm)
        self.fire8 = Fire(384, 64, 256, 256, add_norm)

        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.fire9 = Fire(512, 64, 256, 256, add_norm)

        # Final convolution is initialized differently from the rest
        final_conv = _Conv2d(512, num_classes, False, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            final_conv,
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.apply(self._init_weight)

    def _init_weight(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.maxpool1(self.conv1(x))
        x = self.fire2(x)
        x = residual_connection(x, self.fire3)
        x = self.fire4(x)
        x = self.maxpool2(x)
        x = residual_connection(x, self.fire5)
        x = self.fire6(x)
        x = residual_connection(x, self.fire7)
        x = self.fire8(x)
        x = self.maxpool3(x)
        x = residual_connection(x, self.fire9)
        x = self.classifier(x)
        return torch.flatten(x, 1)
