# Mobilenet V1

import torch
import torch.nn as nn
from torch import Tensor


class _Conv2d(nn.Sequential):
    """Basic Inception 2D convolution block."""

    def __init__(self, in_channel: int, out_channel: int, **kwargs) -> None:
        """Init.

        Args:
            in_channel: input channel to the Conv2d class.
            out_channel: output channel to the Conv2d class.
            kwargs: Other parameters to the Conv2d class.
        """

        layers = [
            nn.Conv2d(in_channel, out_channel, **kwargs),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        ]
        super().__init__(*layers)


class _DepthwiseConv(_Conv2d):
    def __init__(self, channel: int, stride: int) -> None:
        super().__init__(
            channel,
            channel,
            kernel_size=3,
            padding=1,
            stride=stride,
            groups=channel,
        )


class _Conv1x1(_Conv2d):
    def __init__(self, in_channel: int, out_channel: int) -> None:
        super().__init__(
            in_channel,
            out_channel,
            kernel_size=1,
        )


class MobileNetV1(nn.Module):
    def __init__(self, num_classes: int, width_multiplier: float = 1):
        super().__init__()

        def wm(channel: int) -> int:
            return int(channel * width_multiplier)

        self.feature = nn.Sequential(
            _Conv2d(3, wm(32), kernel_size=3, stride=2, padding=1),
            _DepthwiseConv(wm(32), stride=1),
            _Conv1x1(wm(32), wm(64)),
            _DepthwiseConv(wm(64), stride=2),
            _Conv1x1(wm(64), wm(128)),
            _DepthwiseConv(wm(128), stride=1),
            _Conv1x1(wm(128), wm(128)),
            _DepthwiseConv(wm(128), stride=2),
            _Conv1x1(wm(128), wm(256)),
            _DepthwiseConv(wm(256), stride=1),
            _Conv1x1(wm(256), wm(256)),
            _DepthwiseConv(wm(256), stride=2),
            _Conv1x1(wm(256), wm(512)),
            # Five Same Structure
            _DepthwiseConv(wm(512), stride=1),
            _Conv1x1(wm(512), wm(512)),
            _DepthwiseConv(wm(512), stride=1),
            _Conv1x1(wm(512), wm(512)),
            _DepthwiseConv(wm(512), stride=1),
            _Conv1x1(wm(512), wm(512)),
            _DepthwiseConv(wm(512), stride=1),
            _Conv1x1(wm(512), wm(512)),
            _DepthwiseConv(wm(512), stride=1),
            _Conv1x1(wm(512), wm(512)),
            # -----------
            _DepthwiseConv(wm(512), stride=2),
            _Conv1x1(wm(512), wm(1024)),
            _DepthwiseConv(wm(1024), stride=1),
            _Conv1x1(wm(1024), wm(1024)),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(wm(1024), num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.feature(x)
        return self.classifier(x)
