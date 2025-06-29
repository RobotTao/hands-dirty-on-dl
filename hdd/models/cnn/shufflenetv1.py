from typing import List

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch import Tensor


class _Conv2d(nn.Module):
    """Basic Inception 2D convolution block."""

    def __init__(
        self, in_channel: int, out_channel: int, activation: bool = True, **kwargs
    ) -> None:
        """Init.

        Args:
            in_channel: input channel to the Conv2d class.
            out_channel: output channel to the Conv2d class.
            activation: Whether to use activation or not.
            kwargs: Other parameters to the Conv2d class.
        """
        super().__init__()
        self.conv2d = nn.Conv2d(in_channel, out_channel, **kwargs, bias=False)
        self.norm = nn.BatchNorm2d(out_channel)
        self.relu = None
        if activation:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, X: Tensor) -> Tensor:
        X = self.conv2d(X)
        X = self.norm(X)
        if self.relu:
            X = self.relu(X)
        return X


class ShuffleNetBlock(nn.Module):
    """Basic build block of ResNext."""

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        stride: int,
        group: int,
    ):
        """ctor.

        Args:
            in_channel: input channel.
            out_channel: output channel.
            stride: stride.
            group: number of groups
        """
        super().__init__()
        assert stride == 1 or stride == 2
        assert out_channel % 4 == 0
        self.stride = stride
        bottleneck_channel = out_channel // 4
        out_channel = out_channel // stride
        self.residual = nn.Sequential(
            _Conv2d(
                in_channel,
                bottleneck_channel,
                activation=True,
                kernel_size=1,
                stride=1,
                groups=group,
            ),
            Rearrange("B (n g) H W -> B (g n) H W", g=group),
            _Conv2d(
                bottleneck_channel,
                bottleneck_channel,
                activation=False,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=bottleneck_channel,
            ),
            _Conv2d(
                bottleneck_channel,
                out_channel,
                activation=False,
                kernel_size=1,
                stride=1,
                groups=group,
            ),
        )
        self.shortcut = nn.Identity()
        if stride == 2:
            self.shortcut = _Conv2d(
                in_channel,
                out_channel,
                activation=False,
                kernel_size=3,
                stride=stride,
                padding=1,
            )

    def forward(self, x: Tensor) -> Tensor:
        shortcut = self.shortcut(x)
        residual = self.residual(x)
        if self.stride == 1:
            return nn.functional.relu(shortcut + residual, inplace=True)
        return nn.functional.relu(torch.cat([shortcut, residual], dim=1), inplace=True)


def build_stage(
    in_channel: int, out_channel: int, layer: int, group: int
) -> nn.Sequential:
    blocks = [
        ShuffleNetBlock(
            in_channel=in_channel,
            out_channel=out_channel,
            stride=2,
            group=group,
        ),
    ]
    for i in range(1, layer):
        blocks.append(
            ShuffleNetBlock(
                in_channel=out_channel,
                out_channel=out_channel,
                stride=1,
                group=group,
            )
        )
    return nn.Sequential(*blocks)


class ShuffleNetV1(nn.Module):
    def __init__(
        self,
        num_classes: int,
        group: int,
        width_multiplier: float,
        dropout: float,
    ) -> None:
        """ctor.

        Args:
            num_classes: Number of classes.
            group: group.
            width_multiplier: Width multiplier
            dropout: dropout ratio.
        """
        super().__init__()
        configs = {
            1: [24, 144, 288, 576],
            2: [24, 200, 400, 800],
            3: [24, 240, 480, 960],
            4: [24, 272, 544, 1088],
            8: [24, 384, 768, 1536],
        }
        out_channels = configs[group]
        for i in range(1, 4):
            out_channels[i] = int(out_channels[i] * width_multiplier)
        layer_per_stage = [4, 8, 4]
        layers = nn.Sequential(
            _Conv2d(
                3,
                out_channels[0],
                activation=True,
                kernel_size=3,
                padding=1,
                stride=2,
            ),
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
        )
        for i in range(len(out_channels) - 1):
            layers.append(
                build_stage(
                    in_channel=out_channels[i],
                    out_channel=out_channels[i + 1],
                    layer=layer_per_stage[i],
                    group=group,
                )
            )
        self.feature = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(out_channels[-1], num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.feature(x)
        return self.classifier(x)
