from typing import List

import torch
import torch.nn as nn
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


class AggregatedResidualBlock(nn.Module):
    """Basic build block of ResNext."""

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        stride: int,
        cardinality: int,
        bottleneck_width: int,
    ):
        """ctor.

        Args:
            in_channel: input channel.
            out_channel: output channel.
            stride: conv stride.
            cardinality: groups in the group convolution. 4 in paper.
            bottleneck_width: number of layers per group.
        """

        super().__init__()
        assert stride == 1 or stride == 2
        # One design decision is when there is a stride, the output channel is not equal to input channel.
        if stride == 2:
            assert out_channel != in_channel
        if stride == 1:
            assert out_channel == in_channel
        group_channel = int(cardinality * bottleneck_width)
        assert group_channel * 2 == out_channel
        self.residual = nn.Sequential(
            _Conv2d(
                in_channel, group_channel, activation=True, kernel_size=1, stride=1
            ),
            _Conv2d(
                group_channel,
                group_channel,
                activation=True,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=cardinality,
            ),
            _Conv2d(
                group_channel, out_channel, activation=False, kernel_size=1, stride=1
            ),
        )
        self.shortcut = nn.Identity()
        if stride == 2:
            self.shortcut = _Conv2d(
                in_channel, out_channel, activation=False, kernel_size=1, stride=stride
            )

    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.relu(self.shortcut(x) + self.residual(x), inplace=True)


def build_stage(
    in_channel: int,
    out_channel: int,
    layer: int,
    cardinality: int,
    bottleneck_width: int,
) -> nn.Sequential:
    blocks = [
        AggregatedResidualBlock(
            in_channel=in_channel,
            out_channel=out_channel,
            stride=2,
            cardinality=cardinality,
            bottleneck_width=bottleneck_width,
        ),
    ]
    for i in range(1, layer):
        blocks.append(
            AggregatedResidualBlock(
                in_channel=out_channel,
                out_channel=out_channel,
                stride=1,
                cardinality=cardinality,
                bottleneck_width=bottleneck_width,
            )
        )
    return nn.Sequential(*blocks)


class ResNextNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        cardinality: int,
        bottleneck_width: int,
        base_width: int,
        layer_per_stage: List[int],
        channel_per_stage: List[int],
        dropout: float,
    ) -> None:
        """ctor.

        Args:
            num_classes: Number of classes.
            cardinality: groups in the group convolution. 4 in paper.
            bottleneck_width: number of layers per group.
            base_width: Output channel of the first conv. Each stage will double its output.
            layer_per_stage: Number of blocks in each stage.
            channel_per_stage: Numner of output channel in each stage
            dropout: dropout ratio.
        """
        super().__init__()
        assert len(layer_per_stage) == len(channel_per_stage)
        stage_channels = [base_width, *channel_per_stage]

        layers = [
            _Conv2d(
                3,
                stage_channels[0],
                activation=True,
                kernel_size=7,
                padding=3,
                stride=2,
            )
        ]
        for i in range(len(layer_per_stage)):
            layers.append(
                build_stage(
                    in_channel=stage_channels[i],
                    out_channel=stage_channels[i + 1],
                    layer=layer_per_stage[i],
                    cardinality=cardinality,
                    bottleneck_width=bottleneck_width * 2**i,
                )
            )
        self.feature = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(stage_channels[-1], num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.feature(x)
        return self.classifier(x)


def ResNextNet50_32_4(num_classes: int, dropout: float):
    return ResNextNet(
        num_classes=num_classes,
        cardinality=32,
        bottleneck_width=4,
        base_width=64,
        layer_per_stage=[3, 4, 6, 3],
        channel_per_stage=[256, 512, 1024, 2048],
        dropout=dropout,
    )
