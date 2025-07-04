from typing import List

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from rich import padding
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


class ShuffleBlockLayer(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, stride: int):
        super().__init__()
        if stride == 1:
            assert in_channel == out_channel
            branch_channel = int(in_channel // 2)
            right_in_channel = branch_channel
            left_in_channel = -1  # Not used
        elif stride == 2:
            branch_channel = int(out_channel // 2)
            right_in_channel = in_channel
            left_in_channel = in_channel
        else:
            raise Exception(f"Invalid stride value {stride}")

        self.stride = stride
        if stride == 1:
            self.left_branch = nn.Identity()
        else:
            self.left_branch = nn.Sequential(
                _Conv2d(
                    in_channel=left_in_channel,
                    out_channel=left_in_channel,
                    activation=False,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=left_in_channel,
                ),
                _Conv2d(
                    in_channel=left_in_channel,
                    out_channel=branch_channel,
                    activation=True,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
            )

        self.right_branch = nn.Sequential(
            _Conv2d(
                in_channel=right_in_channel,
                out_channel=branch_channel,
                activation=True,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            _Conv2d(
                in_channel=branch_channel,
                out_channel=branch_channel,
                activation=False,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=branch_channel,
            ),
            _Conv2d(
                in_channel=branch_channel,
                out_channel=branch_channel,
                activation=True,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            left, right = x.chunk(2, dim=1)
        else:
            left = x
            right = x
        out = torch.cat(
            [self.left_branch(left), self.right_branch(right)],
            dim=1,
        )
        return rearrange(out, "N (C G) H W -> N (G C) H W", G=2)


class ShuffleBlock(nn.Sequential):
    def __init__(self, in_channel: int, out_channel: int, L: int):
        """_summary_

        Args:
            in_channel: input channel count
            out_channel: output channel count
            L: Number of layers
        """
        layers = [ShuffleBlockLayer(in_channel, out_channel, stride=2)]
        for i in range(1, L):
            layers.append(ShuffleBlockLayer(out_channel, out_channel, stride=1))
        super().__init__(*layers)


class ShuffleNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int,
        stage_layers: List[int],
        stage_out_channels: List[int],
    ):
        super().__init__()
        input_channels = 3
        assert len(stage_out_channels) == 5
        self.stage1 = nn.Sequential(
            _Conv2d(
                input_channels,
                stage_out_channels[0],
                activation=True,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.MaxPool2d((2, 2)),
        )
        self.stage2 = ShuffleBlock(
            stage_out_channels[0], stage_out_channels[1], stage_layers[0]
        )
        self.stage3 = ShuffleBlock(
            stage_out_channels[1], stage_out_channels[2], stage_layers[1]
        )
        self.stage4 = ShuffleBlock(
            stage_out_channels[2], stage_out_channels[3], stage_layers[2]
        )
        self.stage5 = _Conv2d(
            stage_out_channels[3],
            stage_out_channels[4],
            activation=True,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(stage_out_channels[4], num_classes),
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        return self.classifier(x)
