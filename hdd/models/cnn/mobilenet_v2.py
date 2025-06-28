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


class InvertedResidualBlock(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        expand_ratio: int,
        stride: int,
    ) -> None:
        super().__init__()
        self.use_residual = (in_channel == out_channel) and (stride == 1)
        internal_channel = int(in_channel * expand_ratio)
        self.conv1 = _Conv2d(in_channel, internal_channel, kernel_size=1)
        self.conv2 = _Conv2d(
            internal_channel,
            internal_channel,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=internal_channel,
        )
        self.conv3 = nn.Conv2d(internal_channel, out_channel, kernel_size=1)
        # 降维时没有ReLU激活函数
        self.norm = nn.BatchNorm2d(out_channel)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.norm(out)
        if self.use_residual:
            return x + out
        return out


class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int,
        width_multiplier: float = 1,
        dropout: float = 0,
    ):
        super().__init__()

        def wm(channel: int) -> int:
            return int(channel * width_multiplier)

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        input_channel = 3
        output_channel = wm(32)
        features = []
        features.append(
            _Conv2d(input_channel, output_channel, kernel_size=3, stride=2, padding=1)
        )

        for expansion_ratio, channel, layers, s in inverted_residual_setting:
            input_channel = output_channel
            output_channel = wm(channel)
            for i in range(layers):
                stride = s if i == 0 else 1
                features.append(
                    InvertedResidualBlock(
                        input_channel, output_channel, expansion_ratio, stride
                    )
                )
                input_channel = output_channel
        # building last several layers
        input_channel = output_channel
        output_channel = wm(1280)
        features.append(_Conv2d(input_channel, output_channel, kernel_size=1))
        self.feature = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.Dropout(dropout),
            nn.Linear(output_channel, num_classes),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.feature(x)
        return self.classifier(x)
