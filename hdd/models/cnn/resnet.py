"""Simple Resnet Implementation."""

from dataclasses import dataclass
from typing import Tuple, Union

import torch
import torch.nn as nn

from hdd.models.cnn.squeeze_excitation import SqueezeExcitation


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

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.conv2d(X)
        X = self.norm(X)
        if self.relu:
            X = self.relu(X)
        return X


class BasicBlock(nn.Module):
    """Basic block without bottleneck structure."""

    def __init__(self, in_channel: int, out_channel: int, downsample: bool) -> None:
        """Init.

        Args:
            in_channel: Input channel number.
            out_channel: Output channel number.
            downsample: Whether to downsample at this block.
        """
        super().__init__()
        stride = 2 if downsample else 1
        # fmt: off
        self.resbranch = nn.Sequential(
            _Conv2d(in_channel, out_channel, activation=True, kernel_size=3, stride=stride, padding=1),
            _Conv2d(out_channel, out_channel, activation=False, kernel_size=3, padding=1)
        )
        # 在indentity路径上的运算
        self.identity_transform = nn.Identity()
        if downsample:
            self.identity_transform = _Conv2d(in_channel, out_channel, activation=False, kernel_size=1, stride=stride)
        elif in_channel != out_channel:
            self.identity_transform = _Conv2d(in_channel, out_channel, activation=False, kernel_size=1)
        # fmt: on

    def forward(self, X) -> torch.Tensor:
        identity = self.identity_transform(X)
        res = self.resbranch(X)
        return nn.functional.relu(identity + res, inplace=True)


class BottleneckBlock(nn.Module):
    """Basic block without bottleneck structure."""

    def __init__(self, in_channel: int, out_channel: int, downsample: bool) -> None:
        """Init.

        Args:
            in_channel: Input channel number.
            out_channel: Output channel number.
            downsample: Whether to downsample at this block.
        """
        super().__init__()
        # As shown in the paper, the
        bottleneck_channel = out_channel // 4
        stride = 2 if downsample else 1

        # fmt: off
        self.resbranch = nn.Sequential(
            _Conv2d(in_channel, bottleneck_channel, activation=True, kernel_size=1, padding=0),
            _Conv2d(bottleneck_channel, bottleneck_channel, activation=True, kernel_size=3, stride=stride, padding=1),
            _Conv2d(bottleneck_channel, out_channel, activation=False, kernel_size=1, padding=0)
        )
        # 在indentity路径上的运算
        self.identity_transform = nn.Identity()
        if downsample:
            self.identity_transform = _Conv2d(in_channel, out_channel, activation=False, kernel_size=1, stride=stride)
        elif in_channel != out_channel:
            self.identity_transform = _Conv2d(in_channel, out_channel, activation=False, kernel_size=1)
        # fmt: on

    def forward(self, X) -> torch.Tensor:
        identity = self.identity_transform(X)
        res = self.resbranch(X)
        return nn.functional.relu(identity + res, inplace=True)


class SEBottleneckBlock(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        downsample: bool,
        reduction_ratio: int = 16,
    ) -> None:
        """Init.

        Args:
            in_channel: Input channel number.
            out_channel: Output channel number.
            downsample: Whether to downsample at this block.
        """
        super().__init__()
        # As shown in the paper, the
        bottleneck_channel = out_channel // 4
        stride = 2 if downsample else 1

        # fmt: off
        self.resbranch = nn.Sequential(
            _Conv2d(in_channel, bottleneck_channel, activation=True, kernel_size=1, padding=0),
            _Conv2d(bottleneck_channel, bottleneck_channel, activation=True, kernel_size=3, stride=stride, padding=1),
            _Conv2d(bottleneck_channel, out_channel, activation=False, kernel_size=1, padding=0),
            # 这个SqueezeExcitation模块是唯一和BottleneckBlock不同的地方
            SqueezeExcitation(out_channel, reduction_ratio),
        )
        # 在indentity路径上的运算
        self.identity_transform = nn.Identity()
        if downsample:
            self.identity_transform = _Conv2d(in_channel, out_channel, activation=False, kernel_size=1, stride=stride)
        elif in_channel != out_channel:
            self.identity_transform = _Conv2d(in_channel, out_channel, activation=False, kernel_size=1)
        # fmt: on

    def forward(self, X) -> torch.Tensor:
        identity = self.identity_transform(X)
        res = self.resbranch(X)
        return nn.functional.relu(identity + res, inplace=True)


@dataclass
class LayerConfig:
    """Layer config."""

    block_type: type[Union[BasicBlock, BottleneckBlock, SEBottleneckBlock]]
    in_channel: int
    out_channel: int
    block_count: int


def build_layer(config: LayerConfig, downsample: bool):
    in_channel = config.in_channel
    out_channel = config.out_channel

    blocks = [config.block_type(in_channel, out_channel, downsample=downsample)]
    for i in range(1, config.block_count):
        blocks.append(config.block_type(out_channel, out_channel, downsample=False))
    return nn.Sequential(*blocks)


resnet18_config = (
    LayerConfig(BasicBlock, 64, 64, 2),
    LayerConfig(BasicBlock, 64, 128, 2),
    LayerConfig(BasicBlock, 128, 256, 2),
    LayerConfig(BasicBlock, 256, 512, 2),
)

resnet34_config = (
    LayerConfig(BasicBlock, 64, 64, 3),
    LayerConfig(BasicBlock, 64, 128, 4),
    LayerConfig(BasicBlock, 128, 256, 6),
    LayerConfig(BasicBlock, 256, 512, 3),
)

resnet50_config = (
    LayerConfig(BottleneckBlock, 64, 256, 3),
    LayerConfig(BottleneckBlock, 256, 512, 4),
    LayerConfig(BottleneckBlock, 512, 1024, 6),
    LayerConfig(BottleneckBlock, 1024, 2048, 3),
)

se_resnet50_config = (
    LayerConfig(SEBottleneckBlock, 64, 256, 3),
    LayerConfig(SEBottleneckBlock, 256, 512, 4),
    LayerConfig(SEBottleneckBlock, 512, 1024, 6),
    LayerConfig(SEBottleneckBlock, 1024, 2048, 3),
)

resnet101_config = (
    LayerConfig(BottleneckBlock, 64, 256, 3),
    LayerConfig(BottleneckBlock, 256, 512, 4),
    LayerConfig(BottleneckBlock, 512, 1024, 23),
    LayerConfig(BottleneckBlock, 1024, 2048, 3),
)

resnet150_config = (
    LayerConfig(BottleneckBlock, 64, 256, 3),
    LayerConfig(BottleneckBlock, 256, 512, 8),
    LayerConfig(BottleneckBlock, 512, 1024, 36),
    LayerConfig(BottleneckBlock, 1024, 2048, 3),
)


class Resnet(nn.Module):
    """ImageNet上的Resnet,输入为(3,224,224)."""

    def __init__(
        self,
        layer_configs: Tuple[LayerConfig, LayerConfig, LayerConfig, LayerConfig],
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.pre_layer = nn.Sequential(
            _Conv2d(3, 64, activation=True, kernel_size=7, padding=3, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        )
        self.layer1 = build_layer(layer_configs[0], downsample=False)
        self.layer2 = build_layer(layer_configs[1], downsample=True)
        self.layer3 = build_layer(layer_configs[2], downsample=True)
        self.layer4 = build_layer(layer_configs[3], downsample=True)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(layer_configs[-1].out_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, 0, 0.01, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
        # zero_init_residual
        for m in self.modules():
            if isinstance(m, BasicBlock) or isinstance(m, BottleneckBlock):
                nn.init.constant_(m.resbranch[-1].norm.weight, 0)
                nn.init.constant_(m.resbranch[-1].norm.bias, 0)
            elif isinstance(m, SEBottleneckBlock):
                nn.init.constant_(m.resbranch[2].norm.weight, 0)
                nn.init.constant_(m.resbranch[2].norm.bias, 0)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.pre_layer(X)
        X = self.layer1(X)
        X = self.layer2(X)
        X = self.layer3(X)
        X = self.layer4(X)
        X = self.classifier(X)
        return X


class ResnetSmall(nn.Module):
    """Small resnet to handle image of size (3,32,32) or (3,64,64)."""

    def __init__(
        self,
        layer_configs: Tuple[LayerConfig, LayerConfig, LayerConfig, LayerConfig],
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.pre_layer = _Conv2d(
            3, 64, activation=True, kernel_size=3, padding=1, stride=1
        )

        self.layer1 = build_layer(layer_configs[0], downsample=False)
        self.layer2 = build_layer(layer_configs[1], downsample=True)
        self.layer3 = build_layer(layer_configs[2], downsample=True)
        self.layer4 = build_layer(layer_configs[3], downsample=True)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(layer_configs[-1].out_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, 0, 0.01, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
        # zero_init_residual
        for m in self.modules():
            if isinstance(m, BasicBlock) or isinstance(m, BottleneckBlock):
                nn.init.constant_(m.resbranch[-1].norm.weight, 0)
                nn.init.constant_(m.resbranch[-1].norm.bias, 0)
            elif isinstance(m, SEBottleneckBlock):
                nn.init.constant_(m.resbranch[2].norm.weight, 0)
                nn.init.constant_(m.resbranch[2].norm.bias, 0)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.pre_layer(X)
        X = self.layer1(X)
        X = self.layer2(X)
        X = self.layer3(X)
        X = self.layer4(X)
        X = self.classifier(X)
        return X
