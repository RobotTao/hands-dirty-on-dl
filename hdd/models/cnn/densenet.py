"""Densenet implementation based on https://github.com/liuzhuang13/DenseNet"""

from functools import partial
from typing import Generic, Tuple, Union

import torch
import torch.nn as nn


class DNLayerStandard(nn.Module):
    """Standard Densenet layer.."""

    def __init__(self, channel, growth_rate: int, dropout: float) -> None:
        """Init.

        Args:
            channel: input channel number.
            growth_rate: output channel number.
            dropout: Dropout ratio.
        """
        super().__init__()
        self.resbranch = nn.Sequential(
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channel,
                growth_rate,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.Dropout(dropout),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        res = self.resbranch(X)
        return torch.concat([res, X], dim=1)


class DNLayerBottleneck(nn.Module):
    """Standard Densenet layer with Bottleneck layer."""

    def __init__(self, channel, growth_rate: int, dropout: float) -> None:
        """Init.

        Args:
            channel: input channel number.
            growth_rate: output channel number.
            dropout: Dropout ratio.
        """
        super().__init__()
        bottleneck_channel = 4 * growth_rate
        self.resbranch = nn.Sequential(
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            # Start of bottleneck
            nn.Conv2d(
                channel,
                bottleneck_channel,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.Dropout(dropout),
            nn.BatchNorm2d(bottleneck_channel),
            nn.ReLU(inplace=True),
            #  End of bottleneck
            nn.Conv2d(
                bottleneck_channel,
                growth_rate,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.Dropout(dropout),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        res = self.resbranch(X)
        return torch.concat([res, X], dim=1)


class TransitionLayer(nn.Sequential):
    def __init__(self, in_channel) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_channel, in_channel, kernel_size=1, stride=1, bias=False
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.out_channel = in_channel


class TransitionLayerCompression(nn.Sequential):
    def __init__(self, in_channel) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_channel, in_channel // 2, kernel_size=1, stride=1, bias=False
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.out_channel = in_channel // 2


class DNBlock(nn.Module):
    """Densenet block with Bottleneck and Compression."""

    def __init__(
        self,
        in_channel: int,
        growth_rate: int,
        dropout: float,
        L: int,
        DNLayer: type[Union[DNLayerStandard, DNLayerBottleneck]],
    ):
        """Init.

        Args:
            in_channel: Input channel number.
            growth_rate: Growth rate.
            dropout: dropout ratio.
            L: Number of layers in the block.
            DNLayer: densenet layer module.
        """
        super().__init__()
        self.dense_layers = nn.ModuleList()
        for i in range(L):
            self.dense_layers.append(DNLayer(in_channel, growth_rate, dropout))
            in_channel += growth_rate
        self.out_channel = in_channel

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        for layer in self.dense_layers:
            X = layer(X)
        return X


DNBlockStandard = partial(DNBlock, DNLayer=DNLayerStandard)
DNBlockBC = partial(DNBlock, DNLayer=DNLayerBottleneck)


class _DenseNet(nn.Module):
    """Generic Densenet for imagenet."""

    def __init__(
        self,
        num_classes: int,
        growth_rate: int,
        dropout: float,
        is_small: bool = False,
        init_feature: int = 64,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        DenseBlock=DNBlockBC,
        TransLayer: type[
            Union[TransitionLayer, TransitionLayerCompression]
        ] = TransitionLayerCompression,
    ) -> None:
        """_summary_

        Args:
            num_classes: number of classes.
            growth_rate: growth rate.
            dropout: dropout ratio.
            is_small: Whether it is for small image like cifar10. Defaults to False.
            init_feature: init layer feature count. Defaults to 64.
            block_config: block layer config. Defaults to (6, 12, 24, 16).
            DenseBlock: Dense block type. Defaults to DNBlockBC.
            TransLayer: Transition layer type. Defaults to TransitionLayerCompression.
        """
        super().__init__()
        if is_small:
            self.features = nn.Sequential(
                nn.Conv2d(
                    3, init_feature, kernel_size=3, padding=1, stride=1, bias=False
                ),
            )
        else:

            self.features = nn.Sequential(
                nn.Conv2d(
                    3, init_feature, kernel_size=7, padding=3, stride=2, bias=False
                ),
                nn.BatchNorm2d(init_feature),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        in_channel = init_feature
        for i, L in enumerate(block_config):
            last_module = DenseBlock(in_channel, growth_rate, dropout, L)
            self.features.add_module(f"Dense_{i}", last_module)
            # No transition layer at the last block.
            if i + 1 < len(block_config):
                last_module = TransLayer(last_module.out_channel)
                self.features.add_module(f"Transition_{i}", last_module)
            in_channel = last_module.out_channel
        self.map_to_feature = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(in_channel, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, 0, 0.01, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.features(X)
        X = self.map_to_feature(X)
        return self.classifier(X)


DenseNetImageNet = partial(
    _DenseNet, is_small=False, DenseBlock=DNBlockStandard, TransLayer=TransitionLayer
)
DenseNet121 = partial(DenseNetImageNet, block_config=(6, 12, 24, 16))
DenseNet169 = partial(DenseNetImageNet, block_config=(6, 12, 32, 32))
DenseNet201 = partial(DenseNetImageNet, block_config=(6, 12, 48, 32))
DenseNet264 = partial(DenseNetImageNet, block_config=(6, 12, 64, 48))

DenseNetImageNetBC = partial(
    _DenseNet,
    is_small=False,
    DenseBlock=DNBlockBC,
    TransLayer=TransitionLayerCompression,
)
DenseNetBC121 = partial(DenseNetImageNetBC, block_config=(6, 12, 24, 16))
DenseNetBC169 = partial(DenseNetImageNetBC, block_config=(6, 12, 32, 32))
DenseNetBC201 = partial(DenseNetImageNetBC, block_config=(6, 12, 48, 32))
DenseNetBC264 = partial(DenseNetImageNetBC, block_config=(6, 12, 64, 48))


DenseNetSmall = partial(
    _DenseNet, is_small=True, DenseBlock=DNBlockStandard, TransLayer=TransitionLayer
)
# config = (depth - 4) // 3
DenseNetSmall40 = partial(DenseNetSmall, block_config=(12, 12, 12))
DenseNetSmall100 = partial(DenseNetSmall, block_config=(32, 32, 32))


DenseNetImageNetBC = partial(
    _DenseNet,
    is_small=True,
    DenseBlock=DNBlockBC,
    TransLayer=TransitionLayerCompression,
)
# config = (depth - 4) // 6
DenseNetSmallBC100 = partial(DenseNetImageNetBC, block_config=(16, 16, 16))
DenseNetSmallBC250 = partial(DenseNetImageNetBC, block_config=(41, 41, 41))
DenseNetSmallBC190 = partial(DenseNetImageNetBC, block_config=(31, 31, 31))
