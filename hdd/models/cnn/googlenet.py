"""Simple GooLeNet Implementation."""

from typing import Callable, Tuple, Union

import torch
import torch.nn as nn


class InceptionConv2d(nn.Module):
    """Basic Inception 2D convolution block."""

    def __init__(self, in_channel, out_channel, **kwargs) -> None:
        """Init.

        Args:
            in_channel: input channel to the Conv2d class.
            out_channel: output channel to the Conv2d class.
            kwargs: Other parameters to the Conv2d class.
        """
        super().__init__()
        self.conv2d = nn.Conv2d(in_channel, out_channel, bias=False, **kwargs)
        self.norm = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        X = self.conv2d(X)
        X = self.norm(X)
        X = self.relu(X)
        return X


class InceptionAux(nn.Sequential):
    """Auxiliary classifier module."""

    def __init__(self, in_channel, num_classes, dropout=0.7):
        layers = [
            nn.AdaptiveAvgPool2d((4, 4)),
            InceptionConv2d(in_channel, 128, kernel_size=1),
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes),
        ]
        super().__init__(*layers)


class GoogLeNet(nn.Module):
    def __init__(
        self,
        Inception: Callable[..., nn.Module],
        num_classes: int,
        dropout: float = 0.5,
        use_aux: bool = False,
    ) -> None:
        """_summary_

        Args:
            Inception: Inception module type.
            num_classes: Number of classes to classify.
            dropout: Dropout probability. Defaults to 0.5.
            use_aux: Whether to use auxiliary classifier. Defaults to False.
        """
        super().__init__()
        self.use_aux = use_aux

        self.aux1 = None
        self.aux2 = None
        if use_aux:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        self.pre_inception = nn.Sequential(
            InceptionConv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            InceptionConv2d(64, 64, kernel_size=1),
            InceptionConv2d(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        )
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes),
        )

    def forward(
        self, X
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        X = self.pre_inception(X)
        X = self.inception3a(X)
        X = self.inception3b(X)
        X = self.maxpool3(X)
        aux1_input = self.inception4a(X)

        X = self.inception4b(aux1_input)
        X = self.inception4c(X)
        aux2_input = self.inception4d(X)

        X = self.inception4e(aux2_input)
        X = self.maxpool4(X)
        X = self.inception5a(X)
        X = self.inception5b(X)
        X = self.classifier(X)
        if self.aux1 and self.training and self.aux2:
            aux1 = self.aux1(aux1_input)
            aux2 = self.aux2(aux2_input)
            return X, aux1, aux2
        return X


class InceptionV1(nn.Module):
    def __init__(
        self,
        in_channel,
        ch_1x1,
        ch_3x3_reduce,
        ch_3x3,
        ch_5x5_reduce,
        ch_5x5,
        ch_pool_proj,
    ) -> None:
        """Inception Moduler V1.

        Args:
            in_channel: Input channel number.
            ch_1x1: 1x1 convolution output channel.
            ch_3x3_reduce: reduction layer output channel before 3x3 conv.
            ch_3x3: output channel of 3x3 conv.
            ch_5x5_reduce: reduction layer output channel before 5x5 conv.
            ch_5x5: output channel of 5x5 conv.
            ch_pool_proj: output channel of projection branch.
        """
        super().__init__()
        self.branch1 = InceptionConv2d(in_channel, ch_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            InceptionConv2d(in_channel, ch_3x3_reduce, kernel_size=1),
            InceptionConv2d(ch_3x3_reduce, ch_3x3, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            InceptionConv2d(in_channel, ch_5x5_reduce, kernel_size=1),
            InceptionConv2d(ch_5x5_reduce, ch_5x5, kernel_size=5, padding=2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            InceptionConv2d(in_channel, ch_pool_proj, kernel_size=1),
        )

    def forward(self, X) -> torch.Tensor:
        branch1 = self.branch1(X)
        branch2 = self.branch2(X)
        branch3 = self.branch3(X)
        branch4 = self.branch4(X)
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)


class InceptionV2(nn.Module):
    def __init__(
        self,
        in_channel,
        ch_1x1,
        ch_3x3_reduce,
        ch_3x3,
        ch_5x5_reduce,
        ch_5x5,
        ch_pool_proj,
    ) -> None:
        super().__init__()
        self.branch1 = InceptionConv2d(in_channel, ch_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            InceptionConv2d(in_channel, ch_3x3_reduce, kernel_size=1),
            InceptionConv2d(ch_3x3_reduce, ch_3x3, kernel_size=3, padding=1),
        )
        # 唯一和V1不同的地方是5✖5卷积被替换成两个3x3卷积
        self.branch3 = nn.Sequential(
            InceptionConv2d(in_channel, ch_5x5_reduce, kernel_size=1),
            InceptionConv2d(ch_5x5_reduce, ch_5x5_reduce, kernel_size=3, padding=1),
            InceptionConv2d(ch_5x5_reduce, ch_5x5, kernel_size=3, padding=1),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            InceptionConv2d(in_channel, ch_pool_proj, kernel_size=1),
        )

    def forward(self, X) -> torch.Tensor:
        branch1 = self.branch1(X)
        branch2 = self.branch2(X)
        branch3 = self.branch3(X)
        branch4 = self.branch4(X)
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)


class InceptionV3(nn.Module):
    def __init__(
        self,
        in_channel,
        ch_1x1,
        ch_3x3_reduce,
        ch_3x3,
        ch_5x5_reduce,
        ch_5x5,
        ch_pool_proj,
    ) -> None:
        super().__init__()
        # fmt: off
        self.branch1 = InceptionConv2d(in_channel, ch_1x1, kernel_size=1)
        # 所有的nxn卷积都变成nx1, 1xn卷积
        self.branch2 = nn.Sequential(
            InceptionConv2d(in_channel, ch_3x3_reduce, kernel_size=1),
            InceptionConv2d(ch_3x3_reduce, ch_3x3_reduce, kernel_size=(3, 1), padding=(1, 0)),
            InceptionConv2d(ch_3x3_reduce, ch_3x3, kernel_size=(1, 3), padding=(0, 1)),
        )
        self.branch3 = nn.Sequential(
            InceptionConv2d(in_channel, ch_5x5_reduce, kernel_size=1),
            InceptionConv2d(ch_5x5_reduce, ch_5x5_reduce, kernel_size=(3, 1), padding=(1, 0)),
            InceptionConv2d(ch_5x5_reduce, ch_5x5_reduce, kernel_size=(1, 3), padding=(0, 1)),
            InceptionConv2d(ch_5x5_reduce, ch_5x5_reduce, kernel_size=(3, 1), padding=(1, 0)),
            InceptionConv2d(ch_5x5_reduce, ch_5x5, kernel_size=(1, 3), padding=(0, 1)),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            InceptionConv2d(in_channel, ch_pool_proj, kernel_size=1),
        )
        # fmt: on

    def forward(self, X) -> torch.Tensor:
        branch1 = self.branch1(X)
        branch2 = self.branch2(X)
        branch3 = self.branch3(X)
        branch4 = self.branch4(X)
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)
