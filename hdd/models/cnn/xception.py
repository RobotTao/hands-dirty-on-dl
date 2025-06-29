import torch
import torch.nn as nn
from torch import Tensor


class _SeparableConv(nn.Sequential):
    def __init__(self, in_channel: int, out_channel: int):
        """Xception的可分离卷积.注意1x1卷积和depthwise卷积之间没有non-linearity"""
        layers = [
            nn.Conv2d(
                in_channel,
                in_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=in_channel,
            ),
            nn.Conv2d(in_channel, out_channel, kernel_size=1),
        ]
        super().__init__(*layers)


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channel: int, out_channel: int):
        """SeparableConv with BN."""
        layers = [
            _SeparableConv(in_channel, out_channel),
            nn.BatchNorm2d(out_channel),
        ]
        super().__init__(*layers)


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


class Residual(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        fn: nn.Module,
        need_downsample: bool,
    ):
        super().__init__()
        self.downsample = nn.Identity()
        if need_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channel, out_channel),
            )
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.relu(self.downsample(x) + self.fn(x))


class EntryFlow(nn.Sequential):
    def __init__(self):

        fn1 = nn.Sequential(
            SeparableConvBN(64, 128),
            nn.ReLU(inplace=True),
            SeparableConvBN(128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        fn2 = nn.Sequential(
            nn.ReLU(),
            SeparableConvBN(128, 256),
            nn.ReLU(inplace=True),
            SeparableConvBN(256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        fn3 = nn.Sequential(
            nn.ReLU(),
            SeparableConvBN(256, 728),
            nn.ReLU(inplace=True),
            SeparableConvBN(728, 728),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        layers = [
            _Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            _Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            Residual(64, 128, fn1, need_downsample=True),
            Residual(128, 256, fn2, need_downsample=True),
            Residual(256, 728, fn3, need_downsample=True),
        ]
        super().__init__(*layers)


class MiddleFlow(nn.Sequential):
    def __init__(self) -> None:
        channel = 728

        def _build_residual_branch():
            return nn.Sequential(
                nn.ReLU(),
                SeparableConvBN(channel, channel),
                nn.ReLU(inplace=True),
                SeparableConvBN(channel, channel),
                nn.ReLU(inplace=True),
                SeparableConvBN(channel, channel),
            )

        layers = [
            Residual(channel, channel, _build_residual_branch(), False),
            Residual(channel, channel, _build_residual_branch(), False),
            Residual(channel, channel, _build_residual_branch(), False),
            Residual(channel, channel, _build_residual_branch(), False),
            Residual(channel, channel, _build_residual_branch(), False),
            Residual(channel, channel, _build_residual_branch(), False),
            Residual(channel, channel, _build_residual_branch(), False),
            Residual(channel, channel, _build_residual_branch(), False),
        ]
        super().__init__(*layers)


class ExitFlow(nn.Module):
    def __init__(self, num_classes: int, dropout: float):
        super().__init__()
        fn = nn.Sequential(
            nn.ReLU(),
            SeparableConvBN(728, 728),
            nn.ReLU(inplace=True),
            SeparableConvBN(728, 1024),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.feature = nn.Sequential(
            Residual(728, 1024, fn, need_downsample=True),
            SeparableConvBN(1024, 1536),
            nn.ReLU(inplace=True),
            SeparableConvBN(1536, 2048),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.feature(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


class XceptionNet(nn.Sequential):
    def __init__(self, num_classes: int, dropout: float):
        layers = [
            EntryFlow(),
            MiddleFlow(),
            ExitFlow(num_classes, dropout),
        ]
        super().__init__(*layers)
