"""AlexNet implementation"""

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class _Conv2dBlock(nn.Sequential):
    """Basic conv2d block of alexnet."""

    def __init__(self, in_channel: int, out_channel: int, add_norm_layer: bool, **argv):
        """Init function.

        Args:
            in_channel (int): input channels.
            out_channel (int): output channels.
            add_norm_layer (bool): Whether to add norm layer or not.
            argv: arguments passed to nn.Conv2d.

        Returns:
            Conv2d block.
        """
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channel,
            out_channel,
            **argv,
        )
        if add_norm_layer:
            self.nb = nn.BatchNorm2d(out_channel)
        self.activation = nn.ReLU(inplace=True)


class AlexNet(nn.Module):
    def __init__(
        self, num_classes: int = 1000, dropout: float = 0.5, add_norm_layer=True
    ):
        """Initialize AlexNet

        Args:
            num_classes (int, optional): Number of classes. Defaults to 1000.
            dropout (float, optional): dropout probability. Defaults to 0.5.
            add_norm_layer (bool, optional): Whether to add batch norm layer. Defaults to True.
        """
        super().__init__()
        # input shape (N,3,224,224)
        self.features = nn.Sequential(
            _Conv2dBlock(
                3,
                96,
                add_norm_layer=add_norm_layer,
                kernel_size=11,
                stride=4,
                padding=2,
            ),  # output shape (N,96,55,55)
            nn.MaxPool2d(kernel_size=3, stride=2),  # output shape (N,96,27,27)
            _Conv2dBlock(
                96,
                256,
                add_norm_layer=add_norm_layer,
                kernel_size=5,
                stride=1,
                padding=2,
            ),  # output shape (N,256,27,27)
            nn.MaxPool2d(kernel_size=3, stride=2),  # output shape (N,256,13,13)
            _Conv2dBlock(
                256,
                384,
                add_norm_layer=add_norm_layer,
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # output shape (N,384,13,13)
            _Conv2dBlock(
                384,
                384,
                add_norm_layer=add_norm_layer,
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # output shape (N,384,13,13)
            _Conv2dBlock(
                384,
                256,
                add_norm_layer=add_norm_layer,
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # output shape (N,256,13,13)
            nn.MaxPool2d(kernel_size=3, stride=2),  # output shape (N,256,6,6)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, X: Tensor) -> Tensor:
        X = self.features(X)
        X = torch.flatten(X, 1)
        X = self.classifier(X)
        return X
