"""LeNet implementation"""

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LeNet(nn.Module):
    """LeNet-5 Implementation."""

    def __init__(
        self,
        num_classes: int,
        activation: Optional[Callable[..., nn.Module]] = None,
        init_weight: bool = False,
    ) -> None:
        """_summary_

        Args:
            num_classes (int): number of classification task classes.
            activation (Optional[Callable[..., nn.Module]], optional): activation type.
                it could be nn.ReLU, nn.Tanh, nn.Sigmoid.
            init_weight (bool, optional): _description_. Defaults to False.
        """
        super().__init__()

        if activation is None:
            activation = nn.Tanh

        # input shape (N,1,32,32)
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),  # output shape (N,6,28,28)
            activation(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # output shape (N,6,14,14)
            nn.Conv2d(6, 16, kernel_size=5),  # output shape (N,16,10,10)
            activation(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # output shape (N,16,5,5)
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            activation(),
            nn.Linear(120, 84),
            activation(),
            nn.Linear(84, num_classes),
        )

        if init_weight:
            self._init_weight(activation)

    def _init_weight(self, activation: Callable[..., nn.Module]) -> None:
        """Initialize the weight based on the activation type."""

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if activation == nn.ReLU:
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                else:
                    nn.init.xavier_normal_(m.weight, nn.init.calculate_gain("tanh"))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, X: Tensor) -> Tensor:
        X = self.features(X)  # [batch, 1, 28, 28] -> [batch, 16, 5, 5]
        X = torch.flatten(X, 1)  # output shape [batch, 16*5*5]
        X = self.classifier(X)  # output shape [batch, num_classes]
        return X
