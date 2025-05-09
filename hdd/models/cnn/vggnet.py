# 直接从torchision中借鉴来的

from typing import List, Union

from torch import nn


def _make_feature(
    cfgs: List[Union[str, int]], batch_norm: bool = True
) -> nn.Sequential:
    """Make feature extractor from the cfgs.

    Args:
        cfgs: Configuration of each layer of the

    Returns:
        feature extraction module.
    """
    in_channels = 3
    layers = []
    for cfg in cfgs:
        if isinstance(cfg, int):
            layers.append(
                nn.Conv2d(in_channels, cfg, kernel_size=3, stride=1, padding=1)
            )
            if batch_norm:
                layers.append(nn.BatchNorm2d(cfg))
            layers.append(nn.ReLU(inplace=True))
            in_channels = cfg
        elif cfg == "M":
            layers.append(nn.MaxPool2d(2, 2))
        else:
            raise ValueError("Not supported cfg letter.")
    return nn.Sequential(*layers)


class VGGNet(nn.Module):
    def __init__(
        self, cfgs, num_classes, dropout: float = 0.5, batch_norm: bool = True
    ):
        super().__init__()
        self.features = _make_feature(cfgs, batch_norm)
        self.AvgPool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 使用kaiming normal fan_in 初始化居然会发散
                nn.init.trunc_normal_(m.weight, 0, 0.01, -0.03, 0.03)
                nn.init.constant_(m.bias, 0)

    def forward(self, X):
        X = self.features(X)
        X = self.AvgPool(X)
        X = self.classifier(X)
        return X


# fmt: off
cfgs: dict[str, list[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}
# fmt: on
