# From https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py
import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Optional

import torch
from torch import Tensor, nn
from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation

from hdd.models.nn_utils import make_divisible


class SimpleStemIN(Conv2dNormActivation):
    """Simple stem for ImageNet: 3x3, BN, ReLU."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
    ) -> None:
        super().__init__(
            width_in,
            width_out,
            kernel_size=3,
            stride=2,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )


class BottleneckTransform(nn.Sequential):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: Optional[float],
    ) -> None:
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        w_b = int(round(width_out * bottleneck_multiplier))
        g = w_b // group_width

        layers["a"] = Conv2dNormActivation(
            width_in,
            w_b,
            kernel_size=1,
            stride=1,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )
        layers["b"] = Conv2dNormActivation(
            w_b,
            w_b,
            kernel_size=3,
            stride=stride,
            groups=g,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

        if se_ratio:
            # The SE reduction ratio is defined with respect to the
            # beginning of the block
            width_se_out = int(round(se_ratio * width_in))
            layers["se"] = SqueezeExcitation(
                input_channels=w_b,
                squeeze_channels=width_se_out,
                activation=activation_layer,
            )

        layers["c"] = Conv2dNormActivation(
            w_b,
            width_out,
            kernel_size=1,
            stride=1,
            norm_layer=norm_layer,
            activation_layer=None,
        )
        super().__init__(layers)


class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        group_width: int = 1,
        bottleneck_multiplier: float = 1.0,
        se_ratio: Optional[float] = None,
    ) -> None:
        super().__init__()

        # Use skip connection with projection if shape changes
        self.proj = None
        should_proj = (width_in != width_out) or (stride != 1)
        if should_proj:
            self.proj = Conv2dNormActivation(
                width_in,
                width_out,
                kernel_size=1,
                stride=stride,
                norm_layer=norm_layer,
                activation_layer=None,
            )
        self.f = BottleneckTransform(
            width_in,
            width_out,
            stride,
            norm_layer,
            activation_layer,
            group_width,
            bottleneck_multiplier,
            se_ratio,
        )
        self.activation = activation_layer(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        if self.proj is not None:
            x = self.proj(x) + self.f(x)
        else:
            x = x + self.f(x)
        return self.activation(x)


class AnyStage(nn.Sequential):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        depth: int,
        block_constructor: Callable[..., nn.Module],
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: Optional[float] = None,
        stage_index: int = 0,
    ) -> None:
        super().__init__()

        for i in range(depth):
            block = block_constructor(
                width_in if i == 0 else width_out,
                width_out,
                stride if i == 0 else 1,
                norm_layer,
                activation_layer,
                group_width,
                bottleneck_multiplier,
                se_ratio,
            )

            self.add_module(f"block{stage_index}-{i}", block)


class BlockParams:
    def __init__(
        self,
        depths: list[int],
        widths: list[int],
        group_widths: list[int],
        bottleneck_multipliers: list[float],
        strides: list[int],
        se_ratio: Optional[float] = None,
    ) -> None:
        self.depths = depths
        self.widths = widths
        self.group_widths = group_widths
        self.bottleneck_multipliers = bottleneck_multipliers
        self.strides = strides
        self.se_ratio = se_ratio

    @classmethod
    def from_init_params(
        cls,
        depth: int,
        w_0: int,
        w_a: float,
        w_m: float,
        group_width: int,
        bottleneck_multiplier: float = 1.0,
        se_ratio: Optional[float] = None,
        **kwargs: Any,
    ) -> "BlockParams":
        """
        Programmatically compute all the per-block settings,
        given the RegNet parameters.

        The first step is to compute the quantized linear block parameters,
        in log space. Key parameters are:
        - `w_a` is the width progression slope
        - `w_0` is the initial width
        - `w_m` is the width stepping in the log space

        In other terms
        `log(block_width) = log(w_0) + w_m * block_capacity`,
        with `bock_capacity` ramping up following the w_0 and w_a params.
        This block width is finally quantized to multiples of 8.

        The second step is to compute the parameters per stage,
        taking into account the skip connection and the final 1x1 convolutions.
        We use the fact that the output width is constant within a stage.
        """

        QUANT = 8
        STRIDE = 2

        if w_a < 0 or w_0 <= 0 or w_m <= 1 or w_0 % 8 != 0:
            raise ValueError("Invalid RegNet settings")
        # Compute the block widths. Each stage has one unique block width
        widths_cont = torch.arange(depth) * w_a + w_0
        block_capacity = torch.round(torch.log(widths_cont / w_0) / math.log(w_m))
        block_widths = (
            (
                torch.round(torch.divide(w_0 * torch.pow(w_m, block_capacity), QUANT))
                * QUANT
            )
            .int()
            .tolist()
        )
        num_stages = len(set(block_widths))

        # Convert to per stage parameters
        split_helper = zip(
            block_widths + [0],
            [0] + block_widths,
            block_widths + [0],
            [0] + block_widths,
        )
        splits = [w != wp or r != rp for w, wp, r, rp in split_helper]

        stage_widths = [w for w, t in zip(block_widths, splits[:-1]) if t]
        stage_depths = (
            torch.diff(torch.tensor([d for d, t in enumerate(splits) if t]))
            .int()
            .tolist()
        )

        strides = [STRIDE] * num_stages
        bottleneck_multipliers = [bottleneck_multiplier] * num_stages
        group_widths = [group_width] * num_stages

        # Adjust the compatibility of stage widths and group widths
        stage_widths, group_widths = cls._adjust_widths_groups_compatibilty(
            stage_widths, bottleneck_multipliers, group_widths
        )

        return cls(
            depths=stage_depths,
            widths=stage_widths,
            group_widths=group_widths,
            bottleneck_multipliers=bottleneck_multipliers,
            strides=strides,
            se_ratio=se_ratio,
        )

    def _get_expanded_params(self):
        return zip(
            self.widths,
            self.strides,
            self.depths,
            self.group_widths,
            self.bottleneck_multipliers,
        )

    @staticmethod
    def _adjust_widths_groups_compatibilty(
        stage_widths: list[int], bottleneck_ratios: list[float], group_widths: list[int]
    ) -> tuple[list[int], list[int]]:
        """
        Adjusts the compatibility of widths and groups,
        depending on the bottleneck ratio.
        """
        # Compute all widths for the current settings
        widths = [int(w * b) for w, b in zip(stage_widths, bottleneck_ratios)]
        group_widths_min = [min(g, w_bot) for g, w_bot in zip(group_widths, widths)]

        # Compute the adjusted widths so that stage and group widths fit
        ws_bot = [
            make_divisible(w_bot, g) for w_bot, g in zip(widths, group_widths_min)
        ]
        stage_widths = [int(w_bot / b) for w_bot, b in zip(ws_bot, bottleneck_ratios)]
        return stage_widths, group_widths_min


class RegNet(nn.Module):
    def __init__(
        self,
        block_params: BlockParams,
        num_classes: int = 1000,
        stem_width: int = 32,
        stem_type: Optional[Callable[..., nn.Module]] = None,
        block_type: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        if stem_type is None:
            stem_type = SimpleStemIN
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if block_type is None:
            block_type = ResBottleneckBlock
        if activation is None:
            activation = nn.ReLU

        # Ad hoc stem
        self.stem = stem_type(
            3,  # width_in
            stem_width,
            norm_layer,
            activation,
        )

        current_width = stem_width

        blocks = []
        for i, (
            width_out,
            stride,
            depth,
            group_width,
            bottleneck_multiplier,
        ) in enumerate(block_params._get_expanded_params()):
            blocks.append(
                (
                    f"block{i+1}",
                    AnyStage(
                        current_width,
                        width_out,
                        stride,
                        depth,
                        block_type,
                        norm_layer,
                        activation,
                        group_width,
                        bottleneck_multiplier,
                        block_params.se_ratio,
                        stage_index=i + 1,
                    ),
                )
            )

            current_width = width_out

        self.trunk_output = nn.Sequential(OrderedDict(blocks))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=current_width, out_features=num_classes)

        # Performs ResNet-style weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.trunk_output(x)

        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x


def create_regnet(name: str) -> RegNet:
    if name == "regnet_y_400mf":
        params = BlockParams.from_init_params(
            depth=16, w_0=48, w_a=27.89, w_m=2.09, group_width=8, se_ratio=0.25
        )
    elif name == "regnet_y_800mf":
        params = BlockParams.from_init_params(
            depth=14, w_0=56, w_a=38.84, w_m=2.4, group_width=16, se_ratio=0.25
        )
    elif name == "regnet_y_1_6gf":
        params = BlockParams.from_init_params(
            depth=27, w_0=48, w_a=20.71, w_m=2.65, group_width=24, se_ratio=0.25
        )
    elif name == "regnet_y_3_2gf":
        params = BlockParams.from_init_params(
            depth=21, w_0=80, w_a=42.63, w_m=2.66, group_width=24, se_ratio=0.25
        )
    elif name == "regnet_y_8gf":
        params = BlockParams.from_init_params(
            depth=17, w_0=192, w_a=76.82, w_m=2.19, group_width=56, se_ratio=0.25
        )
    elif name == "regnet_y_16gf":
        params = BlockParams.from_init_params(
            depth=18, w_0=200, w_a=106.23, w_m=2.48, group_width=112, se_ratio=0.25
        )
    elif name == "regnet_y_32gf":
        params = BlockParams.from_init_params(
            depth=20, w_0=232, w_a=115.89, w_m=2.53, group_width=232, se_ratio=0.25
        )
    elif name == "regnet_y_128gf":
        params = BlockParams.from_init_params(
            depth=27, w_0=456, w_a=160.83, w_m=2.52, group_width=264, se_ratio=0.25
        )
    elif name == "regnet_x_400mf":
        params = BlockParams.from_init_params(
            depth=22, w_0=24, w_a=24.48, w_m=2.54, group_width=16
        )
    elif name == "regnet_x_800mf":
        params = BlockParams.from_init_params(
            depth=16, w_0=56, w_a=35.73, w_m=2.28, group_width=16
        )
    elif name == "regnet_x_1_6gf":
        params = BlockParams.from_init_params(
            depth=18, w_0=80, w_a=34.01, w_m=2.25, group_width=24
        )
    elif name == "regnet_x_3_2gf":
        params = BlockParams.from_init_params(
            depth=25, w_0=88, w_a=26.31, w_m=2.25, group_width=48
        )
    elif name == "regnet_x_8gf":
        params = BlockParams.from_init_params(
            depth=23, w_0=80, w_a=49.56, w_m=2.88, group_width=120
        )
    elif name == "regnet_x_16gf":
        params = BlockParams.from_init_params(
            depth=22, w_0=216, w_a=55.59, w_m=2.1, group_width=128
        )
    elif name == "regnet_x_32gf":
        params = BlockParams.from_init_params(
            depth=23, w_0=320, w_a=69.86, w_m=2.0, group_width=168
        )
    return RegNet(params)
