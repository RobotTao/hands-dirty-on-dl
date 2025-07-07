# 注解版 https://github.com/ShichenLiu/CondenseNet

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.autograd import Variable


class _Conv2d(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        dropout=0,
    ) -> None:
        super().__init__()
        self.add_module("norm", nn.BatchNorm2d(in_channels))
        self.add_module("relu", nn.ReLU(inplace=True))
        if dropout != 0:
            self.add_module("dropout", nn.Dropout(dropout))
        self.add_module(
            "conv2d",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
                groups=groups,
            ),
        )


def shuffle_layer(x: Tensor, groups: int) -> Tensor:
    """Shuffle Layers"""
    return rearrange(x, "N (G C) H W -> N (C G) H W", G=groups)


class LearnedGroupConv(nn.Module):
    global_progress = 0.0

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        condense_factor: int = -1,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout_rate = dropout_rate
        if self.dropout_rate > 0:
            self.drop = nn.Dropout(dropout_rate, inplace=False)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=1,
            bias=False,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.condense_factor = condense_factor
        if self.condense_factor == -1:
            self.condense_factor = self.groups
        ### Parameters that should be carefully used
        self.register_buffer("_count", torch.zeros(1))
        self.register_buffer("_stage", torch.zeros(1))
        self.register_buffer("_mask", torch.ones(self.conv.weight.size()))
        ### Check if arguments are valid
        assert (
            self.in_channels % self.groups == 0
        ), "group number can not be divided by input channels"
        assert (
            self.in_channels % self.condense_factor == 0
        ), "condensation factor can not be divided by input channels"
        assert (
            self.out_channels % self.groups == 0
        ), "group number can not be divided by output channels"

    def forward(self, x):
        self._check_drop()
        x = self.norm(x)
        x = self.relu(x)
        if self.dropout_rate > 0:
            x = self.drop(x)
        ### Masked output
        weight = self.conv.weight * self.mask
        return F.conv2d(
            x, weight, None, self.conv.stride, self.conv.padding, self.conv.dilation, 1
        )

    def _check_drop(self):
        progress = LearnedGroupConv.global_progress
        delta = 0
        ### Get current stage
        for i in range(self.condense_factor - 1):
            if progress * 2 < (i + 1) / (self.condense_factor - 1):
                stage = i
                break
        else:
            stage = self.condense_factor - 1
        ### Check for dropping
        if not self._reach_stage(stage):
            self.stage = stage
            delta = self.in_channels // self.condense_factor
        if delta > 0:
            self._dropping(delta)
        return

    def _dropping(self, delta):
        weight = self.conv.weight * self.mask
        ### Sum up all kernels
        ### Assume only apply to 1x1 conv to speed up
        assert weight.size()[-1] == 1
        weight = weight.abs().squeeze()
        assert weight.size()[0] == self.out_channels
        assert weight.size()[1] == self.in_channels
        d_out = self.out_channels // self.groups
        ### Shuffle weight
        weight = weight.view(
            d_out, self.groups, self.in_channels
        )  # 它认为属于同一个group的channel不相连!!!!!
        weight = weight.transpose(0, 1).contiguous()
        weight = weight.view(self.out_channels, self.in_channels)
        ### Sort and drop
        for i in range(self.groups):
            wi = weight[i * d_out : (i + 1) * d_out, :]
            ### Take corresponding delta index
            di = wi.sum(0).sort()[1][self.count : self.count + delta]
            for d in di.data:
                self._mask[i :: self.groups, d, :, :].fill_(0)
        self.count = self.count + delta

    @property
    def count(self):
        return int(self._count[0])

    @count.setter
    def count(self, val):
        self._count.fill_(val)

    @property
    def stage(self):
        return int(self._stage[0])

    @stage.setter
    def stage(self, val):
        self._stage.fill_(val)

    @property
    def mask(self):
        return Variable(self._mask)

    def _reach_stage(self, stage):
        return (self._stage >= stage).all()

    @property
    def lasso_loss(self):
        if self._reach_stage(self.groups - 1):
            return 0
        weight = self.conv.weight * self.mask
        ### Assume only apply to 1x1 conv to speed up
        assert weight.size()[-1] == 1
        weight = weight.squeeze().pow(2)
        d_out = self.out_channels // self.groups
        ### Shuffle weight
        weight = weight.view(d_out, self.groups, self.in_channels)
        weight = weight.sum(0).clamp(min=1e-6).sqrt()
        return weight.sum()


class CondensingLinear(nn.Module):
    """将线性层根据其权重L1-Norm挑选出权重最大的一部分"""

    def __init__(self, model: nn.Linear, keep_rate=0.5) -> None:
        """ctor.

        Args:
            model: 线性层
            keep_rate: 输入层保留的比例. Defaults to 0.5.
        """
        super(CondensingLinear, self).__init__()
        self.in_features = int(model.in_features * keep_rate)
        self.out_features = model.out_features
        self.linear = nn.Linear(self.in_features, self.out_features)
        self.register_buffer("index", torch.LongTensor(self.in_features))
        _, index = model.weight.data.abs().sum(0).sort()
        # 选取最大的in_features个维度
        index = index[model.in_features - self.in_features :]
        # output feature 和 输入模型相同,直接拷贝即可
        self.linear.bias.data = model.bias.data.clone()
        for i in range(self.in_features):
            self.index[i] = index[i]
            self.linear.weight.data[:, i] = model.weight.data[:, index[i]]

    def forward(self, x):
        x = torch.index_select(x, 1, Variable(self.index))
        x = self.linear(x)
        return x


class CondenseLinear(nn.Module):
    """Condense之后的线性层.它是由CondensingLinear导出"""

    def __init__(self, in_features, out_features, drop_rate=0.5):
        super().__init__()
        self.in_features = int(in_features * drop_rate)
        self.out_features = out_features
        self.linear = nn.Linear(self.in_features, self.out_features)
        self.register_buffer("index", torch.LongTensor(self.in_features))

    def forward(self, x: Tensor) -> Tensor:
        x = torch.index_select(x, 1, Variable(self.index))
        x = self.linear(x)
        return x


class CondenseConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.conv = _Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=self.groups,
        )
        self.register_buffer("index", torch.LongTensor(self.in_channels))
        self.index.fill_(0)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.index_select(x, 1, Variable(self.index))
        x = self.conv(x)
        x = shuffle_layer(x, self.groups)
        return x


class CondensingConv(nn.Module):
    def __init__(self, model: LearnedGroupConv) -> None:
        super().__init__()
        self.in_channels = (
            model.conv.in_channels * model.groups // model.condense_factor
        )  # 即为论文中的(R*G)//C
        self.out_channels = model.conv.out_channels
        self.groups = model.groups
        self.condense_factor = model.condense_factor
        self.conv = _Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=model.conv.kernel_size,
            stride=model.conv.stride,
            padding=model.conv.padding,
            groups=self.groups,
        )
        self.register_buffer("index", torch.LongTensor(self.in_channels))
        index = 0  # 输出conv的input channel index, < self.in_channels
        mask = model._mask.mean(-1).mean(-1)
        out_channels_per_group = self.out_channels // self.groups
        in_channels_per_group = self.in_channels // self.groups
        for g in range(self.groups):  # g为group index
            for j in range(model.conv.in_channels):  # j为输入维度index
                if index < in_channels_per_group * (g + 1) and mask[g, j] == 1:
                    idx_j = index % in_channels_per_group  # 最终input channel的index
                    # 属于同一个group的output channel选取的input channel是一样的
                    for i in range(
                        out_channels_per_group
                    ):  # i为每个group内的输出channel index
                        # 这里i,j,g的含义和论文中符号一致

                        idx_i = int(
                            i + g * out_channels_per_group
                        )  # 最终output channel的index

                        self.conv.conv2d.weight.data[idx_i, idx_j, :, :] = (
                            model.conv.weight.data[int(g + i * self.groups), j, :, :]
                        )
                        # The arrange of the model.conv.weight.data is
                        # "(out_channel_per_group, groups) in_channels 1 1"
                        # model.conv排布和self.conv.conv2d是反的!!!!

                        self.conv.norm.weight.data[index] = model.norm.weight.data[j]
                        self.conv.norm.bias.data[index] = model.norm.bias.data[j]
                        self.conv.norm.running_mean[index] = model.norm.running_mean[j]
                        self.conv.norm.running_var[index] = model.norm.running_var[j]
                    self.index[index] = j
                    index += 1

    def forward(self, x):
        x = torch.index_select(x, 1, Variable(self.index))
        x = self.conv(x)
        x = shuffle_layer(x, self.groups)
        return x


class _DenseLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        growth_rate,
        group_1x1,
        group_3x3,
        bottleneck,
        condense_factor,
        dropout_rate,
    ):
        super(_DenseLayer, self).__init__()
        self.group_1x1 = group_1x1
        self.group_3x3 = group_3x3
        ### 1x1 conv i --> b*k
        self.conv_1 = LearnedGroupConv(
            in_channels,
            bottleneck * growth_rate,
            kernel_size=1,
            groups=self.group_1x1,
            condense_factor=condense_factor,
            dropout_rate=dropout_rate,
        )
        ### 3x3 conv b*k --> k
        self.conv_2 = _Conv2d(
            bottleneck * growth_rate,
            growth_rate,
            kernel_size=3,
            padding=1,
            groups=self.group_3x3,
        )

    def forward(self, x):
        x_ = x
        x = self.conv_1(x)
        x = self.conv_2(x)
        return torch.cat([x_, x], 1)


class _DenseBlock(nn.Sequential):
    def __init__(
        self,
        num_layers,
        in_channels,
        growth_rate,
        group_1x1,
        group_3x3,
        bottleneck,
        condense_factor,
        dropout_rate,
    ):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                in_channels + i * growth_rate,
                growth_rate,
                group_1x1,
                group_3x3,
                bottleneck,
                condense_factor,
                dropout_rate,
            )
            self.add_module("denselayer_%d" % (i + 1), layer)


class _Transition(nn.Module):
    def __init__(self) -> None:
        super(_Transition, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        return x


class CondenseNet(nn.Module):
    def __init__(
        self,
        num_classes,
        group_1x1,
        group_3x3,
        bottleneck,
        condense_factor,
        dropout_rate,
        stages,
        growth,
        data,
    ) -> None:

        super(CondenseNet, self).__init__()
        self.group_1x1 = group_1x1
        self.group_3x3 = group_3x3
        self.bottleneck = bottleneck
        self.condense_factor = condense_factor
        self.dropout_rate = dropout_rate
        self.stages = stages
        self.growth = growth
        assert len(self.stages) == len(self.growth)
        self.progress = 0.0
        if data in ["cifar10", "cifar100"]:
            self.init_stride = 1
            self.pool_size = 8
        else:
            self.init_stride = 2
            self.pool_size = 7

        self.features = nn.Sequential()
        ### Initial nChannels should be 3
        self.num_features = 2 * self.growth[0]
        ### Dense-block 1 (224x224)
        self.features.add_module(
            "init_conv",
            nn.Conv2d(
                3,
                self.num_features,
                kernel_size=3,
                stride=self.init_stride,
                padding=1,
                bias=False,
            ),
        )
        for i in range(len(self.stages)):
            ### Dense-block i
            self.add_block(i)
        ### Linear layer
        self.classifier = nn.Linear(self.num_features, num_classes)

        ### initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        return

    def add_block(self, i):
        ### Check if ith is the last one
        last = i == len(self.stages) - 1
        block = _DenseBlock(
            num_layers=self.stages[i],
            in_channels=self.num_features,
            growth_rate=self.growth[i],
            group_1x1=self.group_1x1,
            group_3x3=self.group_3x3,
            bottleneck=self.bottleneck,
            condense_factor=self.condense_factor,
            dropout_rate=self.dropout_rate,
        )
        self.features.add_module("denseblock_%d" % (i + 1), block)
        self.num_features += self.stages[i] * self.growth[i]
        if not last:
            trans = _Transition()
            self.features.add_module("transition_%d" % (i + 1), trans)
        else:
            self.features.add_module("norm_last", nn.BatchNorm2d(self.num_features))
            self.features.add_module("relu_last", nn.ReLU(inplace=True))
            self.features.add_module("pool_last", nn.AvgPool2d(self.pool_size))

    def forward(self, x, progress=None):
        if progress:
            LearnedGroupConv.global_progress = progress
        features = self.features(x)
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out


def is_pruned(layer):
    try:
        layer.mask
        return True
    except AttributeError:
        return False


def get_num_gen(gen):
    return sum(1 for x in gen)


def is_leaf(model):
    return get_num_gen(model.children()) == 0


def convert_model(model):
    for m in model._modules:
        child = model._modules[m]
        if is_leaf(child):
            if isinstance(child, nn.Linear):
                model._modules[m] = CondensingLinear(child, 0.5)
                del child
        elif is_pruned(child):
            model._modules[m] = CondensingConv(child)
            del child
        else:
            convert_model(child)
