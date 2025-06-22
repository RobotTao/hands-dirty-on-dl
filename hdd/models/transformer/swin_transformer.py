# 参考代码: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
# 为了方便理解,各层的输入输出皆为 (B,H,W,C)格式
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from sympy import shape
from torch import Tensor


def window_partition(x, window_size):
    """
    Args:
        x: (H, W)
        window_size (int): window size

    Returns:
        windows: (H//window_size, W//window_size, window_size * window_size)
    """
    H, W = x.shape
    x = x.view(H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 1, 3).flatten(2)
    return windows


class PositionWiseMLP(nn.Sequential):
    def __init__(self, embed_dim: int, diff_dim: int, dropout: float) -> None:
        """Position-wise feedforward mlp.

        Args:
            embed_dim: Model dimension.
            diff_dim: MLP hidden dimension. It is typically set to 4*embed_dim.
            dropout: dropout ratio.
        """
        layers = [
            nn.Linear(embed_dim, diff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(diff_dim, embed_dim),
            nn.Dropout(dropout),
        ]
        super().__init__(*layers)


class SublayerConnection(nn.Module):
    def __init__(self, embed_dim: int, post_norm: bool = False) -> None:
        """Sublayer module.

        Args:
            embed_dim: Input feature size.
            post_norm: Whether to use post-norm or pre-norm architecture.
                       关于
                       see: https://zhuanlan.zhihu.com/p/480783670
                            https://arxiv.org/pdf/2002.04745
        """
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim)
        self.post_norm = post_norm

    def forward(self, x: Tensor, sublayer: nn.Module) -> Tensor:
        # 这里把dropout去掉了,每个sublayer在最后一层会有自己的dropout
        if self.post_norm:
            return self.ln(x + sublayer(x))
        else:
            return x + sublayer(self.ln(x))


class PatchEmbed(nn.Module):
    """将输入图片,如(N,3,224,244)的局部patch映射为token."""

    def __init__(
        self, in_chans=3, image_shape=(224, 224), patch_shape=(4, 4), embed_dim=96
    ) -> None:
        """

        Args:
            in_chans: 输入图像channel数量. Defaults to 3.
            image_shape: 图像大小. Defaults to (224, 224).
            patch_shape: patch大小. Defaults to (4, 4).
            embed_dim: 输出特征维度. Defaults to 96.
        """
        super().__init__()
        assert image_shape[0] % patch_shape[0] == 0
        assert image_shape[1] % patch_shape[1] == 0
        self.image_shape = image_shape
        self.patch_shape = patch_shape
        self.project = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_shape,
            stride=patch_shape,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        assert H == self.image_shape[0] and W == self.image_shape[1]
        x = self.project(x)  # shape (N,embed_dim, ph, pw)
        x = x.permute(0, 2, 3, 1).contiguous()  # shape(N, ph, pw, embed_dim)
        return self.norm(x)


class PatchMerging(nn.Module):
    r"""Patch Merging Layer.

    Args:
        embed_dim (int): Number of input channels.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.reduction = nn.Linear(4 * embed_dim, 2 * embed_dim, bias=False)
        self.norm = nn.LayerNorm(4 * embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: B, H, W, C
        """
        B, H, W, C = x.shape
        assert C == self.embed_dim and H % 2 == 0 and W % 2 == 0
        x = rearrange(x, "B (hh h1) (ww h2) C -> B hh ww (h1 h2 C)", h1=2, h2=2)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class WindowedMultiHeadAttention(nn.Module):
    """Windowed multi-head self-attention module. W-MSA"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_shape: Tuple[int, int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.window_shape = window_shape

        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv_project = nn.Linear(embed_dim, 3 * embed_dim)
        self.output_project = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_shape[0] - 1) * (2 * window_shape[1] - 1), num_heads
            )
        )

        # get pair-wise relative position index for each token inside the window
        # 计算window内任意两个之间的坐标差值并将其映射为一个index
        # 坐标差值范围在([-(w-1),w-1],[-(h-1),h-1]),一共有(2h-1)*(2w-1)个
        # 我们只需将其1:1映射到[0,(2h-1)*(2w-1)-1]即可
        coords_h = torch.arange(self.window_shape[0])
        coords_w = torch.arange(self.window_shape[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_shape[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_shape[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_shape[1] - 1  # 乘以每行的数量
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: Tensor, mask: Optional[Tensor]) -> Tensor:
        """
        Args:
            x: shape (N,H,W,C)
            mask: shape (num_windows, elements_in_window, elements_in_window)

        Returns:
            (N,H,W,C)
        """
        N, H, W, C = x.shape
        assert H % self.window_shape[0] == 0 and W % self.window_shape[1] == 0
        assert C == self.embed_dim
        s_h = self.window_shape[0]
        s_w = self.window_shape[1]

        x = self.qkv_project(x)  # shape (N,H,W,3*C)
        x = rearrange(
            x,
            "N (w_h s_h) (w_w s_w) (three head head_dim) -> "
            + "N w_h w_w three head (s_h s_w) head_dim",
            head=self.num_heads,
            head_dim=self.head_dim,
            three=3,
            s_h=s_h,
            s_w=s_w,
        )

        q, k, v = x.chunk(3, dim=3)
        q, k, v = (
            q.squeeze(3),
            k.squeeze(3),
            v.squeeze(3),
        )  # (N, w_h, w_w, head, s_h * s_w, head_dim)

        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )  # (N, w_h, w_w, head, s_h * s_w, s_h * s_w)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ]
        relative_position_bias = rearrange(
            relative_position_bias,
            "(e1 e2) head -> head e1 e2",
            e1=s_h * s_w,
        )
        logits = logits + relative_position_bias

        if mask is not None:
            logits = logits + mask[:, :, None, :, :]
            atten = logits.softmax(dim=-1)
        else:
            atten = logits.softmax(dim=-1)  # (N, w_h, w_w, head, s_h * s_w, s_h * s_w)
        x = torch.matmul(atten, v)
        x = rearrange(
            x,
            "N w_h w_w head (s_h  s_w) head_dim -> "
            + "N (w_h s_h) (w_w s_w) (head head_dim)",
            s_h=s_h,
            head=self.num_heads,
        )
        x = self.dropout(self.output_project(x))
        return x


class SwinTransformerBlock(nn.Module):
    r"""Swin Transformer Block.

    Args:
        embed_dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        dropout (float, optional): Dropout rate. Default: 0.
    """

    def __init__(
        self,
        embed_dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        dropout=0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert (
            0 <= self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"

        attn_mask = self._create_atten_mask()
        self.register_buffer("attn_mask", attn_mask)

        self.sublayer_1 = SublayerConnection(embed_dim)
        self.attn = WindowedMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            window_shape=(window_size, window_size),
            dropout=dropout,
        )

        self.sublayer_2 = SublayerConnection(embed_dim)
        self.mlp = PositionWiseMLP(
            embed_dim=embed_dim,
            diff_dim=int(embed_dim * mlp_ratio),
            dropout=dropout,
        )

    def _create_atten_mask(self):
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((H, W))  # 1 H W 1
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[h, w] = cnt
                    cnt += 1
            # img_mask中来自相同区域的元素会有相同的值

            mask_windows = window_partition(
                img_mask, self.window_size
            )  # nH, nW, window_size* window_size
            attn_mask = mask_windows.unsqueeze(3) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(
                attn_mask == 0, float(0.0)
            )  # shape(h, w, s_h * s_w, s_h * s_w)
        else:
            attn_mask = None
        return attn_mask

    def forward(self, x: Tensor) -> Tensor:

        def shift_atten_unshift(input):
            # cyclic shift
            if self.shift_size > 0:
                input = torch.roll(
                    input, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
                )
            # W-MSA/SW-MSA
            input = self.attn(
                input, mask=self.attn_mask
            )  # nW*B, window_size*window_size, C

            # reverse cyclic shift
            if self.shift_size > 0:
                input = torch.roll(
                    input, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
                )
            return input

        x = self.sublayer_1(x, shift_atten_unshift)
        # FFN
        x = self.sublayer_2(x, self.mlp)

        return x


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.

    Args:
        embed_dim: Number of input channels.
        input_resolution: Input resolution.
        depth: Number of blocks.
        num_heads: Number of attention heads.
        window_size: Local window size.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim.
        dropout: Dropout rate. Default: 0.0
        patch_merge: Whether to do patch merge at the beginning.
    """

    def __init__(
        self,
        embed_dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        patch_merge: bool = False,
    ) -> None:

        super().__init__()
        # patch merging layer
        if patch_merge:
            self.merge_patch = PatchMerging(embed_dim)
            self.input_resolution = (input_resolution[0] // 2, input_resolution[1] // 2)
            self.embed_dim = embed_dim * 2
        else:
            self.merge_patch = None
            self.input_resolution = input_resolution
            self.embed_dim = embed_dim
        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    embed_dim=self.embed_dim,
                    input_resolution=self.input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.merge_patch is not None:
            x = self.merge_patch(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class SwinTransformer(nn.Module):
    r"""Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        dropout (float): Dropout rate. Default: 0
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4

    """

    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        dropout=0.0,
        window_size=7,
        mlp_ratio=4,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            in_chans=in_chans,
            image_shape=(img_size, img_size),
            patch_shape=(patch_size, patch_size),
            embed_dim=embed_dim,
        )
        # build layers
        self.layers = nn.ModuleList()
        input_size = img_size // patch_size
        for i_layer in range(self.num_layers):
            if i_layer > 1:
                embed_dim = embed_dim * 2
                input_size = input_size // 2
            layer = BasicLayer(
                embed_dim=embed_dim,
                input_resolution=(input_size, input_size),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                patch_merge=i_layer != 0,
            )
            self.layers.append(layer)

        self.norm = nn.LayerNorm(self.num_features)
        self.head = nn.Linear(self.num_features, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: Tensor) -> Tensor:

        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # (N,H,W,C)
        x = torch.mean(x, dim=(1, 2))
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        return x
