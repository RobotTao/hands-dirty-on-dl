from typing import Tuple

import torch
from IPython import embed
from scipy.io import hb_write
from torch import Tensor, nn


class PositionWiseMLP(nn.Sequential):
    def __init__(self, embed_dim: int, diff_dim: int, dropout: float) -> None:
        """Positionwise feedforward mlp.

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
            nn.GELU(),
        ]
        super().__init__(*layers)


class SublayerConnection(nn.Module):
    def __init__(self, embed_dim: int, dropout: float, post_norm: bool = False) -> None:
        """Sublayer module.

        Args:
            embed_dim: Input feature size.
            dropout: dropout ratio.
            post_norm: Whether to use post-norm or pre-norm architecture.
                       关于
                       see: https://zhuanlan.zhihu.com/p/480783670
                            https://arxiv.org/pdf/2002.04745
        """
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.post_norm = post_norm

    def forward(self, x: Tensor, sublayer: nn.Module) -> Tensor:
        if self.post_norm:
            return self.ln(x + self.dropout(sublayer(x)))
        else:
            return x + self.dropout(sublayer(self.ln(x)))


class EncoderLayer(nn.Module):
    def __init__(
        self, embed_dim: int, n_heads: int, diff_dim: int, dropout: float
    ) -> None:
        super().__init__()
        # self attention
        self.self_atten = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.sublayer_1 = SublayerConnection(embed_dim, dropout)
        self.mlp = PositionWiseMLP(embed_dim, diff_dim, dropout)
        self.sublayer_2 = SublayerConnection(embed_dim, dropout)

    def forward(self, x: Tensor):
        x = self.sublayer_1(
            x, lambda x: self.self_atten(x, x, x, need_weights=False)[0]
        )
        x = self.sublayer_2(x, self.mlp)
        return x


class Encoder(nn.Sequential):
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        diff_dim: int,
        dropout: float,
        num_layers: int,
    ) -> None:
        layers = [
            EncoderLayer(embed_dim, n_heads, diff_dim, dropout)
            for _ in range(num_layers)
        ]
        super().__init__(*layers)


class ViT(nn.Module):
    def __init__(
        self,
        num_classes: int,
        image_size: int,
        patch_size: int,
        embed_dim: int,
        n_heads: int,
        diff_dim: int,
        dropout: float,
        num_layers: int,
    ) -> None:
        super().__init__()
        assert image_size % patch_size == 0
        self.patch_size = patch_size
        n_h = image_size // patch_size
        n_w = image_size // patch_size
        self.embed_dim = embed_dim
        seq_length = n_h * n_w + 1  # Extra 1 is for class token
        self.pos_embedding = nn.Parameter(
            torch.empty(1, seq_length, embed_dim).normal_(std=0.02)
        )  # from BERT
        self.seq_project = nn.Sequential(
            nn.Linear(3 * patch_size * patch_size, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.dropout = nn.Dropout(0.0)
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)
        self.class_token = nn.Parameter(torch.empty(1, 1, embed_dim).normal_(std=0.05))
        self.encoder = Encoder(embed_dim, n_heads, diff_dim, dropout, num_layers)
        self.classifier = nn.Linear(embed_dim, num_classes)

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def process_input(self, X: Tensor) -> Tensor:
        """X is of shape (N,C,H,W).
        Returns N, (HW/P^2), P*P*C
        """
        N, C, H, W = X.shape
        n_h = H // self.patch_size
        n_w = W // self.patch_size
        X = X.reshape(N, C, n_h, self.patch_size, n_w, self.patch_size)
        X = X.permute(0, 2, 4, 3, 5, 1).reshape(N, n_h * n_w, -1)
        projected_X = self.seq_project(X)  # (N,hidden_dim,NH, NW)
        return projected_X

    def forward(self, X: Tensor) -> Tensor:
        X = self.process_input(X)  # (N,S,D)
        N = X.shape[0]
        tokens = self.class_token.expand(N, -1, -1)
        tokens = torch.cat([tokens, X], dim=1) + self.pos_embedding
        tokens = self.dropout(tokens)
        tokens = self.encoder(tokens)
        feature = self.ln(tokens[:, 0])
        return self.classifier(feature)
