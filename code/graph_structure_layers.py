"""
Graph-Mamba 图支路：多头 GAT + 多头 attention 池化（multi_attn）。

与 graph_mamba_infer.json 中 graph_conv=gat、pooling=multi_attn 一致。
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


def _add_self_loop_mask(adj: torch.Tensor) -> torch.Tensor:
    B, N, _ = adj.shape
    eye = torch.eye(N, device=adj.device, dtype=adj.dtype).unsqueeze(0)
    return adj + eye


class MultiHeadGATBlock(nn.Module):
    """多头图注意力（稠密邻接 + mask），残差输出 hidden_dim。"""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim({hidden_dim}) 须能被 num_heads({num_heads}) 整除")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.lin = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, num_heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.empty(1, num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        self.leaky = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        H, d = self.num_heads, self.head_dim
        h = self.lin(x).view(B, N, H, d)
        a_src = (h * self.att_src).sum(-1)
        a_dst = (h * self.att_dst).sum(-1)
        e = self.leaky(a_src.unsqueeze(2) + a_dst.unsqueeze(1))
        eye = torch.eye(N, device=adj.device, dtype=adj.dtype).unsqueeze(0)
        mask = (adj + eye) > 0
        mask = mask.unsqueeze(-1).expand(B, N, N, H)
        e = e.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(e, dim=2)
        alpha = self.dropout(alpha)
        out = torch.einsum("bnjh,bjhd->bnhd", alpha, h)
        out = out.reshape(B, N, D)
        out = self.out_proj(out)
        return x + self.dropout(out)


def _build_multi_attn_merge(
    in_dim: int,
    hidden_dim: int,
    dropout: float,
    merge_activation: str,
) -> nn.Module:
    act = merge_activation.lower().strip()
    layers: list[nn.Module] = []
    if act == "ln_gelu":
        layers.append(nn.LayerNorm(in_dim))
    layers.append(nn.Linear(in_dim, hidden_dim))
    if act in ("gelu", "ln_gelu"):
        layers.append(nn.GELU())
    elif act == "relu":
        layers.append(nn.ReLU(inplace=False))
    elif act == "none":
        pass
    else:
        raise ValueError(
            f"merge_activation 须为 gelu|none|relu|ln_gelu，当前 {merge_activation!r}"
        )
    layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class MultiHeadAttnPool(nn.Module):
    """并行多条 attention，concat 后经 Linear 压回 hidden_dim。"""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        merge_activation: str = "ln_gelu",
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.merge_activation = str(merge_activation)
        ad = float(attn_dropout)
        self.attn_dropout = nn.Dropout(ad) if ad > 0.0 else None
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_heads)])
        in_dim = num_heads * hidden_dim
        self.merge = _build_multi_attn_merge(
            in_dim, hidden_dim, dropout, self.merge_activation
        )

    def forward_parts(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for att_lin in self.heads:
            logits = att_lin(x).squeeze(-1)
            logits = logits.masked_fill(~mask, float("-inf"))
            att = torch.softmax(logits, dim=-1)
            if self.attn_dropout is not None:
                att = self.attn_dropout(att)
            att = att.unsqueeze(-1)
            parts.append((x * att).sum(dim=1))
        return torch.cat(parts, dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.merge(self.forward_parts(x, mask))


def build_graph_block(
    kind: str,
    hidden_dim: int,
    dropout: float,
    gat_heads: int,
    sage_agg: str = "mean",
) -> nn.Module:
    """仅支持 graph_conv=gat（sage_agg 保留参数以兼容旧调用，已忽略）。"""
    k = kind.lower()
    if k not in ("gat",):
        raise ValueError(f"仅支持 graph_conv='gat'，收到 {kind!r}")
    return MultiHeadGATBlock(hidden_dim, gat_heads, dropout)


def build_pooling(
    mode: str,
    hidden_dim: int,
    pool_heads: int,
    set2set_steps: int,
    dropout: float,
    pool_merge_activation: str = "ln_gelu",
    pool_attn_dropout: float = 0.0,
) -> Tuple[nn.Module, int]:
    """仅支持 pooling=multi_attn（set2set_steps 保留以兼容旧调用，已忽略）。"""
    m = mode.lower()
    if m not in ("multi_attn",):
        raise ValueError(f"仅支持 pooling='multi_attn'，收到 {mode!r}")
    return (
        MultiHeadAttnPool(
            hidden_dim,
            pool_heads,
            dropout,
            merge_activation=pool_merge_activation,
            attn_dropout=pool_attn_dropout,
        ),
        hidden_dim,
    )
