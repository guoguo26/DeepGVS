#!/usr/bin/env python3
"""
Graph-Mamba 推理模型（与 graph_mamba_infer.json 一致）：

  - 每层：GAT + BiMamba，parallel_fusion=gated，cross_interaction=none
  - 池化：multi_attn
  - 全局：late ProtT5（use_global_prott5 + global_fusion=concat）
  - 导出 embedding：graph_heads_global（池化多头向量 + ProtT5 全局向量）
"""

from __future__ import annotations

from typing import List, Optional, Tuple, cast

import torch
import torch.nn.functional as F
from torch import nn

from bi_mamba_gcn_encoder import BiMambaBlock
from esm2_feature_extractor import ESM2FeatureExtractor
from graph_structure_layers import build_graph_block, build_pooling


def _require_config(name: str, value: str, allowed: Tuple[str, ...]) -> str:
    v = (value or "").lower().strip()
    if v not in allowed:
        raise ValueError(f"{name} 仅支持 {allowed}，收到 {value!r}")
    return v


class GatedNodeFusion(nn.Module):
    """parallel_fusion=gated：σ(W[g;s])⊙g + (1-σ)⊙s"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Linear(2 * int(hidden_dim), int(hidden_dim))

    def forward(self, g: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([g, s], dim=-1)
        gate = torch.sigmoid(self.gate(cat))
        return gate * g + (1.0 - gate) * s


class ParallelGraphMambaLayer(nn.Module):
    """GAT(x, adj) 与 BiMamba(x) 并联，门控融合（cross_interaction=none）。"""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        gat_heads: int,
        mamba_d_state: int,
    ):
        super().__init__()
        self.graph = build_graph_block("gat", hidden_dim, dropout, gat_heads)
        self.mamba = BiMambaBlock(
            d_model=hidden_dim,
            d_state=mamba_d_state,
            use_simplified=False,
            dropout=dropout,
        )
        self.fuse = GatedNodeFusion(hidden_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x_graph = self.graph(x, adj)
        x_seq = self.mamba(x)
        return self.fuse(x_graph, x_seq)


class SimpleGraphMamba(nn.Module):
    """
    固定配置（与训练 checkpoint 一致）：
      parallel_fusion=gated, cross_interaction=none,
      graph_conv=gat, pooling=multi_attn,
      use_global_prott5=True, global_fusion=concat,
      prott5_branch=False, mamba_global_token_prott5=False
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 2,
        num_mamba_layers: int = 4,
        mamba_d_state: int = 16,
        dropout: float = 0.1,
        parallel_fusion: str = "gated",
        cross_interaction: str = "none",
        use_global_prott5: bool = True,
        mamba_global_token_prott5: bool = False,
        global_prott5_dim: int = 1024,
        global_proj_in_dim: Optional[int] = None,
        graph_conv: str = "gat",
        gat_heads: int = 4,
        sage_agg: str = "mean",
        pooling: str = "multi_attn",
        pool_heads: int = 6,
        set2set_steps: int = 4,
        pool_merge_activation: str = "gelu",
        pool_attn_dropout: float = 0.1,
        global_fusion: str = "concat",
        prott5_branch: bool = False,
        prott5_branch_widths: Tuple[int, int, int] = (512, 256, 128),
    ):
        super().__init__()
        del sage_agg, set2set_steps, prott5_branch_widths

        _require_config("parallel_fusion", parallel_fusion, ("gated",))
        _require_config("cross_interaction", cross_interaction, ("none",))
        _require_config("graph_conv", graph_conv, ("gat",))
        _require_config("pooling", pooling, ("multi_attn",))
        _require_config("global_fusion", global_fusion, ("concat",))

        if bool(prott5_branch):
            raise ValueError("本仓库仅支持 prott5_branch=False")
        if bool(mamba_global_token_prott5):
            raise ValueError("本仓库仅支持 mamba_global_token_prott5=False")
        if not bool(use_global_prott5):
            raise ValueError("推理导出 graph_heads_global 需要 use_global_prott5=True")

        self.hidden_dim = int(hidden_dim)
        self.input_dim = int(input_dim)
        self.mamba_d_state = int(mamba_d_state)

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        n_layers = max(0, int(num_mamba_layers))
        self.parallel_layers = nn.ModuleList(
            [
                ParallelGraphMambaLayer(
                    hidden_dim,
                    dropout,
                    gat_heads,
                    self.mamba_d_state,
                )
                for _ in range(n_layers)
            ]
        )
        pool_mod, _ = build_pooling(
            "multi_attn",
            hidden_dim,
            pool_heads,
            0,
            dropout,
            pool_merge_activation=pool_merge_activation,
            pool_attn_dropout=pool_attn_dropout,
        )
        self.graph_pool = pool_mod

        g_in = int(global_proj_in_dim) if global_proj_in_dim is not None else int(global_prott5_dim)
        self.global_proj = nn.Linear(g_in, hidden_dim)

        # 分类头仍保留，便于 load_state_dict；DeepGVS 推理走 graph_heads_global 嵌入
        cls_in = hidden_dim * 2
        cls_mid2 = max(64, hidden_dim // 2)
        self.classifier = nn.Sequential(
            nn.Linear(cls_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, cls_mid2),
            nn.LayerNorm(cls_mid2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cls_mid2, num_classes),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        adj: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        global_prott5: Optional[torch.Tensor] = None,
        return_embedding: bool = False,
        embedding_kind: str = "graph_heads_global",
    ) -> torch.Tensor:
        B, N, _ = node_features.shape
        if mask is None:
            mask = node_features.new_ones(B, N, dtype=torch.bool)

        x = self.dropout(self.input_proj(node_features))
        for layer in self.parallel_layers:
            x = layer(x, adj)

        if return_embedding:
            ek = (embedding_kind or "graph_heads_global").lower().strip()
            if ek != "graph_heads_global":
                raise ValueError(f"仅支持 embedding_kind='graph_heads_global'，收到 {embedding_kind!r}")
            if global_prott5 is None:
                raise ValueError("graph_heads_global 需要 global_prott5 向量")
            g_pre = self.graph_pool.forward_parts(x, mask)
            gg = self.global_proj(global_prott5)
            return torch.cat([g_pre, gg], dim=-1)

        if global_prott5 is None:
            raise ValueError("分类 forward 需要 global_prott5")
        g_merged = self.graph_pool(x, mask)
        gg = self.global_proj(global_prott5)
        return self.classifier(torch.cat([g_merged, gg], dim=-1))


def _esm_module(extractor: ESM2FeatureExtractor) -> nn.Module:
    m = extractor.model
    if m is None:
        raise RuntimeError("ESM 模型未加载，请先调用 load_model()")
    if not isinstance(m, nn.Module):
        raise TypeError(f"ESM model 期望 nn.Module，得到 {type(m)!r}")
    return cast(nn.Module, m)


def parse_int_tuple3(s: str, name: str) -> Tuple[int, int, int]:
    xs = [int(x.strip()) for x in str(s).split(",") if x.strip()]
    if len(xs) != 3:
        raise ValueError(f"{name} 需要 3 个整数，当前得到: {s}")
    if any(x <= 0 for x in xs):
        raise ValueError(f"{name} 中每个值都必须 > 0，当前得到: {s}")
    return int(xs[0]), int(xs[1]), int(xs[2])
