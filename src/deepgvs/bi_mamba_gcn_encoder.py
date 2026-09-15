#!/usr/bin/env python3
"""
Bi-Mamba 块：供图训练主干并联序列分支使用（双向 Mamba 或双向 LSTM 回退）。
"""

import sys
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    from mamba_ssm import Mamba

    MAMBA_AVAILABLE = True
except ImportError as e:
    Mamba = None  # type: ignore[misc, assignment]
    MAMBA_AVAILABLE = False
    logger.warning(
        "mamba_ssm 无法导入（%s），将使用双向 LSTM 简化实现。"
        "当前解释器: %s。若已在某 conda 环境安装 mamba-ssm，请用该环境的 python 启动训练（nohup 前需 activate 或 PYTHON=/path/to/env/bin/python）。",
        e,
        sys.executable,
    )


class BiMambaBlock(nn.Module):
    """
    双向 Mamba 块：前向 + 后向序列建模，残差与 LayerNorm。
    mamba_ssm 不可用时使用双向 LSTM 近似。
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_simplified: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_simplified = use_simplified if use_simplified else (not MAMBA_AVAILABLE)

        if not self.use_simplified and MAMBA_AVAILABLE:
            assert Mamba is not None
            self.forward_mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.backward_mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            logger.warning("使用简化实现（双向LSTM）替代Mamba")
            self.forward_mamba = nn.LSTM(
                input_size=d_model,
                hidden_size=d_model // 2,
                num_layers=2,
                batch_first=True,
                bidirectional=False,
            )
            self.backward_mamba = nn.LSTM(
                input_size=d_model,
                hidden_size=d_model // 2,
                num_layers=2,
                batch_first=True,
                bidirectional=False,
            )
            self.forward_proj = nn.Linear(d_model // 2, d_model)
            self.backward_proj = nn.Linear(d_model // 2, d_model)

        self.linear_proj = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.layer_norm(x)

        if not self.use_simplified and MAMBA_AVAILABLE:
            forward_out = self.forward_mamba(x_norm)
        else:
            forward_out, _ = self.forward_mamba(x_norm)
            forward_out = self.forward_proj(forward_out)

        x_reversed = torch.flip(x_norm, dims=[1])
        if not self.use_simplified and MAMBA_AVAILABLE:
            backward_out = self.backward_mamba(x_reversed)
            backward_out = torch.flip(backward_out, dims=[1])
        else:
            backward_out, _ = self.backward_mamba(x_reversed)
            backward_out = torch.flip(backward_out, dims=[1])
            backward_out = self.backward_proj(backward_out)

        combined = forward_out + backward_out
        output = self.linear_proj(combined)
        output = self.activation(output)
        output = self.dropout(output)
        return output + residual
