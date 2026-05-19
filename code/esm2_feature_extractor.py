#!/usr/bin/env python3
"""
使用ESM-2模型提取蛋白质序列的残基特征
提取每个残基的上下文感知特征向量（Contextualized Residue Embedding）
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

try:
    import esm
    ESM_AVAILABLE = True
    USE_TRANSFORMERS = False
except ImportError:
    try:
        # 尝试使用transformers库中的ESM模型
        from transformers import EsmModel, AutoTokenizer
        ESM_AVAILABLE = True
        USE_TRANSFORMERS = True
    except ImportError:
        ESM_AVAILABLE = False
        USE_TRANSFORMERS = False
        logger.warning("ESM包未安装，请安装: pip install fair-esm 或 pip install transformers")


def configure_esm_partial_finetune_fair_esm(
    esm_model: "torch.nn.Module",
    last_n_transformer_layers: int,
) -> int:
    """
    冻结 ESM-2（fair-esm ProteinBertModel）全部参数，再仅解冻最后 N 个 Transformer 块。
    嵌入层与前面块保持冻结，适合任务微调 + 小 lr。
    返回 requires_grad=True 的参数个数。
    """
    for p in esm_model.parameters():
        p.requires_grad = False
    if not hasattr(esm_model, "layers"):
        raise ValueError(
            "当前 ESM 无 .layers 属性，仅支持 fair-esm；transformers 版 ESM 请关闭 --esm_finetune"
        )
    layers = esm_model.layers
    n = len(layers)
    last_n = max(0, min(int(last_n_transformer_layers), n))
    for i in range(n - last_n, n):
        for p in layers[i].parameters():
            p.requires_grad = True
    trainable = sum(p.numel() for p in esm_model.parameters() if p.requires_grad)
    logger.info(
        "ESM 部分解冻: 最后 %d/%d 个 Transformer 层可训，可训练参数约 %d",
        last_n,
        n,
        trainable,
    )
    return trainable


class ESM2FeatureExtractor:
    """ESM-2特征提取器"""
    
    def __init__(self, 
                 model_name: str = "esm2_t33_650M_UR50D",
                 device: Optional[torch.device] = None,
                 offline_mode: bool = True):
        """
        初始化ESM-2特征提取器
        
        Args:
            model_name: ESM-2模型名称
                      可选: "esm2_t33_650M_UR50D", "esm2_t36_3B_UR50D", "esm2_t48_15B_UR50D"
            device: 计算设备，None则自动选择
            offline_mode: 是否使用离线模式
        """
        if not ESM_AVAILABLE:
            raise ImportError("ESM包未安装，请安装: pip install fair-esm")
        
        self.model_name = model_name
        self.device = device if device is not None else self._get_device()
        self.offline_mode = offline_mode
        self.model: Optional[torch.nn.Module] = None
        self.alphabet = None
        self.batch_converter = None
        self.tokenizer = None
        self.use_transformers = USE_TRANSFORMERS
        self._model_loaded = False
        
        logger.info(f"初始化ESM-2特征提取器，模型: {model_name}, 设备: {self.device}, 使用transformers: {self.use_transformers}")
    
    def _get_device(self) -> torch.device:
        """自动选择计算设备"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    
    def load_model(self):
        """加载ESM-2模型"""
        if self._model_loaded:
            return
        
        try:
            logger.info(f"正在加载ESM-2模型: {self.model_name}")
            
            if self.use_transformers:
                # 使用transformers库
                from transformers import EsmModel, AutoTokenizer
                
                # 映射模型名称
                model_map = {
                    "esm2_t33_650M_UR50D": "facebook/esm2_t33_650M_UR50D",
                    "esm2_t36_3B_UR50D": "facebook/esm2_t36_3B_UR50D",
                    "esm2_t48_15B_UR50D": "facebook/esm2_t48_15B_UR50D"
                }
                hf_model_name = model_map.get(self.model_name, f"facebook/{self.model_name}")
                
                self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
                self.model = EsmModel.from_pretrained(hf_model_name)
                self.model = self.model.to(self.device)
                self.model.eval()
            else:
                # 使用fair-esm库（勿写成 self.model, alphabet = …，否则类型检查器会把 model 标成 tuple）
                esm_core, self.alphabet = esm.pretrained.load_model_and_alphabet(
                    self.model_name
                )
                self.model = esm_core.to(self.device)
                self.model.eval()
                
                # 创建批处理转换器
                self.batch_converter = self.alphabet.get_batch_converter()
            
            self._model_loaded = True
            logger.info(f"✅ ESM-2模型加载成功，设备: {self.device}")
            
        except Exception as e:
            logger.error(f"❌ ESM-2模型加载失败: {e}")
            raise
    
    def extract_features(self, 
                        sequence: str,
                        representation_layer: int = 33) -> np.ndarray:
        """
        提取蛋白质序列的残基特征
        
        Args:
            sequence: 蛋白质序列（单字母代码）
            representation_layer: 使用的表示层，默认33（最后一层）
            
        Returns:
            features: 残基特征矩阵，形状为 (L, D)，L为序列长度，D为特征维度
        """
        if not self._model_loaded:
            self.load_model()
        
        # 提取特征
        with torch.no_grad():
            if self.use_transformers:
                # 使用transformers库
                inputs = self.tokenizer(sequence, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs, output_hidden_states=True)
                # 获取指定层的隐藏状态
                hidden_states = outputs.hidden_states[representation_layer]  # (batch_size, seq_len, hidden_dim)
                
                # 移除特殊token（<cls>和<eos>）
                residue_features = hidden_states[0, 1:-1, :].cpu().numpy()
            else:
                # 使用fair-esm库
                data = [("protein", sequence)]
                batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
                batch_tokens = batch_tokens.to(self.device)
                
                results = self.model(batch_tokens, repr_layers=[representation_layer])
                token_representations = results["representations"][representation_layer]
                
                # 移除第一个和最后一个token（<cls>和<eos>）
                residue_features = token_representations[0, 1:-1, :].cpu().numpy()
        
        logger.debug(f"提取特征: 序列长度={len(sequence)}, 特征维度={residue_features.shape[1]}")
        
        return residue_features

    def extract_features_multi_layer(
        self,
        sequence: str,
        representation_layers: List[int],
    ) -> np.ndarray:
        """
        多隐藏层拼接（沿特征维）：形状 (L, D * n_layers)，用于多尺度表征。
        representation_layers 为层号列表，如 [18, 33]（fair-esm 与旧脚本层号一致）。
        """
        if not self._model_loaded:
            self.load_model()
        layers = sorted({int(x) for x in representation_layers})
        if not layers:
            raise ValueError("representation_layers 不能为空")

        with torch.no_grad():
            if self.use_transformers:
                inputs = self.tokenizer(sequence, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs, output_hidden_states=True)
                hs = outputs.hidden_states
                parts = []
                for li in layers:
                    if li < 0 or li >= len(hs):
                        raise ValueError(
                            f"层号 {li} 越界：hidden_states 长度为 {len(hs)}"
                        )
                    parts.append(hs[li][0, 1:-1, :])
                cat = torch.cat(parts, dim=-1)
                residue_features = cat.cpu().numpy()
            else:
                data = [("protein", sequence)]
                batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
                batch_tokens = batch_tokens.to(self.device)
                results = self.model(batch_tokens, repr_layers=layers)
                reps = results["representations"]
                parts = [reps[li][0, 1:-1, :] for li in layers]
                cat = torch.cat(parts, dim=-1)
                residue_features = cat.cpu().numpy()

        logger.debug(
            f"多层提取: L={len(sequence)}, layers={layers}, dim={residue_features.shape[1]}"
        )
        return residue_features

    def forward_residue_features_batch(
        self,
        sequences: List[str],
        representation_layers: List[int],
        train_mode: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        fair-esm 批量前向（可带梯度），用于 Graph Mamba 训练循环内在线算残基特征。

        Returns:
            feats: [B, L_max, D]，与多层 extract_features_multi_layer 一致沿最后一维拼接
            mask: [B, L_max] bool，True 为真实残基位置
        """
        if self.use_transformers:
            raise NotImplementedError(
                "forward_residue_features_batch 当前仅支持 fair-esm；"
                "请使用 fair-esm 或关闭 --esm_finetune"
            )
        if not self._model_loaded:
            self.load_model()
        assert self.batch_converter is not None and self.model is not None

        layers = sorted({int(x) for x in representation_layers})
        if not layers:
            raise ValueError("representation_layers 不能为空")

        batch_data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
        _, _, batch_tokens = self.batch_converter(batch_data)
        batch_tokens = batch_tokens.to(self.device)

        if train_mode:
            self.model.train()
        else:
            self.model.eval()

        ctx = torch.enable_grad() if train_mode else torch.no_grad()
        with ctx:
            results = self.model(batch_tokens, repr_layers=layers)
            reps = results["representations"]

        B = len(sequences)
        max_L = max((len(s) for s in sequences), default=0)
        d_each = reps[layers[0]].shape[-1]
        D = d_each * len(layers)
        out = torch.zeros(
            B,
            max_L,
            D,
            device=self.device,
            dtype=reps[layers[0]].dtype,
        )
        mask = torch.zeros(B, max_L, dtype=torch.bool, device=self.device)
        for j, seq in enumerate(sequences):
            L = len(seq)
            if L == 0:
                continue
            parts = []
            for li in layers:
                tok = reps[li][j, 1 : 1 + L, :]
                parts.append(tok)
            cat = torch.cat(parts, dim=-1)
            out[j, :L] = cat
            mask[j, :L] = True
        return out, mask


if __name__ == "__main__":
    # 测试代码
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python esm2_feature_extractor.py <sequence>")
        print("示例: python esm2_feature_extractor.py MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL")
        sys.exit(1)
    
    sequence = sys.argv[1]
    
    try:
        extractor = ESM2FeatureExtractor()
        extractor.load_model()
        
        features = extractor.extract_features(sequence)
        print(f"序列长度: {len(sequence)}")
        print(f"特征形状: {features.shape}")
        print(f"特征维度: {features.shape[1]}")
        print(f"特征统计: mean={features.mean():.4f}, std={features.std():.4f}")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

