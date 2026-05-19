#!/usr/bin/env python3
"""Torch ``Dataset``, batch collation, coordinate augment, edge dropout."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from esm2_feature_extractor import ESM2FeatureExtractor
from pdb_graph import (
    build_knn_adj,
    canonical_esm_layers,
    compute_distance_matrix,
    node_esm_cache_path,
    parse_pdb,
)

class SimpleGraphMambaDataset(Dataset):
    """
    基于 PDB 的极简图数据集（截断对齐）：
      - 序列由 PDB 解析；节点特征仅为 **ESM**
      - kNN（binary）图；可选全局 **ProtT5** `{stem}.npy`
    """

    def __init__(
        self,
        records: List[Dict],
        pdb_dir: Path,
        esm_extractor: ESM2FeatureExtractor,
        k: int = 30,
        cache_dir: Optional[Path] = None,
        max_len: Optional[int] = None,
        use_global_prott5: bool = False,
        global_prott5_dir: Optional[Path] = None,
        global_prott5_dim: int = 1024,
        coord_augment: bool = False,
        ca_noise_std: float = 0.0,
        coord_rotate_augment: bool = False,
        coord_rotate_max_deg: float = 15.0,
        coord_translate_std: float = 0.0,
        esm_repr_layers: Optional[List[int]] = None,
        esm_node_full_sequence: bool = False,
        esm_node_encode_max_len: int = 0,
        esm_finetune: bool = False,
        feat_dim: int = 1280,
    ):
        self.records = records
        self.pdb_dir = pdb_dir
        self.esm_extractor = esm_extractor
        self.k = k
        self.cache_dir = cache_dir
        self.max_len = max_len
        self.use_global_prott5 = bool(use_global_prott5)
        self.global_prott5_dir = global_prott5_dir
        self.global_prott5_dim = int(global_prott5_dim)
        self.coord_augment = coord_augment
        self.ca_noise_std = float(ca_noise_std)
        self.coord_rotate_augment = bool(coord_rotate_augment)
        self.coord_rotate_max_deg = float(coord_rotate_max_deg)
        self.coord_translate_std = float(coord_translate_std)
        self.esm_repr_layers = (
            canonical_esm_layers(esm_repr_layers)
            if esm_repr_layers is not None
            else [33]
        )
        self.esm_node_full_sequence = bool(esm_node_full_sequence)
        self.esm_node_encode_max_len = max(0, int(esm_node_encode_max_len))
        self.esm_finetune = bool(esm_finetune)
        self._feat_dim = int(feat_dim)

    def __len__(self) -> int:
        return len(self.records)

    def _extract_esm_residue(self, sequence: str) -> np.ndarray:
        Ls = self.esm_repr_layers
        if len(Ls) == 1:
            return self.esm_extractor.extract_features(
                sequence, representation_layer=int(Ls[0])
            )
        return self.esm_extractor.extract_features_multi_layer(sequence, Ls)

    def _load_esm_node_features(
        self,
        stem: str,
        L: int,
        sequence: str,
        full_sequence: str,
    ) -> np.ndarray:
        """
        节点 ESM：finetune 时用零占位；否则优先读缓存，缺失则在线提取并写回缓存（若配置了 cache_dir）。
        """
        cache_hit = False
        cache_path: Optional[Path] = None
        esm_feats = np.zeros((L, self._feat_dim), dtype=np.float32)

        if self.esm_finetune:
            cache_hit = True
        elif self.cache_dir is not None:
            cache_path = node_esm_cache_path(
                self.cache_dir,
                stem,
                self.max_len,
                self.esm_repr_layers,
                node_full_sequence=self.esm_node_full_sequence,
                node_encode_max_len=self.esm_node_encode_max_len,
            )
            if cache_path.is_file():
                esm_feats = np.load(cache_path)
                cache_hit = True

        if not self.esm_finetune and not cache_hit:
            if self.esm_node_full_sequence:
                enc_seq = full_sequence
                if self.esm_node_encode_max_len > 0:
                    enc_seq = full_sequence[: self.esm_node_encode_max_len]
                esm_feats = self._extract_esm_residue(enc_seq)
            else:
                esm_feats = self._extract_esm_residue(sequence)

        if (not self.esm_finetune) and (not cache_hit) and cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, esm_feats)

        return esm_feats

    def __getitem__(self, idx: int) -> Dict:
        item = self.records[idx]
        stem = item["sequence_id"]
        label = int(item["label"])

        pdb_path_raw = item.get("pdb_path")
        if pdb_path_raw:
            pdb_path = Path(str(pdb_path_raw))
        else:
            pdb_path = self.pdb_dir / f"{stem}.pdb"
        if not pdb_path.is_file():
            raise FileNotFoundError(f"PDB 不存在: {pdb_path}")

        ca_coords, seq_list, _ = parse_pdb(str(pdb_path), extract_plddt=False)
        L_graph = len(ca_coords)
        if L_graph == 0:
            raise ValueError(f"空结构: {pdb_path}")

        full_sequence = "".join(seq_list)
        if len(full_sequence) == 0:
            raise ValueError(f"空序列: {pdb_path}")

        global_prott5_vec: Optional[torch.Tensor] = None
        if self.use_global_prott5:
            if self.global_prott5_dir is None:
                raise ValueError("use_global_prott5=True 但未提供 global_prott5_dir")
            p5_path = self.global_prott5_dir / f"{stem}.npy"
            if p5_path.is_file():
                p5 = np.load(p5_path).astype(np.float32).ravel()
            else:
                p5 = np.zeros((self.global_prott5_dim,), dtype=np.float32)
            if p5.shape[0] != self.global_prott5_dim:
                raise ValueError(
                    f"ProtT5 维度不符 stem={stem}: 得到 {p5.shape[0]}，期望 {self.global_prott5_dim}"
                )
            global_prott5_vec = torch.from_numpy(p5.copy()).float()

        # 与结构长度对齐：取 PDB 序列前 L_graph
        sequence = full_sequence[:L_graph]
        L = len(sequence)

        if self.max_len is not None and self.max_len > 0:
            L = min(L, self.max_len)
            sequence = sequence[:L]
            ca_coords = ca_coords[:L]

        ca_coords = np.asarray(ca_coords, dtype=np.float32)
        if self.coord_augment:
            if self.ca_noise_std > 0:
                ca_coords = ca_coords + np.random.randn(*ca_coords.shape).astype(
                    np.float32
                ) * np.float32(self.ca_noise_std)
            if self.coord_rotate_augment:
                ca_coords = rodrigues_rotate_coords(
                    ca_coords, float(self.coord_rotate_max_deg)
                )
            if self.coord_translate_std > 0:
                ca_coords = ca_coords + np.random.randn(1, 3).astype(np.float32) * np.float32(
                    self.coord_translate_std
                )

        esm_feats = self._load_esm_node_features(stem, L, sequence, full_sequence)

        if esm_feats.shape[0] != L:
            if esm_feats.shape[0] > L:
                esm_feats = esm_feats[:L]
            else:
                pad = np.zeros(
                    (L - esm_feats.shape[0], esm_feats.shape[1]),
                    dtype=esm_feats.dtype,
                )
                esm_feats = np.concatenate([esm_feats, pad], axis=0)

        L_graph_eff = int(ca_coords.shape[0])
        dist = compute_distance_matrix(ca_coords)
        k_eff = min(self.k, max(1, L_graph_eff - 1))
        knn_adj = build_knn_adj(dist, k_eff)

        out: Dict = {
            "sequence_id": stem,
            "label": label,
            "node_features": torch.from_numpy(esm_feats).float(),
            "adj": torch.from_numpy(knn_adj).float(),
        }
        if self.esm_finetune:
            out["esm_sequence"] = sequence
        if global_prott5_vec is not None:
            out["global_prott5"] = global_prott5_vec
        return out


def collate_graph_mamba_batch(batch: List[Dict]) -> Dict:
    B = len(batch)
    # 对于个别样本，node_features 长度和 adj 尺寸可能不完全一致；
    # 这里对每个样本按二者的较小值截断，避免维度不匹配。
    def _eff_len(item: Dict) -> int:
        Lm = min(item["node_features"].shape[0], item["adj"].shape[0])
        ab = item.get("adj_b")
        if ab is not None:
            Lm = min(Lm, ab.shape[0])
        am = item.get("adj_multi")
        if am is not None:
            Lm = min(Lm, am.shape[-1])
        return Lm

    per_lengths = [_eff_len(item) for item in batch]
    max_len = max(per_lengths)
    feat_dim = batch[0]["node_features"].shape[1]

    node_feats = torch.zeros(B, max_len, feat_dim, dtype=torch.float32)
    adjs = torch.zeros(B, max_len, max_len, dtype=torch.float32)
    masks = torch.zeros(B, max_len, dtype=torch.bool)
    labels = torch.zeros(B, dtype=torch.long)
    seq_ids: List[str] = []
    has_adj_b = batch[0].get("adj_b") is not None
    adjs_b: Optional[torch.Tensor] = None
    if has_adj_b:
        adjs_b = torch.zeros(B, max_len, max_len, dtype=torch.float32)
    has_adj_multi = batch[0].get("adj_multi") is not None
    adjs_multi: Optional[torch.Tensor] = None
    if has_adj_multi:
        num_scales = int(batch[0]["adj_multi"].shape[0])
        adjs_multi = torch.zeros(B, num_scales, max_len, max_len, dtype=torch.float32)

    for i, item in enumerate(batch):
        L = _eff_len(item)
        node_feats[i, :L] = item["node_features"][:L]
        adjs[i, :L, :L] = item["adj"][:L, :L]
        if adjs_b is not None:
            adj_b = item.get("adj_b")
            if adj_b is None:
                raise ValueError("collate: batch 中 adj_b 缺失（与首样本不一致）")
            adjs_b[i, :L, :L] = adj_b[:L, :L]
        if adjs_multi is not None:
            adj_m = item.get("adj_multi")
            if adj_m is None:
                raise ValueError("collate: batch 中 adj_multi 缺失（与首样本不一致）")
            adjs_multi[i, :, :L, :L] = adj_m[:, :L, :L]
        masks[i, :L] = True
        labels[i] = item["label"]
        seq_ids.append(item["sequence_id"])

    out: Dict = {
        "sequence_id": seq_ids,
        "node_features": node_feats,
        "adj": adjs,
        "mask": masks,
        "labels": labels,
    }
    if adjs_b is not None:
        out["adj_b"] = adjs_b
    if adjs_multi is not None:
        out["adj_multi"] = adjs_multi
    if batch[0].get("global_prott5") is not None:
        out["global_prott5"] = torch.stack([it["global_prott5"] for it in batch], dim=0)
    if batch[0].get("esm_sequence") is not None:
        out["esm_sequences"] = [it["esm_sequence"] for it in batch]
    return out


def rodrigues_rotate_coords(ca_coords: np.ndarray, max_deg: float) -> np.ndarray:
    """对 N×3 Cα 坐标施加随机轴角旋转（度）。"""
    if max_deg <= 0:
        return ca_coords
    axis = np.random.randn(3).astype(np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    angle = float(np.random.uniform(-max_deg, max_deg)) * np.pi / 180.0
    kx, ky, kz = float(axis[0]), float(axis[1]), float(axis[2])
    K = np.array(
        [[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]],
        dtype=np.float64,
    )
    R = (
        np.eye(3, dtype=np.float64)
        + np.sin(angle) * K
        + (1.0 - np.cos(angle)) * (K @ K)
    ).astype(np.float32)
    return ca_coords @ R.T


def apply_edge_dropout(adj: torch.Tensor, p: float) -> torch.Tensor:
    """训练时随机丢弃无向边（保留自环）。"""
    if p <= 0 or p >= 1:
        return adj
    B, N, _ = adj.shape
    drop = torch.rand_like(adj) > p
    drop = drop & drop.transpose(1, 2)
    eye = torch.eye(N, device=adj.device, dtype=torch.bool).unsqueeze(0)
    keep = drop | eye
    return adj * keep.to(adj.dtype)


def apply_edge_dropout_multi(adjs: torch.Tensor, p: float) -> torch.Tensor:
    """
    支持对多尺度邻接做 edge dropout。
    - adjs: [B, N, N] 或 [B, S, N, N]
    """
    if p <= 0 or p >= 1:
        return adjs
    if adjs.dim() == 3:
        return apply_edge_dropout(adjs, p)
    if adjs.dim() == 4:
        B, S, N, _ = adjs.shape
        flat = adjs.reshape(B * S, N, N)
        out = apply_edge_dropout(flat, p)
        return out.reshape(B, S, N, N)
    raise ValueError(f"apply_edge_dropout_multi: 期望 3D/4D 张量，得到 dim={adjs.dim()}")

