#!/usr/bin/env python3
"""PDB 解析、kNN（binary）邻接与 ESM 层配置（推理用）。"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def parse_pdb(
    pdb_path: str, extract_plddt: bool = True
) -> Tuple[np.ndarray, List[str], Optional[np.ndarray]]:
    """解析 PDB：Cα 坐标、单字母序列、可选 pLDDT（b_factor）。"""
    ca_coords: list = []
    sequence: list = []
    residue_indices: list = []
    plddt_scores: list = []

    with open(pdb_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("ATOM"):
                atom_name = line[12:16].strip()
                residue_name = line[17:20].strip()
                residue_idx = int(line[22:26].strip())

                if atom_name == "CA":
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())

                    b_factor = 0.0
                    if extract_plddt and len(line) >= 66:
                        try:
                            b_factor = float(line[60:66].strip())
                        except (ValueError, IndexError):
                            b_factor = 0.0

                    if not residue_indices or residue_idx != residue_indices[-1]:
                        residue_indices.append(residue_idx)
                        sequence.append(residue_name)
                        ca_coords.append([x, y, z])
                        if extract_plddt:
                            plddt_scores.append(b_factor)
                    else:
                        ca_coords[-1] = [x, y, z]
                        if extract_plddt:
                            plddt_scores[-1] = b_factor

    if len(ca_coords) == 0:
        raise ValueError(f"PDB文件 {pdb_path} 中未找到Cα原子")

    ca_arr = np.array(ca_coords, dtype=np.float32)

    aa_3to1 = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
        "UNK": "X",
        "XXX": "X",
        "MSE": "M",
        "SEC": "C",
        "PYL": "O",
    }

    sequence_1letter = [aa_3to1.get(aa, "X") for aa in sequence]

    plddt: Optional[np.ndarray] = None
    if extract_plddt:
        if len(plddt_scores) == len(ca_arr):
            plddt = np.array(plddt_scores, dtype=np.float32)
            if plddt.max() <= 1.0:
                plddt = plddt * 100.0
        else:
            logger.warning(
                f"pLDDT数量 ({len(plddt_scores)}) 与残基数 ({len(ca_arr)}) 不匹配，使用默认值"
            )
            plddt = np.full(len(ca_arr), 50.0, dtype=np.float32)

    logger.debug(f"从 {pdb_path} 提取了 {len(ca_arr)} 个残基的Cα坐标")
    if plddt is not None:
        logger.debug(
            f"pLDDT范围: [{plddt.min():.1f}, {plddt.max():.1f}], 均值: {plddt.mean():.1f}"
        )

    return ca_arr, sequence_1letter, plddt


def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """欧氏距离矩阵 (N, N)。"""
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1)).astype(np.float32)


def build_knn_adj(dist: np.ndarray, k: int) -> np.ndarray:
    """无向 kNN 图，边权为 1（与 graph_mamba_infer: edge_weight=binary 一致）。"""
    L = int(dist.shape[0])
    if L <= 1:
        return np.zeros((L, L), dtype=np.float32)
    k = min(int(k), max(1, L - 1))
    adj = np.zeros((L, L), dtype=np.float32)
    for i in range(L):
        d_i = dist[i].copy()
        d_i[i] = np.inf
        neigh_idx = np.argpartition(d_i, k)[:k]
        neigh_idx = neigh_idx[np.argsort(d_i[neigh_idx], kind="mergesort")]
        adj[i, neigh_idx] = 1.0
    adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 0.0)
    return adj


def parse_esm_repr_layers(s: str) -> List[int]:
    t = (s or "").strip()
    if not t:
        return [33]
    out = [int(x.strip()) for x in t.split(",") if x.strip()]
    return out if out else [33]


def canonical_esm_layers(layers: List[int]) -> List[int]:
    return sorted({int(x) for x in layers})


def node_esm_cache_path(
    cache_dir: Path,
    stem: str,
    max_len: Optional[int],
    layers: List[int],
    *,
    node_full_sequence: bool = False,
    node_encode_max_len: int = 0,
) -> Path:
    uniq = canonical_esm_layers(layers)
    mid: List[str] = []
    if node_full_sequence:
        if node_encode_max_len > 0:
            mid.append(f"encml{node_encode_max_len}")
        mid.append("fullseq")
    if uniq != [33]:
        mid.append("L" + "-".join(str(x) for x in uniq))
    if max_len is not None and max_len > 0:
        mid.append(f"trunc_max{max_len}")
    if not mid:
        return cache_dir / f"{stem}.npy"
    return cache_dir / f"{stem}__{'_'.join(mid)}.npy"


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python pdb_graph.py <pdb_file> [k]")
        sys.exit(1)

    pdb_path = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    ca, seq, _ = parse_pdb(pdb_path, extract_plddt=False)
    d = compute_distance_matrix(ca)
    adj = build_knn_adj(d, k)
    print(f"序列长度: {len(seq)}")
    print(f"邻接非零边: {int((adj > 0).sum() // 2)}")
