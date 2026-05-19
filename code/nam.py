#!/usr/bin/env python3
"""NAM ensemble 与单流特征预处理。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.linear_model import LogisticRegression


def split_protein_dna_feature_cols(feature_cols: Sequence[str]) -> tuple[list[str], list[str]]:
    protein_cols = [
        c
        for c in feature_cols
        if c.startswith("graph_mamba_feat_") or c.startswith("gm_") or c.startswith("graph_")
    ]
    dna_cols = [c for c in feature_cols if c not in protein_cols]
    return protein_cols, dna_cols


class FeatureNN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(max(0, num_layers - 1)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class NeuralAdditiveModel(nn.Module):
    def __init__(self, num_features: int, num_classes: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.feature_nns = nn.ModuleList(
            [
                FeatureNN(1, num_classes, hidden_dim, max(num_layers, 1), dropout)
                for _ in range(num_features)
            ]
        )
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        contributions = [nn_(x[:, i : i + 1]) for i, nn_ in enumerate(self.feature_nns)]
        logits = torch.stack(contributions, dim=0).sum(dim=0) + self.bias
        return logits


def safe_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def proba_to_meta_block(probas: np.ndarray, use_logit: bool) -> np.ndarray:
    probas = probas.astype(np.float32)
    if probas.shape[1] == 2:
        p1 = probas[:, 1:2]
        if use_logit:
            return safe_logit(p1).astype(np.float32)
        return p1.astype(np.float32)
    return probas


@dataclass
class SingleStreamPreprocessConfig:
    ablate_modality: str = "none"
    use_dna_feature_whitelist: bool = False
    dna_whitelist_prefixes: str = "k3_,k4_,disc4_,local_,gc_,at_,n_frac,nt_shannon_entropy,orf_"
    feature_select_var_threshold: float = 0.0
    feature_select_method: str = "mi"
    feature_select_k: int = 700
    graph_pca_var: float = 0.95
    random_seed: int = 42


class FittedSingleStreamPreprocessor:
    def __init__(self) -> None:
        self.initial_feature_cols: Optional[List[str]] = None
        self._ablate_keep_idx: Optional[np.ndarray] = None
        self._whitelist_keep_idx: Optional[np.ndarray] = None
        self._vt: Optional[VarianceThreshold] = None
        self._mi_idx: Optional[np.ndarray] = None
        self._pca: Optional[PCA] = None
        self._protein_idx_after_mi: Optional[List[int]] = None
        self._non_protein_idx_after_mi: Optional[List[int]] = None
        self.final_dim_: int = 0
        self.config_: Optional[SingleStreamPreprocessConfig] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_cols: Sequence[str],
        cfg: SingleStreamPreprocessConfig,
        split_protein_dna_feature_cols_fn=split_protein_dna_feature_cols,
    ) -> "FittedSingleStreamPreprocessor":
        self.config_ = cfg
        self.initial_feature_cols = list(feature_cols)
        Xtr = X_train.astype(np.float32, copy=False)
        current_cols = list(feature_cols)

        protein_cols, dna_cols = split_protein_dna_feature_cols_fn(current_cols)
        protein_set = set(protein_cols)
        dna_set = set(dna_cols)

        if cfg.ablate_modality != "none":
            if cfg.ablate_modality == "dna_only":
                keep_cols = [c for c in current_cols if c in dna_set]
            else:
                keep_cols = [c for c in current_cols if c in protein_set]
            if not keep_cols:
                raise ValueError(f"Ablation {cfg.ablate_modality!r} removed all features.")
            self._ablate_keep_idx = np.array([current_cols.index(c) for c in keep_cols], dtype=int)
            Xtr = Xtr[:, self._ablate_keep_idx]
            current_cols = keep_cols
        else:
            self._ablate_keep_idx = None

        if cfg.use_dna_feature_whitelist:
            dna_prefixes = [p.strip() for p in cfg.dna_whitelist_prefixes.split(",") if p.strip()]
            keep_cols = []
            for c in current_cols:
                if c in protein_set:
                    keep_cols.append(c)
                    continue
                if c in dna_set and any(c.startswith(pref) for pref in dna_prefixes):
                    keep_cols.append(c)
            if not keep_cols:
                raise ValueError("DNA whitelist removed all columns.")
            self._whitelist_keep_idx = np.array([current_cols.index(c) for c in keep_cols], dtype=int)
            Xtr = Xtr[:, self._whitelist_keep_idx]
            current_cols = keep_cols
        else:
            self._whitelist_keep_idx = None

        if cfg.feature_select_var_threshold > 0.0:
            self._vt = VarianceThreshold(threshold=float(cfg.feature_select_var_threshold))
            Xtr = self._vt.fit_transform(Xtr).astype(np.float32)
            mask = self._vt.get_support()
            if mask is None:
                raise RuntimeError("VarianceThreshold.get_support() returned None")
            support = np.asarray(mask, dtype=bool)
            current_cols = [c for c, k in zip(current_cols, support) if bool(k)]
        else:
            self._vt = None

        if cfg.feature_select_method != "none" and cfg.feature_select_k > 0 and Xtr.shape[1] > cfg.feature_select_k:
            k = int(cfg.feature_select_k)
            if cfg.feature_select_method == "mi":
                mi = mutual_info_classif(Xtr, y_train, random_state=cfg.random_seed)
                idx = np.argsort(mi)[::-1][:k]
                self._mi_idx = idx
                Xtr = Xtr[:, idx]
                current_cols = [current_cols[i] for i in idx]
            elif cfg.feature_select_method == "l1":
                selector_lr = LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    C=0.1,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=cfg.random_seed,
                )
                selector_lr.fit(Xtr, y_train)
                coef = np.abs(selector_lr.coef_).sum(axis=0)
                idx = np.argsort(coef)[::-1][:k]
                self._mi_idx = idx
                Xtr = Xtr[:, idx]
                current_cols = [current_cols[i] for i in idx]
            else:
                raise ValueError(f"Unknown feature_select_method: {cfg.feature_select_method!r}")
        else:
            self._mi_idx = None

        if 0.0 < cfg.graph_pca_var < 1.0:
            protein_cols_now, _ = split_protein_dna_feature_cols_fn(current_cols)
            protein_idx = [i for i, c in enumerate(current_cols) if c in set(protein_cols_now)]
            non_protein_idx = [i for i in range(len(current_cols)) if i not in set(protein_idx)]
            if protein_idx:
                self._pca = PCA(n_components=float(cfg.graph_pca_var), svd_solver="full")
                self._pca.fit(Xtr[:, protein_idx])
                self._protein_idx_after_mi = protein_idx
                self._non_protein_idx_after_mi = non_protein_idx
                protein_tr = self._pca.transform(Xtr[:, protein_idx]).astype(np.float32)
                non_protein_tr = (
                    Xtr[:, non_protein_idx] if non_protein_idx else np.zeros((Xtr.shape[0], 0), dtype=np.float32)
                )
                Xtr = np.hstack([non_protein_tr, protein_tr]).astype(np.float32)
            else:
                self._pca = None
                self._protein_idx_after_mi = None
                self._non_protein_idx_after_mi = None
        else:
            self._pca = None
            self._protein_idx_after_mi = None
            self._non_protein_idx_after_mi = None

        self.final_dim_ = int(Xtr.shape[1])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.initial_feature_cols is None:
            raise RuntimeError("Preprocessor is not fitted.")
        if X.shape[1] != len(self.initial_feature_cols):
            raise ValueError(
                f"Expected {len(self.initial_feature_cols)} raw features, got {X.shape[1]}."
            )
        Xtr = X.astype(np.float32, copy=False)
        if self._ablate_keep_idx is not None:
            Xtr = Xtr[:, self._ablate_keep_idx]
        if self._whitelist_keep_idx is not None:
            Xtr = Xtr[:, self._whitelist_keep_idx]
        if self._vt is not None:
            vt_out = self._vt.transform(Xtr)
            Xtr = np.asarray(vt_out, dtype=np.float32)
        if self._mi_idx is not None:
            Xtr = Xtr[:, self._mi_idx]
        if self._pca is not None and self._protein_idx_after_mi is not None and self._non_protein_idx_after_mi is not None:
            p_idx = self._protein_idx_after_mi
            np_idx = self._non_protein_idx_after_mi
            protein_te = self._pca.transform(Xtr[:, p_idx]).astype(np.float32)
            non_protein_te = Xtr[:, np_idx] if np_idx else np.zeros((Xtr.shape[0], 0), dtype=np.float32)
            Xtr = np.hstack([non_protein_te, protein_te]).astype(np.float32)
        if int(Xtr.shape[1]) != self.final_dim_:
            raise RuntimeError(f"Internal error: final dim {Xtr.shape[1]} != {self.final_dim_}")
        return Xtr
