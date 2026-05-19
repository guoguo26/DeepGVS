#!/usr/bin/env python3
"""
蛋白质结构特征（Graph-Mamba 嵌入）。

- 默认：从预计算 CSV 查表（``graph_mamba_feat_*``）
- 在线：从 PDB 构图 + 加载训练 checkpoint 前向（实现见同目录 ``graph_mamba_*`` / ``esm2_feature_extractor``）
- FASTA：在 ``pdb_root`` / ``pdb_dir`` 下按 ``sequence_id`` 解析已有 PDB；无结构则报错并提示先用 ESMFold 生成

推荐在作者本机 **conda 环境 ``VFS``** 下运行（含 ``mamba-ssm``、``fair-esm``、``torch 2.1+cu118``）::

  ./run_vfs.sh code/prediction.py -m model/ -pseq ... -cds ... --protein-mode compute
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from nam import split_protein_dna_feature_cols


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def project_root() -> Path:
    return repo_root().parent


def load_feature_schema(model_dir: Path | None = None) -> dict:
    model_dir = model_dir or repo_root() / "model"
    schema_path = Path(model_dir) / "feature_schema.json"
    if not schema_path.is_file():
        raise FileNotFoundError(
            f"Missing {schema_path}. Use -m model/dataset_a/ (or copy feature_schema.json into the model dir)."
        )
    with schema_path.open(encoding="utf-8") as f:
        return json.load(f)


def protein_feature_columns(schema: dict) -> List[str]:
    return list(schema["protein_columns"])


def load_protein_features_table(path: Path) -> pd.DataFrame:
    """预计算 Graph-Mamba 特征表（须含 sequence_id 与 graph_mamba_feat_*）。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def lookup_protein_features(
    sequence_ids: Sequence[str],
    table: pd.DataFrame,
    protein_cols: Sequence[str],
) -> np.ndarray:
    sub = table.set_index("sequence_id")
    missing = [sid for sid in sequence_ids if sid not in sub.index]
    if missing:
        raise KeyError(
            f"{len(missing)} sequence_id(s) not in protein feature table, e.g. {missing[:3]}"
        )
    return sub.loc[list(sequence_ids), list(protein_cols)].values.astype(np.float32)


def merge_cds_and_protein_rows(
    cds_df: pd.DataFrame,
    protein_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    """按 feature_schema 列顺序合并 CDS 与蛋白质结构特征。"""
    col_data: dict[str, np.ndarray] = {}
    for c in feature_cols:
        if c in cds_df.columns:
            col_data[c] = np.asarray(cds_df[c].values)
        elif c in protein_df.columns:
            col_data[c] = np.asarray(protein_df[c].values)
        else:
            raise KeyError(f"Feature column missing in inputs: {c}")
    out = pd.DataFrame(col_data)
    out.insert(0, "sequence_id", cds_df["sequence_id"].values)
    return out


def canonical_sequence_id(raw: str) -> str:
    return str(raw).strip().replace("|", "_")


def read_pdb_root(explicit: Path | None = None) -> Path:
    if explicit is not None and str(explicit).strip():
        return Path(explicit).expanduser().resolve()
    env = (os.environ.get("DATASETA_PDB_ROOT") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    for base in (project_root() / "data" / "DatasetA", repo_root() / "data" / "DatasetA"):
        txt = base / "manifest" / "pdb_root.txt"
        if txt.is_file():
            line = txt.read_text(encoding="utf-8").strip().splitlines()[0].strip()
            return Path(line).expanduser().resolve()
    raise FileNotFoundError(
        "PDB root not set. Export DATASETA_PDB_ROOT or create data/DatasetA/manifest/pdb_root.txt"
    )


def resolve_pdb_path(
    sequence_id: str,
    *,
    pdb_root: Path | None = None,
    pdb_dir: Path | None = None,
    rel_path: str = "",
) -> Path:
    """解析单条 PDB：显式路径 > pdb_dir/{id}.pdb > pdb_root 下相对路径或递归搜索。"""
    sid = canonical_sequence_id(sequence_id)
    if rel_path and str(rel_path).strip():
        root = pdb_root or read_pdb_root()
        p = (root / str(rel_path).strip()).resolve()
        if p.is_file():
            return p
    if pdb_dir is not None:
        for name in (f"{sid}.pdb", f"{sequence_id.strip()}.pdb"):
            cand = Path(pdb_dir) / name
            if cand.is_file():
                return cand.resolve()
    root = pdb_root or read_pdb_root(None)
    for rel in (
        rel_path,
        f"{sid}.pdb",
        f"positive_pdb_train/{sid}.pdb",
        f"positive_pdb_val/{sid}.pdb",
        f"positive_pdb_test/{sid}.pdb",
    ):
        if not rel:
            continue
        cand = (root / str(rel)).resolve()
        if cand.is_file():
            return cand
    matches = list(root.rglob(f"{sid}.pdb"))
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        return matches[0].resolve()
    raise FileNotFoundError(
        f"No PDB for sequence_id={sequence_id!r} under {root}. "
        "Provide --pdb-dir, place {id}.pdb files, or run ESMFold first."
    )


def default_graph_mamba_config_path() -> Path:
    env = (os.environ.get("DEEPGVS_GRAPH_MAMBA_CONFIG") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return repo_root() / "model" / "graph_mamba_infer.json"


def load_graph_mamba_config(path: Path | None = None) -> dict:
    cfg_path = Path(path) if path else default_graph_mamba_config_path()
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"Graph-Mamba config not found: {cfg_path}. "
            "Copy model/graph_mamba_infer.json and set checkpoint / global_prott5_dir."
        )
    with cfg_path.open(encoding="utf-8") as f:
        return json.load(f)


def _require_graph_mamba_runtime_deps() -> None:
    """Checkpoint 在真实 Mamba 下训练；LSTM 回退无法加载权重。"""
    try:
        import mamba_ssm  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "现场 Graph-Mamba 需要 mamba-ssm（与训练环境一致）。"
            "请安装: pip install mamba-ssm fair-esm；或改用 --protein-mode csv 读预计算特征。"
        ) from e
    try:
        import esm  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "现场 Graph-Mamba 需要 fair-esm（ESM-2）。"
            "请安装: pip install fair-esm；或改用 --protein-mode csv。"
        ) from e


@dataclass
class GraphMambaRuntime:
    """懒加载 Graph-Mamba + ESM-2，按训练配置导出 graph-level embedding。"""

    config_path: Path
    device: str = "auto"
    _bundle: Optional[dict] = None

    def _load_bundle(self) -> dict:
        if self._bundle is not None:
            return self._bundle
        _require_graph_mamba_runtime_deps()
        import torch
        from esm2_feature_extractor import ESM2FeatureExtractor
        from graph_mamba_dataset import SimpleGraphMambaDataset, collate_graph_mamba_batch
        from graph_mamba_model import SimpleGraphMamba, _esm_module, parse_int_tuple3
        from pdb_graph import canonical_esm_layers, parse_esm_repr_layers

        cfg = load_graph_mamba_config(self.config_path)
        args = dict(cfg.get("args") or {})
        ckpt_path = Path(str(cfg["checkpoint"])).expanduser().resolve()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Graph-Mamba checkpoint missing: {ckpt_path}")

        if self.device == "auto":
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.device)

        esm_layers = canonical_esm_layers(parse_esm_repr_layers(str(args.get("esm_repr_layers", "33"))))
        esm_model_name = str(args.get("esm_model", "esm2_t33_650M_UR50D"))
        esm_dims = {
            "esm2_t33_650M_UR50D": 1280,
            "esm2_t36_3B_UR50D": 2560,
            "esm2_t48_15B_UR50D": 5120,
        }
        base_esm = esm_dims.get(esm_model_name, 1280)
        node_in_dim = base_esm * len(esm_layers)
        feat_dim = node_in_dim

        esm_extractor = ESM2FeatureExtractor(
            model_name=str(args.get("esm_model", "esm2_t33_650M_UR50D")),
            device=device,
            offline_mode=True,
        )
        esm_extractor.load_model()
        if bool(args.get("freeze_esm", True)):
            _esm_module(esm_extractor).eval()
            for p in _esm_module(esm_extractor).parameters():
                p.requires_grad = False

        use_glob = bool(args.get("use_global_prott5", False))
        mamba_tok = bool(args.get("mamba_global_token_prott5", False))
        glob_dim = int(args.get("global_prott5_dim", 1024))
        prott5_w = parse_int_tuple3(
            str(args.get("prott5_branch_widths", "512,256,128")),
            "prott5_branch_widths",
        )
        model = SimpleGraphMamba(
            input_dim=node_in_dim,
            hidden_dim=int(args.get("hidden_dim", 256)),
            num_classes=2,
            num_mamba_layers=int(args.get("num_mamba_layers", 4)),
            mamba_d_state=int(args.get("mamba_d_state", 16)),
            dropout=float(args.get("dropout", 0.1)),
            use_global_prott5=use_glob,
            mamba_global_token_prott5=mamba_tok,
            global_prott5_dim=glob_dim,
            global_proj_in_dim=glob_dim if (use_glob and not mamba_tok) else None,
            graph_conv=str(args.get("graph_conv", "gat")),
            gat_heads=int(args.get("gat_heads", 4)),
            sage_agg=str(args.get("sage_agg", "mean")),
            pooling=str(args.get("pooling", "multi_attn")),
            pool_heads=int(args.get("pool_heads", 6)),
            set2set_steps=int(args.get("set2set_steps", 4)),
            pool_merge_activation=str(args.get("pool_merge_activation", "gelu")),
            pool_attn_dropout=float(args.get("pool_attn_dropout", 0.0) or 0.0),
            parallel_fusion=str(args.get("parallel_fusion", "concat")),
            cross_interaction=str(args.get("cross_interaction", "none")),
            global_fusion=str(args.get("global_fusion", "concat")),
            prott5_branch=bool(args.get("prott5_branch", False)),
            prott5_branch_widths=prott5_w,
        ).to(device)

        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and "graph_mamba" in state:
            model.load_state_dict(state["graph_mamba"])
        else:
            model.load_state_dict(state)
        model.eval()

        esm_cache = (cfg.get("esm_cache_dir") or args.get("esm_cache_dir") or "").strip()
        cache_dir = Path(esm_cache).expanduser().resolve() if esm_cache else None  # noqa: PTH118
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

        g5_dir_s = (cfg.get("global_prott5_dir") or args.get("global_prott5_dir") or "").strip()
        g5_dir = Path(g5_dir_s).expanduser().resolve() if g5_dir_s else None  # noqa: PTH118

        self._bundle = {
            "torch": torch,
            "device": device,
            "model": model,
            "esm_extractor": esm_extractor,
            "esm_layers": esm_layers,
            "feat_dim": feat_dim,
            "args": args,
            "cfg": cfg,
            "cache_dir": cache_dir,
            "global_prott5_dir": g5_dir,
            "collate": collate_graph_mamba_batch,
            "Dataset": SimpleGraphMambaDataset,
        }
        return self._bundle

    def _make_dataset(self, records: List[dict]):
        b = self._load_bundle()
        args = b["args"]
        max_len = args.get("max_len")
        max_len_i = int(max_len) if max_len not in (None, "", 0) else None
        return b["Dataset"](
            records,
            Path("."),
            b["esm_extractor"],
            k=int(args.get("k", 30)),
            cache_dir=b["cache_dir"],
            max_len=max_len_i,
            use_global_prott5=bool(args.get("use_global_prott5", False)),
            global_prott5_dir=b["global_prott5_dir"],
            global_prott5_dim=int(args.get("global_prott5_dim", 1024)),
            esm_repr_layers=b["esm_layers"],
            esm_node_full_sequence=bool(args.get("esm_node_full_sequence", False)),
            esm_node_encode_max_len=int(args.get("esm_node_encode_max_len", 0) or 0),
            feat_dim=int(b.get("feat_dim") or 1280 * len(b["esm_layers"])),
        )

    def extract_from_pdb_records(
        self,
        records: List[dict],
        *,
        batch_size: Optional[int] = None,
    ) -> Tuple[List[str], np.ndarray]:
        """
        records: dict with keys sequence_id, pdb_path (absolute), label (optional, default 0)
        Returns (sequence_ids, embeddings float32 [N, D])
        """
        if not records:
            return [], np.zeros((0, 0), dtype=np.float32)

        b = self._load_bundle()
        torch = b["torch"]
        from torch.utils.data import DataLoader

        args = b["args"]
        bs = int(batch_size or args.get("batch_size", 4) or 4)
        emb_kind = str(b["cfg"].get("embedding_export_kind", "graph_heads_global"))

        ds = self._make_dataset(records)
        loader = DataLoader(
            ds,
            batch_size=bs,
            shuffle=False,
            num_workers=0,
            collate_fn=b["collate"],
        )

        out_ids: List[str] = []
        out_rows: List[np.ndarray] = []
        model = b["model"]
        device = b["device"]

        with torch.no_grad():
            for batch in loader:
                node_feats = batch["node_features"].to(device)
                adjs = batch["adj"].to(device)
                masks = batch["mask"].to(device)
                gp5 = batch.get("global_prott5")
                if gp5 is not None:
                    gp5 = gp5.to(device)
                g = model(
                    node_feats,
                    adjs,
                    masks,
                    global_prott5=gp5,
                    return_embedding=True,
                    embedding_kind=emb_kind,
                )
                g_np = g.cpu().numpy()
                for sid, emb in zip(batch["sequence_id"], g_np):
                    out_ids.append(str(sid))
                    out_rows.append(emb.astype(np.float32, copy=False))

        if not out_rows:
            return [], np.zeros((0, 0), dtype=np.float32)
        return out_ids, np.stack(out_rows, axis=0)

    def extract_for_sequence_ids(
        self,
        sequence_ids: Sequence[str],
        *,
        pdb_paths: Optional[Mapping[str, str | Path]] = None,
        pdb_root: Path | None = None,
        pdb_dir: Path | None = None,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        pdb_paths = pdb_paths or {}
        records: List[dict] = []
        order: List[str] = [canonical_sequence_id(s) for s in sequence_ids]
        for sid_raw in sequence_ids:
            sid = canonical_sequence_id(sid_raw)
            rel = ""
            if sid in pdb_paths:
                p = Path(pdb_paths[sid]).expanduser().resolve()
            elif sid_raw in pdb_paths:
                p = Path(pdb_paths[sid_raw]).expanduser().resolve()
            else:
                p = resolve_pdb_path(
                    sid,
                    pdb_root=pdb_root,
                    pdb_dir=pdb_dir,
                    rel_path=rel,
                )
            if not p.is_file():
                raise FileNotFoundError(f"PDB not found for {sid}: {p}")
            records.append({"sequence_id": sid, "label": 0, "pdb_path": str(p)})
        ids_out, mat = self.extract_from_pdb_records(records, batch_size=batch_size)
        idx = {s: i for i, s in enumerate(ids_out)}
        if set(idx) != set(order):
            missing = [s for s in order if s not in idx]
            raise RuntimeError(f"Graph-Mamba export missing IDs: {missing[:5]}")
        return np.stack([mat[idx[s]] for s in order], axis=0).astype(np.float32)


_RUNTIME: Optional[GraphMambaRuntime] = None


def get_graph_mamba_runtime(
    config_path: Path | None = None,
    device: str = "auto",
) -> GraphMambaRuntime:
    global _RUNTIME
    path = Path(config_path) if config_path else default_graph_mamba_config_path()
    if _RUNTIME is None or _RUNTIME.config_path.resolve() != path.resolve():
        _RUNTIME = GraphMambaRuntime(config_path=path, device=device)
    return _RUNTIME


def compute_protein_features_online(
    sequence_ids: Sequence[str],
    protein_cols: Sequence[str],
    *,
    pdb_paths: Optional[Mapping[str, str | Path]] = None,
    pdb_root: Path | None = None,
    pdb_dir: Path | None = None,
    graph_mamba_config: Path | None = None,
    device: str = "auto",
    batch_size: Optional[int] = None,
) -> np.ndarray:
    """从 PDB 现场计算 Graph-Mamba 嵌入，列维与 ``protein_cols`` 一致。"""
    runtime = get_graph_mamba_runtime(graph_mamba_config, device=device)
    mat = runtime.extract_for_sequence_ids(
        sequence_ids,
        pdb_paths=pdb_paths,
        pdb_root=pdb_root,
        pdb_dir=pdb_dir,
        batch_size=batch_size,
    )
    if mat.shape[1] != len(protein_cols):
        raise ValueError(
            f"Graph-Mamba embedding dim {mat.shape[1]} != schema protein columns {len(protein_cols)}. "
            "Use the checkpoint / graph_mamba_infer.json from the same training run."
        )
    return mat


def lookup_or_compute_protein_features(
    sequence_ids: Sequence[str],
    protein_cols: Sequence[str],
    *,
    mode: str = "auto",
    protein_table: Path | None = None,
    pdb_paths: Optional[Mapping[str, str | Path]] = None,
    pdb_root: Path | None = None,
    pdb_dir: Path | None = None,
    graph_mamba_config: Path | None = None,
    device: str = "auto",
) -> np.ndarray:
    """
    mode:
      - csv: 仅查表
      - compute: 仅从 PDB 计算
      - auto: 能查表则查表，其余用 PDB 计算
    """
    mode = (mode or "auto").lower().strip()
    ids = [canonical_sequence_id(s) for s in sequence_ids]

    if mode == "compute":
        return compute_protein_features_online(
            ids,
            protein_cols,
            pdb_paths=pdb_paths,
            pdb_root=pdb_root,
            pdb_dir=pdb_dir,
            graph_mamba_config=graph_mamba_config,
            device=device,
        )

    table_df: Optional[pd.DataFrame] = None
    if mode in ("csv", "auto") and protein_table is not None and Path(protein_table).is_file():
        table_df = load_protein_features_table(protein_table)

    if mode == "csv":
        if table_df is None:
            raise FileNotFoundError(f"protein feature table required for mode=csv: {protein_table}")
        return lookup_protein_features(ids, table_df, protein_cols)

    # auto
    if table_df is not None:
        sub = table_df.set_index("sequence_id")
        have = [s for s in ids if s in sub.index]
        need_compute = [s for s in ids if s not in sub.index]
        if not need_compute:
            return lookup_protein_features(ids, table_df, protein_cols)
        mat_csv = (
            lookup_protein_features(have, table_df, protein_cols) if have else np.zeros((0, len(protein_cols)), np.float32)
        )
        mat_new = compute_protein_features_online(
            need_compute,
            protein_cols,
            pdb_paths=pdb_paths,
            pdb_root=pdb_root,
            pdb_dir=pdb_dir,
            graph_mamba_config=graph_mamba_config,
            device=device,
        )
        out = np.zeros((len(ids), len(protein_cols)), dtype=np.float32)
        hi = {s: i for i, s in enumerate(have)}
        ni = {s: i for i, s in enumerate(need_compute)}
        for j, sid in enumerate(ids):
            if sid in hi:
                out[j] = mat_csv[hi[sid]]
            else:
                out[j] = mat_new[ni[sid]]
        return out

    return compute_protein_features_online(
        ids,
        protein_cols,
        pdb_paths=pdb_paths,
        pdb_root=pdb_root,
        pdb_dir=pdb_dir,
        graph_mamba_config=graph_mamba_config,
        device=device,
    )
