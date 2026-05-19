#!/usr/bin/env python3
"""
DeepGVS 新样本毒力因子（VF）预测入口。

整体流程
========

  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 输入（三选一，互斥）                                                      │
  ├─────────────────┬─────────────────┬───────────────────────────────────────┤
  │ A merged CSV    │ B 配对 FASTA    │ C features182 CSV                     │
  │ 2146 维已合并   │ -pseq + -cds    │ 182 维已预处理                        │
  └────────┬────────┴────────┬────────┴──────────────┬────────────────────────┘
           │                 │                         │
           │                 ▼                         │
           │    ┌────────────────────────────┐         │
           │    │ read_fasta + ID 配对       │         │
           │    │ cds_feature → ~354 维    │         │
           │    │ protein_feature → 1792 维  │         │
           │    │ merge → 2146 维 X_raw    │         │
           │    └─────────────┬──────────────┘         │
           │                  │                         │
           └──────────────────┼─────────────────────────┘
                              ▼
           ┌──────────────────────────────────────────────┐
           │ [1/3] validate_model_bundle(model/)           │
           │  preprocess.joblib 输出维 == rf.pkl 输入维    │
           └──────────────────────┬───────────────────────┘
                                  ▼
           ┌──────────────────────────────────────────────┐
           │ [2/3] 特征矩阵 X                              │
           │  A/B: (N, 2146)  C: (N, 182) 跳过下一步       │
           └──────────────────────┬───────────────────────┘
                                  ▼
           ┌──────────────────────────────────────────────┐
           │ [3/3] model.predict_labels /                   │
           │      predict_labels_from_preprocessed          │
           │                                               │
           │  2146 ──preprocess.joblib──► 182              │
           │  182 ──rf/svm/xgb/mlp──► 4×概率块             │
           │  概率块 ──nam.pt──► VF 概率 + 标签            │
           └──────────────────────┬───────────────────────┘
                                  ▼
           ┌──────────────────────────────────────────────┐
           │ results/prediction.csv                        │
           │  Sample_ID, VF_probability, Prediction        │
           └──────────────────────────────────────────────┘

模式 B 中蛋白结构特征（--protein-mode）
------------------------------------
  csv     : 只查 --protein-features 表（缺 ID 报错）
  compute : 只从 PDB 跑 Graph-Mamba（需 --graph-mamba-config 或 DEEPGVS_GRAPH_MAMBA_CONFIG）
  auto    : 先查表，缺的 ID 再尝试 PDB 计算

模型目录 model/ 必需文件
------------------------
  feature_schema.json   # 2146 列名与 CDS/蛋白列划分
  preprocess.joblib     # 训练集 fit 的 2146→182 变换（MI + 蛋白 PCA）
  rf.pkl svm.pkl xgboost.pkl mlp.pkl
  nam.pt

示例（在 DeepGVS/ 下执行）
--------------------------
  python code/prediction.py -m model/ --merged-features ../data/example_features.csv
  python code/prediction.py -m model/ -pseq Dataset/example/example_protein.fasta \\
      -cds Dataset/example/example_cds.fasta --protein-mode csv
  ./run_vfs.sh code/prediction.py -m model/ -pseq ... -cds ... \\
      --protein-mode compute --pdb-dir Dataset/example/PDB

注意事项
========
1. 依赖模块（code/cds_feature.py, protein_feature.py, model.py, nam.py）
   compute_feature_row / lookup_or_compute_protein_features / merge_cds_and_protein_rows
   须返回 dict、ndarray、DataFrame；接口变更会导致 [2/3] 失败。

2. 模型目录 -m model/
   须含 feature_schema.json, preprocess.joblib, rf/svm/xgboost/mlp.pkl, nam.pt
   （或 base_learners.pkl、nam_meta_learner.pt 等等效文件）；[1/3] 会逐项检查。

3. FASTA ID
   header 第一个 token 为 ID；'|' -> '_'。仅两侧都有的 ID 参与预测，其余 [warn] 丢弃。
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from cds_feature import compute_feature_row, read_fasta_first_token_as_id
from model import (
    predict_labels,
    predict_labels_from_preprocessed,
    repo_root,
    validate_model_bundle,
)
from nam import split_protein_dna_feature_cols
from protein_feature import (
    canonical_sequence_id,
    default_graph_mamba_config_path,
    load_feature_schema,
    lookup_or_compute_protein_features,
    merge_cds_and_protein_rows,
    project_root,
    protein_feature_columns,
)

# 相对 DeepGVS/ 或项目根 guo/5.5/ 的默认数据路径
_DEFAULT_MERGED = (
    "data/example_features.csv",
    "../data/example_features.csv",
)
_DEFAULT_PROTEIN_FASTA = (
    "Dataset/example/example_protein.fasta",
    "data/example_protein.fasta",
    "../data/example_protein.fasta",
)
_DEFAULT_CDS_FASTA = (
    "Dataset/example/example_cds.fasta",
    "Dataset/example/example_cds.fna",
    "data/example_cds.fna",
    "../data/example_cds.fna",
)
_DEFAULT_PROTEIN_TABLE = (
    "data/reference_protein_features.csv",
    "../data/reference_protein_features.csv",
)


def _first_existing(bases: Sequence[Path], rel_paths: Sequence[str]) -> Path | None:
    for base in bases:
        for rel in rel_paths:
            p = (base / rel).resolve()
            if p.is_file():
                return p
    return None


def _resolve_user_path(
    user: str,
    *,
    bases: Sequence[Path],
    defaults: Sequence[str],
    label: str,
    required: bool = False,
) -> Path | None:
    if user.strip():
        p = Path(user).expanduser()
        if not p.is_absolute():
            for base in (Path.cwd(), *bases):
                cand = (base / p).resolve()
                if cand.is_file():
                    return cand
        p = p.resolve()
        if not p.is_file():
            raise FileNotFoundError(f"{label} not found: {p}")
        return p
    hit = _first_existing(bases, defaults)
    if hit is not None:
        return hit
    if required:
        tried = ", ".join(str((b / defaults[0]).resolve()) for b in bases[:2])
        raise FileNotFoundError(f"{label} not found. Tried defaults under repo/project, e.g. {tried}")
    return None


def _torch_device(name: str) -> torch.device:
    name = (name or "cpu").strip().lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _match_paired_fastas(
    protein_records: list[tuple[str, str]],
    cds_records: list[tuple[str, str]],
    *,
    max_examples: int = 5,
) -> list[tuple[str, str, str]]:
    """Pair protein/CDS by sequence_id; unmatched IDs are dropped with a warning."""
    prot = {canonical_sequence_id(sid): seq for sid, seq in protein_records}
    cds = {canonical_sequence_id(sid): seq for sid, seq in cds_records}
    common = sorted(set(prot.keys()) & set(cds.keys()))
    if not common:
        raise ValueError(
            "No matched sequence IDs between protein and CDS FASTA. "
            "Check headers (first token = ID) and that '|' vs '_' naming is consistent."
        )
    only_p = sorted(set(prot.keys()) - set(cds.keys()))
    only_c = sorted(set(cds.keys()) - set(prot.keys()))
    if only_p:
        ex = ", ".join(only_p[:max_examples])
        suffix = f" ... (+{len(only_p) - max_examples} more)" if len(only_p) > max_examples else ""
        print(f"[warn] {len(only_p)} protein ID(s) without CDS — skipped, e.g. {ex}{suffix}")
    if only_c:
        ex = ", ".join(only_c[:max_examples])
        suffix = f" ... (+{len(only_c) - max_examples} more)" if len(only_c) > max_examples else ""
        print(f"[warn] {len(only_c)} CDS ID(s) without protein — skipped, e.g. {ex}{suffix}")
    print(f"[info] {len(common)} paired sample(s) will be predicted")
    return [(sid, prot[sid], cds[sid]) for sid in common]


def _validate_feature_matrix(X: np.ndarray, ids: Sequence[str], n_features: int, label: str) -> None:
    if X.ndim != 2:
        raise ValueError(f"{label}: expected 2-D array, got shape {X.shape}")
    if X.shape[0] != len(ids):
        raise ValueError(f"{label}: {X.shape[0]} rows != {len(ids)} sequence_id(s)")
    if X.shape[1] != n_features:
        raise ValueError(f"{label}: {X.shape[1]} columns != expected {n_features}")


def build_raw_feature_matrix(
    pairs: list[tuple[str, str, str]],
    model_dir: Path,
    protein_table: Path | None,
    *,
    protein_mode: str = "auto",
    pdb_dir: Path | None = None,
    pdb_root: Path | None = None,
    graph_mamba_config: Path | None = None,
    device: str = "auto",
) -> tuple[np.ndarray, list[str]]:
    schema = load_feature_schema(model_dir)
    feature_cols: list[str] = list(schema["feature_columns"])
    _, dna_cols = split_protein_dna_feature_cols(feature_cols)
    prot_cols = protein_feature_columns(schema)

    cds_rows = []
    for sid, _pseq, cds_seq in pairs:
        row = compute_feature_row(
            seq_id=sid,
            seq=cds_seq,
            feat_cols=dna_cols,
            local_window_size=int(schema.get("local_window_size", 300)),
            local_window_step=int(schema.get("local_window_step", 150)),
            orf_max_len_nt=int(schema.get("orf_max_len_nt", 0)),
        )
        cds_rows.append(row)
    cds_df = pd.DataFrame(cds_rows)

    ids = [canonical_sequence_id(sid) for sid, _, _ in pairs]
    prot_mat = lookup_or_compute_protein_features(
        ids,
        prot_cols,
        mode=protein_mode,
        protein_table=protein_table,
        pdb_dir=pdb_dir,
        pdb_root=pdb_root,
        graph_mamba_config=graph_mamba_config,
        device=device,
    )
    prot_idx = pd.Index(prot_cols, dtype="string")
    prot_part = pd.DataFrame(prot_mat, columns=prot_idx)
    prot_part = prot_part.assign(sequence_id=pd.Series(ids, dtype="string"))
    prot_part = prot_part.reindex(columns=pd.Index(["sequence_id", *prot_cols]))

    if not isinstance(prot_mat, np.ndarray):
        raise TypeError(
            "lookup_or_compute_protein_features must return numpy.ndarray; "
            f"got {type(prot_mat).__name__}"
        )
    if prot_mat.shape != (len(ids), len(prot_cols)):
        raise ValueError(
            f"Protein features shape {prot_mat.shape} != ({len(ids)}, {len(prot_cols)})"
        )

    merged = merge_cds_and_protein_rows(cds_df, prot_part, feature_cols)
    if not isinstance(merged, pd.DataFrame):
        raise TypeError("merge_cds_and_protein_rows must return pandas.DataFrame")

    X = merged[list(feature_cols)].values.astype(np.float32)
    _validate_feature_matrix(X, ids, len(feature_cols), "build_raw_feature_matrix")
    return X, ids


def _load_merged_features_csv(path: Path, model_dir: Path) -> tuple[np.ndarray, list[str]]:
    schema = load_feature_schema(model_dir)
    feature_cols = list(schema["feature_columns"])
    merged = pd.read_csv(path)
    if "sequence_id" not in merged.columns:
        raise ValueError(f"{path}: missing column 'sequence_id'")
    missing = [c for c in feature_cols if c not in merged.columns]
    if missing:
        raise ValueError(f"{path}: missing {len(missing)} feature columns, e.g. {missing[:5]}")
    ids = merged["sequence_id"].astype(str).tolist()
    X = merged[list(feature_cols)].values.astype(np.float32)
    _validate_feature_matrix(X, ids, len(feature_cols), str(path))
    return X, ids


def _load_features182_csv(path: Path, model_dir: Path) -> tuple[np.ndarray, list[str]]:
    df = pd.read_csv(path)
    if "sequence_id" not in df.columns:
        raise ValueError(f"{path}: missing column 'sequence_id'")
    from model import expected_base_input_dim

    n = expected_base_input_dim(model_dir)
    feat_cols = [c for c in df.columns if c != "sequence_id"]
    if len(feat_cols) != n:
        raise ValueError(
            f"{path}: expected {n} numeric feature columns (besides sequence_id), got {len(feat_cols)}"
        )
    ids = df["sequence_id"].astype(str).tolist()
    X = df[feat_cols].values.astype(np.float32)
    _validate_feature_matrix(X, ids, n, str(path))
    return X, ids


def _write_predictions(
    ids: list[str],
    p_vf: np.ndarray,
    pred: np.ndarray,
    out_path: Path,
) -> None:
    out = pd.DataFrame(
        {
            "Sample_ID": ids,
            "VF_probability": p_vf.astype(float),
            "Prediction": np.where(pred == 1, "VF", "non-VF"),
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[done] wrote {out_path} ({len(out)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepGVS VF prediction (2146-dim raw features or paired FASTA).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (run from DeepGVS/):
  python code/prediction.py -m model/ --merged-features ../data/example_features.csv
  python code/prediction.py -m model/ \\
      -pseq Dataset/example/example_protein.fasta \\
      -cds Dataset/example/example_cds.fasta \\
      --protein-mode csv --protein-features data/reference_protein_features.csv

Graph-Mamba on-the-fly (conda env VFS recommended):
  ./run_vfs.sh code/prediction.py -m model/ -pseq ... -cds ... \\
      --protein-mode compute --pdb-dir Dataset/example/PDB
""",
    )
    parser.add_argument("-m", "--model_path", type=str, default="model", help="Model directory")
    parser.add_argument(
        "--merged-features",
        type=str,
        default="",
        help="CSV: sequence_id + 2146 raw feature columns from feature_schema.json",
    )
    parser.add_argument(
        "--features182",
        type=str,
        default="",
        help="CSV: sequence_id + preprocessed columns (182-dim); skips preprocess.joblib",
    )
    parser.add_argument("-pseq", "--protein_fasta", type=str, default="", help="Protein FASTA")
    parser.add_argument("-cds", "--cds_fasta", type=str, default="", help="CDS nucleotide FASTA")
    parser.add_argument(
        "--protein-features",
        type=str,
        default="",
        help="Precomputed graph_mamba_feat_* table (default: data/reference_protein_features.csv)",
    )
    parser.add_argument(
        "--protein-mode",
        type=str,
        default="compute",
        choices=("auto", "csv", "compute"),
        help="Protein features: compute=Graph-Mamba from PDB (default); csv=precomputed table only; auto=table then PDB",
    )
    parser.add_argument("--pdb-dir", type=str, default="", help="PDB dir with {sequence_id}.pdb")
    parser.add_argument("--pdb-root", type=str, default="", help="External PDB root (DATASETA_PDB_ROOT)")
    parser.add_argument(
        "--graph-mamba-config",
        type=str,
        default="",
        help="Graph-Mamba JSON (or set DEEPGVS_GRAPH_MAMBA_CONFIG); required for --protein-mode compute",
    )
    parser.add_argument(
        "--gm-device",
        type=str,
        default="auto",
        help="Torch device for Graph-Mamba feature extraction (auto|cpu|cuda:0)",
    )
    parser.add_argument(
        "--inference-device",
        type=str,
        default="cpu",
        help="Torch device for NAM meta-learner (cpu|cuda:0|auto)",
    )
    parser.add_argument("-o", "--output_path", type=str, default="results/prediction.csv")
    parser.add_argument("-p", "--threshold", type=float, default=0.5)
    parser.add_argument(
        "--skip-model-check",
        action="store_true",
        help="Skip preprocess vs base-learner dimension check",
    )
    args = parser.parse_args()

    root = repo_root()
    proj = project_root()
    bases = (root, proj)
    model_dir = (root / args.model_path).resolve()
    out_path = (root / args.output_path).resolve()
    infer_dev = _torch_device(args.inference_device)

    if not args.skip_model_check:
        dim = validate_model_bundle(model_dir)
        print(f"[1/3] model OK: preprocess 2146 -> {dim} -> stacking (rf/svm/xgb/mlp + nam)")

    modes: list[str] = []
    if args.features182.strip():
        modes.append("features182")
    if args.merged_features.strip():
        modes.append("merged")
    fasta_partial = bool(args.protein_fasta.strip()) ^ bool(args.cds_fasta.strip())
    if fasta_partial:
        parser.error("FASTA mode requires both -pseq and -cds.")
    if args.protein_fasta.strip() and args.cds_fasta.strip():
        modes.append("fasta")

    if not modes:
        merged_default = _first_existing(bases, _DEFAULT_MERGED)
        if merged_default is not None:
            print(f"[info] using default merged features: {merged_default}")
            args.merged_features = str(merged_default)
            modes = ["merged"]
        else:
            parser.error(
                "Specify one input: --merged-features, --features182, or both -pseq and -cds."
            )
    if len(modes) > 1:
        parser.error(f"Conflicting inputs ({', '.join(modes)}); use only one mode.")

    prot_table: Path | None
    if args.protein_mode == "compute":
        prot_table = (
            Path(args.protein_features).resolve() if args.protein_features.strip() else None
        )
    else:
        prot_table = _resolve_user_path(
            args.protein_features,
            bases=bases,
            defaults=_DEFAULT_PROTEIN_TABLE,
            label="Protein feature table",
            required=args.protein_mode == "csv",
        )

    pdb_dir = Path(args.pdb_dir).resolve() if args.pdb_dir.strip() else None
    pdb_root = Path(args.pdb_root).resolve() if args.pdb_root.strip() else None
    gm_cfg: Path | None
    if args.graph_mamba_config.strip():
        gm_cfg = Path(args.graph_mamba_config).resolve()
    else:
        _gm_default = default_graph_mamba_config_path()
        gm_cfg = _gm_default if _gm_default is not None and _gm_default.is_file() else None

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

        if args.features182.strip():
            path182 = _resolve_user_path(
                args.features182,
                bases=bases,
                defaults=(),
                label="features182 CSV",
                required=True,
            )
            assert path182 is not None
            print(f"[2/3] load preprocessed features: {path182}")
            X, ids = _load_features182_csv(path182, model_dir)
            print("[3/3] stacking inference (182-dim, no preprocess)...")
            p_vf, pred = predict_labels_from_preprocessed(
                X, model_dir, threshold=args.threshold, device=infer_dev
            )
        elif args.merged_features.strip():
            merged_path = _resolve_user_path(
                args.merged_features,
                bases=bases,
                defaults=_DEFAULT_MERGED,
                label="Merged features CSV",
                required=True,
            )
            assert merged_path is not None
            print(f"[2/3] load merged raw features: {merged_path}")
            X, ids = _load_merged_features_csv(merged_path, model_dir)
            print("[3/3] preprocess (2146->182) + stacking inference...")
            p_vf, pred = predict_labels(X, model_dir, threshold=args.threshold, device=infer_dev)
        else:
            pseq_path = _resolve_user_path(
                args.protein_fasta,
                bases=bases,
                defaults=_DEFAULT_PROTEIN_FASTA,
                label="Protein FASTA",
                required=True,
            )
            cds_path = _resolve_user_path(
                args.cds_fasta,
                bases=bases,
                defaults=_DEFAULT_CDS_FASTA,
                label="CDS FASTA",
                required=True,
            )
            assert pseq_path is not None and cds_path is not None
            protein_records = read_fasta_first_token_as_id(pseq_path)
            cds_records = read_fasta_first_token_as_id(cds_path)
            pairs = _match_paired_fastas(protein_records, cds_records)
            print(f"[2/3] FASTA -> 2146-dim features ({len(pairs)} sample(s))")
            X, ids = build_raw_feature_matrix(
                pairs,
                model_dir,
                prot_table,
                protein_mode=args.protein_mode,
                pdb_dir=pdb_dir,
                pdb_root=pdb_root,
                graph_mamba_config=gm_cfg,
                device=args.gm_device,
            )
            print("[3/3] preprocess + stacking inference...")
            p_vf, pred = predict_labels(X, model_dir, threshold=args.threshold, device=infer_dev)

    _write_predictions(ids, p_vf, pred, out_path)


if __name__ == "__main__":
    main()
