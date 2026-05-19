#!/usr/bin/env python3
"""Fit and save preprocess.joblib (2146 -> 182) without train_stacked_nam_ensemble.py."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from nam import FittedSingleStreamPreprocessor, SingleStreamPreprocessConfig, split_protein_dna_feature_cols
from protein_feature import repo_root


def _graph_cols(cols: list[str]) -> list[str]:
    protein, _ = split_protein_dna_feature_cols(cols)
    return protein


def merge_cds_and_structure(cds_path: Path, struct_path: Path) -> tuple[pd.DataFrame, list[str]]:
    cds = pd.read_csv(cds_path)
    struct = pd.read_csv(struct_path)
    prot_cols = [c for c in struct.columns if c.startswith("graph_mamba_feat_")]
    meta = [c for c in ("sequence_id", "label", "split") if c in cds.columns or c in struct.columns]
    cds_feat = [c for c in cds.columns if c not in meta and c not in prot_cols]
    merged = cds[["sequence_id"] + [c for c in meta if c != "sequence_id" and c in cds.columns]].merge(
        struct[["sequence_id"] + prot_cols],
        on="sequence_id",
        how="inner",
    )
    for c in cds_feat:
        if c in cds.columns:
            merged[c] = cds[c].values
    feature_cols = prot_cols + cds_feat
    if "length" in cds.columns and "length" not in feature_cols:
        feature_cols = ["length"] + feature_cols
    schema_path = repo_root() / "model" / "feature_schema.json"
    if schema_path.is_file():
        feature_cols = list(json.loads(schema_path.read_text(encoding="utf-8"))["feature_columns"])
        for c in feature_cols:
            if c not in merged.columns:
                raise ValueError(f"Merged table missing column: {c}")
        merged = merged[["sequence_id"] + [c for c in meta if c != "sequence_id" and c in merged.columns] + feature_cols]
    return merged, feature_cols


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Dump preprocess.joblib for DeepGVS.")
    parser.add_argument(
        "--cds-features",
        type=str,
        default=str(root / "Feature/Dataset_A/CDS/features_all.csv"),
    )
    parser.add_argument(
        "--structural-features",
        type=str,
        default=str(
            root / "Feature/Dataset_A/protein_structure/graph_mamba_features_all_splits.idfixed.csv"
        ),
    )
    parser.add_argument("--merged-features", type=str, default="", help="Or one merged CSV with split+label")
    parser.add_argument("--train-splits", type=str, default="train,val")
    parser.add_argument("--feature-select-k", type=int, default=700)
    parser.add_argument("--graph-pca-var", type=float, default=0.95)
    parser.add_argument(
        "--output-joblib",
        type=str,
        default=str(root / "model/exports/dataset_a/preprocess.joblib"),
    )
    args = parser.parse_args()

    if args.merged_features.strip():
        merged = pd.read_csv(Path(args.merged_features).resolve())
        schema_path = root / "model/feature_schema.json"
        feature_cols = list(json.loads(schema_path.read_text(encoding="utf-8"))["feature_columns"])
    else:
        merged, feature_cols = merge_cds_and_structure(
            Path(args.cds_features).resolve(),
            Path(args.structural_features).resolve(),
        )

    if "split" not in merged.columns or "label" not in merged.columns:
        raise ValueError("Need columns: split, label")

    train_names = {s.strip().lower() for s in args.train_splits.split(",") if s.strip()}
    mask = merged["split"].astype(str).str.strip().str.lower().isin(train_names)
    X = merged.loc[mask, feature_cols].values.astype(np.float32)
    y = merged.loc[mask, "label"].astype(int).values

    cfg = SingleStreamPreprocessConfig(
        feature_select_method="mi",
        feature_select_k=args.feature_select_k,
        graph_pca_var=args.graph_pca_var,
    )
    pre = FittedSingleStreamPreprocessor().fit(X, y, feature_cols, cfg)
    out = Path(args.output_joblib).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"preprocessor": pre, "feature_cols": list(feature_cols)}, out)
    print(f"[done] saved {out} (final_dim={pre.final_dim_}, train_rows={int(mask.sum())})")


if __name__ == "__main__":
    main()
