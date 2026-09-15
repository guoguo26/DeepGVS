#!/usr/bin/env python3
"""Evaluate a trained DeepGVS bundle on a merged-feature CSV with labels.

Usage:
  python scripts/evaluate_model.py --model-dir model/dataset_a --test-csv data/Dataset_A/test.csv \
      --out results/metrics.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deepgvs.predict import predict_from_raw, validate_bundle  # noqa: E402
from deepgvs.protein_feature import load_feature_schema  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate DeepGVS.")
    ap.add_argument("--model-dir", default="model/dataset_a")
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--out", default="results/metrics.csv")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    dim = validate_bundle(args.model_dir)
    schema = load_feature_schema(Path(args.model_dir))
    cols = list(schema["feature_columns"])
    df = pd.read_csv(args.test_csv)
    X = df[cols].values.astype(np.float32)
    ids = df["sequence_id"].astype(str).tolist()
    p_vf, pred = predict_from_raw(X, args.model_dir, threshold=args.threshold)
    out = pd.DataFrame({"Sample_ID": ids, "VF_probability": p_vf,
                        "Prediction": np.where(pred == 1, "VF", "non-VF")})
    if "label" in df.columns:
        y = df["label"].values.astype(int)
        out.insert(1, "label", y)
        from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
        print(f"accuracy={accuracy_score(y, pred):.4f} "
              f"f1={f1_score(y, pred):.4f} mcc={matthews_corrcoef(y, pred):.4f} "
              f"auc={roc_auc_score(y, p_vf):.4f} (n={len(y)}, pre_dim={dim})")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
