#!/usr/bin/env python3
"""在打包测试集上评估 DeepGVS stacking 模型。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import predict_labels, repo_root


def load_test_data(path: Path) -> dict:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError("test_data.pt must be a dict with keys: features, labels, sequence_ids")
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DeepGVS on packaged test data.")
    parser.add_argument("-m", "--model_path", type=str, default="model", help="Model directory")
    parser.add_argument("-i", "--test_data_path", type=str, default="data/test_data.pt")
    parser.add_argument("-o", "--output_path", type=str, default="results/output.csv")
    parser.add_argument("-p", "--threshold", type=float, default=0.5)
    args = parser.parse_args()

    root = repo_root()
    model_dir = (root / args.model_path).resolve()
    test_path = (root / args.test_data_path).resolve()
    out_path = (root / args.output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = load_test_data(test_path)
    X = bundle["features"]
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    ids = list(bundle["sequence_ids"])
    labels = bundle.get("labels")
    if labels is not None and isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    p_vf, pred = predict_labels(X, model_dir, threshold=args.threshold)

    out = pd.DataFrame(
        {
            "Sample_ID": ids,
            "VF_probability": p_vf.astype(float),
            "Prediction": np.where(pred == 1, "VF", "non-VF"),
        }
    )
    if labels is not None:
        out.insert(1, "label", labels.astype(int))
        acc = float((pred == labels.astype(int)).mean())
        print(f"[metrics] accuracy={acc:.4f} (n={len(ids)})")

    out.to_csv(out_path, index=False)
    print(f"[done] wrote {out_path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
