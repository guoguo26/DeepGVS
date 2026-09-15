"""DeepGVS stacking training (RF/SVM/XGBoost/MLP + NAM meta-learner).

Trains on merged 2146-dim raw features (1792 Graph-Mamba + 354 CDS).
For the exact manuscript run see configs/*.yaml and docs/reproduction.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:  # pragma: no cover
    _HAS_XGB = False

from nam import (
    FittedSingleStreamPreprocessor,
    NeuralAdditiveModel,
    SingleStreamPreprocessConfig,
    proba_to_meta_block,
)


def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_merged_csv(path: Path, feature_cols: list[str] | None = None):
    df = pd.read_csv(path)
    if "sequence_id" not in df.columns:
        raise ValueError(f"{path}: missing 'sequence_id'")
    if "label" not in df.columns:
        raise ValueError(f"{path}: missing 'label' (0/1)")
    feat_cols = feature_cols or [c for c in df.columns if c not in ("sequence_id", "label")]
    X = df[feat_cols].values.astype(np.float32)
    y = df["label"].values.astype(int)
    ids = df["sequence_id"].astype(str).tolist()
    return X, y, ids, feat_cols


def build_base_learners(seed: int = 42) -> dict:
    learners = {
        "random_forest": RandomForestClassifier(
            n_estimators=500, n_jobs=-1, class_weight="balanced", random_state=seed),
        "svm_rbf": make_pipeline(
            StandardScaler(),
            SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced", random_state=seed)),
        "mlp": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=800, random_state=seed)),
    }
    if _HAS_XGB:
        learners["xgboost"] = XGBClassifier(
            n_estimators=600, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            tree_method="hist", eval_metric="logloss", random_state=seed)
    return learners


def train_nam(meta_X: np.ndarray, y: np.ndarray, num_classes: int = 2,
              hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1,
              epochs: int = 50, lr: float = 1e-3, seed: int = 42,
              device: str = "cpu") -> NeuralAdditiveModel:
    dev = torch.device(device)
    g = torch.Generator().manual_seed(seed)
    model = NeuralAdditiveModel(meta_X.shape[1], num_classes, hidden_dim, num_layers, dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    X_t = torch.from_numpy(meta_X.astype(np.float32))
    y_t = torch.from_numpy(y.astype(np.int64))
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X_t), generator=g)
        opt.zero_grad()
        logits = model(X_t[perm].to(dev))
        loss = loss_fn(logits, y_t[perm].to(dev))
        loss.backward()
        opt.step()
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description="Train DeepGVS stacking model.")
    ap.add_argument("--train-csv", required=True, help="Merged 2146-dim CSV with sequence_id,label")
    ap.add_argument("--val-csv", default="", help="Optional validation CSV (for reporting only)")
    ap.add_argument("--feature-columns", default="", help="feature_schema.json to fix column order")
    ap.add_argument("-o", "--out-dir", default="model/dataset_a")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nam-epochs", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    set_seed(args.seed)

    feat_cols = None
    if args.feature_columns.strip():
        feat_cols = json.loads(Path(args.feature_columns).read_text())["feature_columns"]
    Xtr, ytr, _, feat_cols = load_merged_csv(Path(args.train_csv), feat_cols)

    cfg = SingleStreamPreprocessConfig(random_seed=args.seed)
    pre = FittedSingleStreamPreprocessor().fit(Xtr, ytr, feat_cols, cfg)
    Xtr182 = pre.transform(Xtr)

    learners = build_base_learners(args.seed)
    for name, est in learners.items():
        print(f"[train] {name} on ({Xtr182.shape})...")
        est.fit(Xtr182, ytr)

    blocks = []
    for name in ("random_forest", "svm_rbf", "xgboost", "mlp"):
        if name in learners:
            blocks.append(proba_to_meta_block(learners[name].predict_proba(Xtr182), False))
    meta_X = np.concatenate(blocks, axis=1)
    nam = train_nam(meta_X, ytr, seed=args.seed, epochs=args.nam_epochs, device=args.device)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"preprocessor": pre, "feature_cols": feat_cols}, out / "preprocess.joblib")
    joblib.dump(learners, out / "base_learners.pkl")
    torch.save({"model_state_dict": nam.state_dict(),
                "num_features": meta_X.shape[1], "num_classes": 2,
                "config": {"hidden_dim": 64, "num_layers": 2, "dropout": 0.1},
                "use_logit": False}, out / "nam_meta_learner.pt")
    (out / "feature_schema.json").write_text(json.dumps(
        {"feature_columns": feat_cols}, indent=2))
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
