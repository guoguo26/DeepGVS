#!/usr/bin/env python3
"""RF / SVM / XGBoost / MLP 基学习器加载与 stacking 推理。"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import torch

from nam import FittedSingleStreamPreprocessor, NeuralAdditiveModel, proba_to_meta_block

def _patch_sklearn_13_tree(est: Any) -> None:
    """Allow sklearn 1.3.x pickles under sklearn 1.4+."""
    if hasattr(est, "estimators_"):
        for sub in est.estimators_:
            _patch_sklearn_13_tree(sub)
    if est.__class__.__name__ == "DecisionTreeClassifier" and not hasattr(est, "monotonic_cst"):
        est.monotonic_cst = None


def _patch_pipeline(pipe: Any) -> Any:
    _patch_sklearn_13_tree(pipe)
    final = pipe.named_steps.get("model", pipe)
    _patch_sklearn_13_tree(final)
    return pipe


LEARNER_FILES: List[Tuple[str, str]] = [
    ("rf.pkl", "random_forest"),
    ("svm.pkl", "svm_rbf"),
    ("xgboost.pkl", "xgboost"),
    ("mlp.pkl", "mlp"),
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_preprocess_bundle(model_dir: Path) -> Tuple[FittedSingleStreamPreprocessor, List[str]]:
    import sys

    import nam as _nam_mod

    sys.modules.setdefault("predict_stacking", _nam_mod)
    main_mod = sys.modules.get("__main__")
    if main_mod is not None:
        for name in ("FittedSingleStreamPreprocessor", "SingleStreamPreprocessConfig"):
            setattr(main_mod, name, getattr(_nam_mod, name))

    joblib_path = model_dir / "preprocess.joblib"
    if not joblib_path.is_file():
        joblib_path = model_dir / "single_stream_preprocess.joblib"
    if not joblib_path.is_file():
        raise FileNotFoundError(f"Missing preprocess.joblib under {model_dir}")
    bundle = joblib.load(joblib_path)
    pre: FittedSingleStreamPreprocessor = bundle["preprocessor"]
    feature_cols: List[str] = list(bundle["feature_cols"])
    return pre, feature_cols


def expected_base_input_dim(model_dir: Path) -> int:
    """Input dimension expected by exported RF/SVM/XGB/MLP (after preprocess)."""
    learners = load_base_learners(model_dir)
    est = learners["random_forest"]
    final = est.named_steps.get("model", est) if hasattr(est, "named_steps") else est
    n = getattr(final, "n_features_in_", None)
    if n is None:
        raise AttributeError("Could not read n_features_in_ from random_forest model.")
    return int(n)


def list_missing_model_files(model_dir: Path) -> List[str]:
    """Return human-readable names of required artifacts missing under ``model_dir``."""
    model_dir = Path(model_dir)
    missing: List[str] = []
    if not (model_dir / "feature_schema.json").is_file():
        missing.append("feature_schema.json")
    if not any((model_dir / p).is_file() for p in ("preprocess.joblib", "single_stream_preprocess.joblib")):
        missing.append("preprocess.joblib (or single_stream_preprocess.joblib)")
    has_split_learners = all((model_dir / fname).is_file() for fname, _ in LEARNER_FILES)
    if not has_split_learners and not (model_dir / "base_learners.pkl").is_file():
        missing.append("rf.pkl, svm.pkl, xgboost.pkl, mlp.pkl (or legacy base_learners.pkl)")
    if not any((model_dir / p).is_file() for p in ("nam.pt", "nam_meta_learner.pt")):
        missing.append("nam.pt (or nam_meta_learner.pt)")
    return missing


def assert_model_files(model_dir: Path) -> None:
    missing = list_missing_model_files(model_dir)
    if missing:
        raise FileNotFoundError(
            f"Incomplete model directory {model_dir}. Missing:\n  - "
            + "\n  - ".join(missing)
        )


def validate_model_bundle(model_dir: Path) -> int:
    """
    Ensure required model files exist and preprocess.joblib matches base learners.
    Returns the shared preprocess output dim (typically 182).
    """
    model_dir = Path(model_dir)
    assert_model_files(model_dir)
    pre, feature_cols = load_preprocess_bundle(model_dir)
    expected = expected_base_input_dim(model_dir)
    out_dim = int(pre.final_dim_)
    if out_dim != expected:
        raise ValueError(
            f"preprocess.joblib outputs {out_dim} features but base learners expect {expected}. "
            f"Re-export preprocess on the training merge table (train,val) with the same "
            f"MI/PCA settings as stacking, then save to {model_dir / 'preprocess.joblib'}."
        )
    schema_cols = load_feature_schema_cols(model_dir)
    if len(feature_cols) != len(schema_cols):
        raise ValueError(
            f"preprocess.joblib has {len(feature_cols)} feature_cols; "
            f"feature_schema.json has {len(schema_cols)}."
        )
    return out_dim


def load_feature_schema_cols(model_dir: Path) -> List[str]:
    with (Path(model_dir) / "feature_schema.json").open(encoding="utf-8") as f:
        return list(json.load(f)["feature_columns"])


def load_base_learners(model_dir: Path) -> Dict[str, Any]:
    learners: Dict[str, Any] = {}
    for fname, key in LEARNER_FILES:
        p = model_dir / fname
        if p.is_file():
            with p.open("rb") as f:
                learners[key] = _patch_pipeline(pickle.load(f))
            continue
    legacy = model_dir / "base_learners.pkl"
    if legacy.is_file():
        raw = pickle.load(legacy.open("rb"))
        if isinstance(raw, dict):
            for _, key in LEARNER_FILES:
                if key in raw:
                    learners[key] = _patch_pipeline(raw[key])
    if not learners:
        raise FileNotFoundError(
            f"No base learners found in {model_dir}. Expected rf.pkl, svm.pkl, xgboost.pkl, mlp.pkl."
        )
    return learners


def load_nam(model_dir: Path, device: torch.device) -> Tuple[NeuralAdditiveModel, Dict[str, Any]]:
    for name in ("nam.pt", "nam_meta_learner.pt"):
        ckpt_path = model_dir / name
        if ckpt_path.is_file():
            break
    else:
        raise FileNotFoundError(f"Missing nam.pt under {model_dir}")
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model = NeuralAdditiveModel(
        num_features=int(ckpt["num_features"]),
        num_classes=int(ckpt["num_classes"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_layers=int(cfg["num_layers"]),
        dropout=float(cfg["dropout"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


def meta_use_logit(model_dir: Path) -> bool:
    cfg_json = model_dir / "inference_preprocess_config.json"
    if cfg_json.is_file():
        with cfg_json.open(encoding="utf-8") as f:
            j = json.load(f)
        return bool(j.get("meta_use_logit", False))
    return False


def preprocess_raw_features(
    X_raw: np.ndarray,
    model_dir: Path,
    pre: Optional[FittedSingleStreamPreprocessor] = None,
    feature_cols: Optional[Sequence[str]] = None,
) -> np.ndarray:
    if pre is None or feature_cols is None:
        pre, feature_cols = load_preprocess_bundle(model_dir)
    if X_raw.shape[1] != len(feature_cols):
        raise ValueError(f"Raw features {X_raw.shape[1]} != expected {len(feature_cols)}")
    return pre.transform(X_raw)


def base_learner_probas(
    X182: np.ndarray,
    learners: Dict[str, Any],
    *,
    use_logit: bool = False,
) -> List[np.ndarray]:
    order = [k for k in ("random_forest", "svm_rbf", "xgboost", "mlp") if k in learners]
    blocks: List[np.ndarray] = []
    for name in order:
        prob = learners[name].predict_proba(X182)
        blocks.append(proba_to_meta_block(prob, use_logit))
    return blocks


def predict_proba(
    X_raw: np.ndarray,
    model_dir: Path,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Raw merged features (N, 2146) -> class probabilities (N, 2)."""
    model_dir = Path(model_dir)
    dev = device or torch.device("cpu")
    pre, feature_cols = load_preprocess_bundle(model_dir)
    X182 = preprocess_raw_features(X_raw, model_dir, pre, feature_cols)
    learners = load_base_learners(model_dir)
    nam, ckpt = load_nam(model_dir, dev)
    blocks = base_learner_probas(X182, learners, use_logit=meta_use_logit(model_dir))
    meta_X = np.concatenate(blocks, axis=1)
    if meta_X.shape[1] != int(ckpt["num_features"]):
        raise ValueError(
            f"Meta feature dim {meta_X.shape[1]} != NAM expects {ckpt['num_features']}"
        )
    with torch.no_grad():
        logits = nam(torch.from_numpy(meta_X).float().to(dev))
        probas = torch.softmax(logits, dim=1).cpu().numpy()
    return probas


def predict_labels_from_preprocessed(
    X_preprocessed: np.ndarray,
    model_dir: Path,
    threshold: float = 0.5,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run stacking when features are already preprocessed (e.g. 182-dim)."""
    model_dir = Path(model_dir)
    dev = device or torch.device("cpu")
    expected = expected_base_input_dim(model_dir)
    if X_preprocessed.shape[1] != expected:
        raise ValueError(
            f"Preprocessed features have {X_preprocessed.shape[1]} columns; expected {expected}."
        )
    learners = load_base_learners(model_dir)
    nam, ckpt = load_nam(model_dir, dev)
    use_logit = meta_use_logit(model_dir)
    blocks: List[np.ndarray] = []
    for name in [k for k in ("random_forest", "svm_rbf", "xgboost", "mlp") if k in learners]:
        prob = learners[name].predict_proba(X_preprocessed)
        blocks.append(proba_to_meta_block(prob, use_logit))
    meta_X = np.concatenate(blocks, axis=1)
    if meta_X.shape[1] != int(ckpt["num_features"]):
        raise ValueError(
            f"Meta feature dim {meta_X.shape[1]} != NAM expects {ckpt['num_features']}"
        )
    with torch.no_grad():
        logits = nam(torch.from_numpy(meta_X).float().to(dev))
        probas = torch.softmax(logits, dim=1).cpu().numpy()
    p_vf = probas[:, 1] if probas.shape[1] > 1 else probas[:, 0]
    labels = (p_vf >= float(threshold)).astype(np.int64)
    return p_vf, labels


def predict_labels(
    X_raw: np.ndarray,
    model_dir: Path,
    threshold: float = 0.5,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    probas = predict_proba(X_raw, model_dir, device=device)
    p_vf = probas[:, 1] if probas.shape[1] > 1 else probas[:, 0]
    labels = (p_vf >= float(threshold)).astype(np.int64)
    return p_vf, labels
