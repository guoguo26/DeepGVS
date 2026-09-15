"""DeepGVS inference and training entry points (import-safe wrappers)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

import model as _model


def validate_bundle(model_dir: Path | str) -> int:
    return _model.validate_model_bundle(Path(model_dir))


def predict_from_raw(
    X_raw: np.ndarray,
    model_dir: Path | str,
    threshold: float = 0.5,
    device: Optional[torch.device | str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    dev = torch.device(device) if isinstance(device, str) else device
    return _model.predict_labels(np.asarray(X_raw), Path(model_dir), threshold=threshold, device=dev)


def predict_from_preprocessed(
    X: np.ndarray,
    model_dir: Path | str,
    threshold: float = 0.5,
    device: Optional[torch.device | str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    dev = torch.device(device) if isinstance(device, str) else device
    return _model.predict_labels_from_preprocessed(np.asarray(X), Path(model_dir), threshold=threshold, device=dev)


def expected_preprocessed_dim(model_dir: Path | str) -> int:
    return _model.expected_base_input_dim(Path(model_dir))
