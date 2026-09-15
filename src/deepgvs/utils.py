"""Shared DeepGVS utilities: paths, FASTA IO, seeding, env report."""
from __future__ import annotations

import platform
import random
import sys
from pathlib import Path

import numpy as np


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def read_fasta(path: Path | str) -> list[tuple[str, str]]:
    seqs: list[tuple[str, str]] = []
    cur_id: str | None = None
    cur: list[str] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    seqs.append((cur_id, "".join(cur)))
                cur_id = line[1:].strip().split()[0]
                cur = []
            else:
                cur.append(line)
    if cur_id is not None:
        seqs.append((cur_id, "".join(cur)))
    return seqs


def software_versions() -> str:
    import importlib
    lines = [f"python={platform.python_version()} ({sys.version.split()[0]})"]
    for mod in ("torch", "numpy", "pandas", "scikit-learn", "scikit_learn",
                "xgboost", "scipy", "joblib", "tqdm", "transformers"):
        try:
            m = importlib.import_module("sklearn" if mod in ("scikit-learn", "scikit_learn") else mod)
            lines.append(f"{mod}={getattr(m, '__version__', 'unknown')}")
        except Exception as e:
            lines.append(f"{mod}=not-installed ({e})")
    for mod in ("mamba_ssm", "esm"):
        try:
            m = importlib.import_module(mod)
            lines.append(f"{mod}={getattr(m, '__version__', 'unknown')}")
        except Exception:
            lines.append(f"{mod}=not-installed")
    try:
        import torch
        lines.append(f"cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"cuda_version={torch.version.cuda}")
    except Exception:
        pass
    return "\n".join(lines) + "\n"
