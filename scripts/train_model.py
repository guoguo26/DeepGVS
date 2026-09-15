#!/usr/bin/env python3
"""Train DeepGVS stacking model (thin wrapper over src/deepgvs/train.py).

Usage:
  python scripts/train_model.py --train-csv data/Dataset_A/train.csv --out-dir model/dataset_a --seed 42
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if __name__ == "__main__":
    runpy.run_module("deepgvs.train", run_name="__main__")
