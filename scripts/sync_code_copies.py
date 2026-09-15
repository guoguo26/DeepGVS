#!/usr/bin/env python3
"""Sync canonical src/deepgvs/*.py into legacy code/*.py single-file copies.

The legacy `code/` tree keeps standalone copies for `python code/prediction.py`
style use. The canonical implementations live in src/deepgvs/. This script
copies a file and rewrites the repo-root path depth for the flat layout
(parents[2] -> parents[1]; utils.py and prediction.py are excluded by the
caller list below, see comments).

Usage:
  python scripts/sync_code_copies.py            # sync default file list
  python scripts/sync_code_copies.py --check    # report drift only, exit 1 if any
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "deepgvs"
DST = ROOT / "code"

# Files that are layout-dependent and must NOT be blindly copied:
#   prediction.py   -> legacy entry uses relative "model/" default, src uses parents[2]
#   utils.py        -> parents[2] is correct for src layout only
SYNC_FILES = [
    "model.py",
    "protein_feature.py",
    "cds_feature.py",
    "nam.py",
    "dataset.py",
    "feature_extraction.py",
    "pdb_graph.py",
    "graph_mamba_model.py",
    "graph_mamba_dataset.py",
    "graph_structure_layers.py",
    "esm2_feature_extractor.py",
    "bi_mamba_gcn_encoder.py",
]


def transform(text: str, name: str) -> str:
    if name in ("model.py", "protein_feature.py"):
        # repo root from src/deepgvs/<f> is parents[2]; from code/<f> it is parents[1]
        text = text.replace(
            "return Path(__file__).resolve().parents[2]",
            "return Path(__file__).resolve().parents[1]",
        )
    return text


def main() -> int:
    check = "--check" in sys.argv
    drift = 0
    for name in SYNC_FILES:
        s, d = SRC / name, DST / name
        if not s.is_file():
            print(f"skip (src missing): {name}")
            continue
        want = transform(s.read_text(encoding="utf-8"), name)
        have = d.read_text(encoding="utf-8") if d.is_file() else ""
        if want == have:
            print(f"same: {name}")
            continue
        drift += 1
        if check:
            print(f"DRIFT: {name}")
        else:
            d.write_text(want, encoding="utf-8")
            print(f"synced: {name}")
    if check and drift:
        print(f"{drift} file(s) out of sync; run without --check")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
