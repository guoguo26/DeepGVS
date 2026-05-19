#!/usr/bin/env python3
"""Check that a DeepGVS model directory is complete and internally consistent."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from model import assert_model_files, list_missing_model_files, repo_root, validate_model_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify DeepGVS model/ release bundle.")
    parser.add_argument("-m", "--model_path", type=str, default="model")
    args = parser.parse_args()

    model_dir = (repo_root() / args.model_path).resolve()
    manifest_path = model_dir / "manifest.json"
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as f:
            manifest = json.load(f)
        print(f"[manifest] {manifest.get('description', '')}")

    missing = list_missing_model_files(model_dir)
    if missing:
        print("[FAIL] missing files:")
        for m in missing:
            print(f"  - {m}")
        raise SystemExit(1)

    dim = validate_model_bundle(model_dir)
    print(f"[OK] {model_dir}")
    print(f"     raw_features=2146 preprocess_out={dim} meta_input=4 (NAM)")
    print("     ready for: python code/prediction.py -m", args.model_path)


if __name__ == "__main__":
    main()
