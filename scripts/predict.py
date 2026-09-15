#!/usr/bin/env python3
"""Predict VF probability for new samples.

Friendly wrapper: translates stable CLI flags to src/deepgvs/prediction.py.

Usage:
  python scripts/predict.py --model-dir model/dataset_a \
      --protein-fasta example/example_protein.fasta --cds-fasta example/example_cds.fasta \
      --pdb-dir Dataset/example/PDB --out results/prediction.csv
  python scripts/predict.py --model-dir model/dataset_a --merged-features features.csv --out results/prediction.csv
  python scripts/predict.py --model-dir model/dataset_a --features182 features182.csv --out results/prediction.csv
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    ap = argparse.ArgumentParser(description="DeepGVS VF prediction (friendly wrapper).")
    ap.add_argument("--model-dir", default="model/dataset_a")
    ap.add_argument("--protein-fasta", default="")
    ap.add_argument("--cds-fasta", default="")
    ap.add_argument("--merged-features", default="")
    ap.add_argument("--features182", default="")
    ap.add_argument("--protein-table", default="")
    ap.add_argument("--protein-mode", default="compute", choices=("auto", "csv", "compute"))
    ap.add_argument("--pdb-dir", default="")
    ap.add_argument("--pdb-root", default="")
    ap.add_argument("--graph-mamba-config", default="configs/graph_mamba_infer.json")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--inference-device", default="cpu")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="results/prediction.csv")
    ap.add_argument("--skip-model-check", action="store_true")
    args = ap.parse_args()

    argv = ["prediction.py", "-m", args.model_dir]
    if args.merged_features:
        argv += ["--merged-features", args.merged_features]
    if args.features182:
        argv += ["--features182", args.features182]
    if args.protein_fasta:
        argv += ["-pseq", args.protein_fasta]
    if args.cds_fasta:
        argv += ["-cds", args.cds_fasta]
    if args.protein_table:
        argv += ["--protein-features", args.protein_table]
    argv += ["--protein-mode", args.protein_mode]
    if args.pdb_dir:
        argv += ["--pdb-dir", args.pdb_dir]
    if args.pdb_root:
        argv += ["--pdb-root", args.pdb_root]
    if args.graph_mamba_config:
        argv += ["--graph-mamba-config", args.graph_mamba_config]
    argv += ["--gm-device", args.device, "--inference-device", args.inference_device,
             "-o", args.out, "-p", str(args.threshold)]
    if args.skip_model_check:
        argv.append("--skip-model-check")
    sys.argv = argv
    runpy.run_module("deepgvs.prediction", run_name="__main__")


if __name__ == "__main__":
    main()
