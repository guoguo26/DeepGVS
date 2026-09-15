#!/usr/bin/env python3
"""Extract DeepGVS features: CDS (354-dim) + Graph-Mamba protein (1792-dim) -> merged 2146-dim CSV.

Usage:
  python scripts/extract_features.py --protein-fasta example/example_protein.fasta \
      --cds-fasta example/example_cds.fasta --model-dir model/dataset_a \
      --pdb-dir Dataset/example/PDB --out features.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deepgvs.dataset import pair_fasta_records  # noqa: E402
from deepgvs.feature_extraction import build_raw_feature_matrix  # noqa: E402
from deepgvs.protein_feature import load_feature_schema  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract DeepGVS 2146-dim features.")
    ap.add_argument("--protein-fasta", required=True)
    ap.add_argument("--cds-fasta", required=True)
    ap.add_argument("--model-dir", default="model/dataset_a")
    ap.add_argument("--protein-table", default="")
    ap.add_argument("--protein-mode", default="compute", choices=("auto", "csv", "compute"))
    ap.add_argument("--pdb-dir", default="")
    ap.add_argument("--graph-mamba-config", default="")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="features.csv")
    args = ap.parse_args()

    pairs = pair_fasta_records(args.protein_fasta, args.cds_fasta)
    print(f"[1/2] paired {len(pairs)} sample(s)")
    X, ids = build_raw_feature_matrix(
        pairs, Path(args.model_dir),
        Path(args.protein_table) if args.protein_table else None,
        protein_mode=args.protein_mode,
        pdb_dir=Path(args.pdb_dir) if args.pdb_dir else None,
        graph_mamba_config=Path(args.graph_mamba_config) if args.graph_mamba_config else None,
        device=args.device,
    )
    schema = load_feature_schema(Path(args.model_dir))
    df = pd.DataFrame(X, columns=list(schema["feature_columns"]))
    df.insert(0, "sequence_id", ids)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[2/2] wrote {out} shape={df.shape}")


if __name__ == "__main__":
    main()
