#!/usr/bin/env python3
"""Prepare DeepGVS datasets: validate paired FASTA, write ID splits and stats.

Usage:
  python scripts/prepare_dataset.py --protein-fasta Dataset/Dataset_A/protein/train.fasta \
      --cds-fasta Dataset/Dataset_A/CDS/train.fasta --out-dir data/Dataset_A --split train
  python scripts/prepare_dataset.py --from-legacy --out-root data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deepgvs.dataset import pair_fasta_records, write_ids  # noqa: E402
from deepgvs.protein_feature import canonical_sequence_id  # noqa: E402
from deepgvs.cds_feature import read_fasta_first_token_as_id  # noqa: E402


def prepare_split(protein_fasta: Path, cds_fasta: Path, out_dir: Path, split: str) -> dict:
    pairs = pair_fasta_records(protein_fasta, cds_fasta)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = [canonical_sequence_id(s) for s, _, _ in pairs]
    write_ids(ids, out_dir / f"{split}_ids.txt")
    stats = {
        "split": split,
        "n_paired": len(pairs),
        "protein_fasta": str(protein_fasta),
        "cds_fasta": str(cds_fasta),
    }
    print(f"[{split}] paired={len(pairs)} -> {out_dir / f'{split}_ids.txt'}")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare DeepGVS dataset splits.")
    ap.add_argument("--protein-fasta", default="")
    ap.add_argument("--cds-fasta", default="")
    ap.add_argument("--out-dir", default="data/Dataset_A")
    ap.add_argument("--split", default="train")
    ap.add_argument("--from-legacy", action="store_true",
                    help="Walk Dataset/Dataset_A|B legacy FASTA and emit data/*/<split>_ids.txt")
    ap.add_argument("--out-root", default="data")
    args = ap.parse_args()

    if args.from_legacy:
        legacy = {
            "Dataset_A": {"train": ("protein/train.fasta", "CDS/train.fasta"),
                          "validation": ("protein/val.fasta", "CDS/val.fasta"),
                          "test": ("protein/test.fasta", "CDS/test.fasta")},
            "Dataset_B": {"train": ("protein/dataSet.fasta", "CDS/dataSet.fasta"),
                          "validation": ("protein/ind.fasta", "CDS/ind.fasta")},
        }
        for ds, splits in legacy.items():
            for split, (p_rel, c_rel) in splits.items():
                pf = ROOT / "Dataset" / ds / p_rel
                cf = ROOT / "Dataset" / ds / c_rel
                if pf.is_file() and cf.is_file():
                    prepare_split(pf, cf, ROOT / args.out_root / ds, split)
                else:
                    print(f"[skip] missing {pf} or {cf}")
        return

    if not args.protein_fasta or not args.cds_fasta:
        ap.error("Provide --protein-fasta and --cds-fasta (or --from-legacy).")
    prepare_split(Path(args.protein_fasta), Path(args.cds_fasta), Path(args.out_dir), args.split)


if __name__ == "__main__":
    main()
