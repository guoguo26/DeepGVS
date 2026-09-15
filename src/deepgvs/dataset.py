"""DeepGVS dataset helpers: FASTA pairing, CSV loaders, ID canonicalization."""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from cds_feature import read_fasta_first_token_as_id
from protein_feature import canonical_sequence_id


def pair_fasta_records(
    protein_fasta: Path | str,
    cds_fasta: Path | str,
) -> List[Tuple[str, str, str]]:
    """Pair protein/CDS records by canonical sequence_id.

    Returns list of (sequence_id, protein_seq, cds_seq). Unmatched IDs raise
    ValueError if intersection is empty, otherwise they are dropped.
    """
    prot = read_fasta_first_token_as_id(Path(protein_fasta))
    cds = read_fasta_first_token_as_id(Path(cds_fasta))
    prot_d = {canonical_sequence_id(sid): seq for sid, seq in prot}
    cds_d = {canonical_sequence_id(sid): seq for sid, seq in cds}
    common = sorted(set(prot_d) & set(cds_d))
    if not common:
        raise ValueError("No matched sequence IDs between protein and CDS FASTA.")
    return [(sid, prot_d[sid], cds_d[sid]) for sid in common]


def load_merged_features_csv(path: Path | str, feature_cols: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    df = pd.read_csv(path)
    if "sequence_id" not in df.columns:
        raise ValueError(f"{path}: missing column 'sequence_id'")
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing {len(missing)} feature columns, e.g. {missing[:5]}")
    ids = df["sequence_id"].astype(str).tolist()
    return df[list(feature_cols)].values.astype(np.float32), ids


def write_ids(ids: Sequence[str], path: Path | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(str(s) for s in ids) + "\n", encoding="utf-8")


def read_ids(path: Path | str) -> List[str]:
    return [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
