"""DeepGVS feature extraction facade: CDS (354-dim) + Graph-Mamba (1792-dim)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from cds_feature import compute_feature_row
from nam import split_protein_dna_feature_cols
from protein_feature import (
    canonical_sequence_id,
    compute_protein_features_online,
    load_feature_schema,
    lookup_or_compute_protein_features,
    merge_cds_and_protein_rows,
    protein_feature_columns,
)

RAW_DIM = 2146
PROTEIN_DIM = 1792
CDS_DIM = 354


def cds_feature_columns(schema: dict | None = None, model_dir: Path | str | None = None) -> List[str]:
    if schema is None:
        schema = load_feature_schema(Path(model_dir) if model_dir else None)
    _, dna_cols = split_protein_dna_feature_cols(list(schema["feature_columns"]))
    return list(dna_cols)


def compute_cds_dataframe(
    pairs: Sequence[Tuple[str, str, str]],
    dna_cols: Sequence[str],
    *,
    local_window_size: int = 128,
    local_window_step: int = 128,
    orf_max_len_nt: int = 8192,
) -> pd.DataFrame:
    rows = [
        compute_feature_row(
            seq_id=sid,
            seq=cds_seq,
            feat_cols=list(dna_cols),
            local_window_size=int(local_window_size),
            local_window_step=int(local_window_step),
            orf_max_len_nt=int(orf_max_len_nt),
        )
        for sid, _pseq, cds_seq in pairs
    ]
    return pd.DataFrame(rows)


def compute_protein_matrix(
    sequence_ids: Sequence[str],
    protein_cols: Sequence[str],
    *,
    mode: str = "compute",
    protein_table: Path | str | None = None,
    pdb_dir: Path | str | None = None,
    pdb_root: Path | str | None = None,
    graph_mamba_config: Path | str | None = None,
    device: str = "auto",
) -> np.ndarray:
    return lookup_or_compute_protein_features(
        list(sequence_ids),
        list(protein_cols),
        mode=mode,
        protein_table=Path(protein_table) if protein_table else None,
        pdb_dir=Path(pdb_dir) if pdb_dir else None,
        pdb_root=Path(pdb_root) if pdb_root else None,
        graph_mamba_config=Path(graph_mamba_config) if graph_mamba_config else None,
        device=device,
    )


def build_raw_feature_matrix(
    pairs: Sequence[Tuple[str, str, str]],
    model_dir: Path | str,
    protein_table: Path | str | None = None,
    *,
    protein_mode: str = "compute",
    pdb_dir: Path | str | None = None,
    pdb_root: Path | str | None = None,
    graph_mamba_config: Path | str | None = None,
    device: str = "auto",
) -> Tuple[np.ndarray, List[str]]:
    """pairs -> (N, 2146) raw matrix + ids (canonical order = schema column order)."""
    model_dir_p = Path(model_dir)
    schema = load_feature_schema(model_dir_p)
    feature_cols: List[str] = list(schema["feature_columns"])
    _, dna_cols = split_protein_dna_feature_cols(feature_cols)
    prot_cols = protein_feature_columns(schema)
    ids = [canonical_sequence_id(sid) for sid, _, _ in pairs]

    cds_df = compute_cds_dataframe(
        pairs,
        dna_cols,
        local_window_size=int(schema.get("local_window_size", 128)),
        local_window_step=int(schema.get("local_window_step", 128)),
        orf_max_len_nt=int(schema.get("orf_max_len_nt", 8192)),
    )
    prot_mat = compute_protein_matrix(
        ids,
        prot_cols,
        mode=protein_mode,
        protein_table=protein_table,
        pdb_dir=pdb_dir,
        pdb_root=pdb_root,
        graph_mamba_config=graph_mamba_config,
        device=device,
    )
    prot_part = pd.DataFrame(np.asarray(prot_mat), columns=pd.Index(prot_cols, dtype="string"))
    prot_part = prot_part.assign(sequence_id=pd.Series(ids, dtype="string"))
    merged = merge_cds_and_protein_rows(cds_df, prot_part, feature_cols)
    return merged[list(feature_cols)].values.astype(np.float32), ids


__all__ = [
    "RAW_DIM",
    "PROTEIN_DIM",
    "CDS_DIM",
    "cds_feature_columns",
    "compute_cds_dataframe",
    "compute_protein_matrix",
    "compute_protein_features_online",
    "build_raw_feature_matrix",
    "canonical_sequence_id",
]
