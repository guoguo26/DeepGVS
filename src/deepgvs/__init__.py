"""DeepGVS package init: exposes canonical submodules with path fallback.

Supports both layouts:
  - GitHub canonical: ``src/deepgvs/*.py`` imported as ``deepgvs.*``
  - Legacy flat: ``code/*.py`` with ``from nam import ...`` style imports
"""
from __future__ import annotations

import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

__version__ = "1.0.0"
__all__ = [
    "model",
    "nam",
    "dataset",
    "feature_extraction",
    "train",
    "predict",
    "utils",
    "cds_feature",
    "protein_feature",
    "pdb_graph",
    "graph_mamba_model",
    "graph_mamba_dataset",
    "graph_structure_layers",
    "bi_mamba_gcn_encoder",
    "esm2_feature_extractor",
]
