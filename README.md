# DeepGVS

DeepGVS predicts bacterial **virulence factors (VF)** from **354-dim CDS features** + **1792-dim Graph-Mamba structure embeddings** (2146 raw → 182 after preprocess). Stacking (**RF / SVM / XGBoost / MLP**) + **NAM** outputs VF probability.

Run from **`DeepGVS/`**:

```bash
cd DeepGVS
pip install -r requirements.txt
```

---

## Quick start

**FASTA + PDB** (CDS features computed in code; protein features from **Graph-Mamba on PDB**, not a lookup table):

```bash
export DEEPGVS_GRAPH_MAMBA_CONFIG=/path/to/your_graph_mamba_config.json

python code/prediction.py \
  -m model/dataset_a/ \
  -pseq Dataset/example/example_protein.fasta \
  -cds Dataset/example/example_cds.fasta \
  --pdb-dir Dataset/example/PDB \
  -o results/prediction.csv
```

The JSON must point to your trained **checkpoint** and **ProtT5** directory (same settings as training). Install **fair-esm** and **mamba-ssm** for this path.

**Or** use a ready-made **2146-column merged CSV** (CDS + protein features already combined):

```bash
python code/prediction.py \
  -m model/dataset_a/ \
  --merged-features /path/to/features.csv \
  -o results/prediction.csv
```

Output: `Sample_ID`, `VF_probability`, `Prediction` (`VF` / `non-VF`; threshold `-p`, default `0.5`).

---

## Layout

```text
DeepGVS/
├── code/prediction.py      # Main entry
├── model/dataset_a/        # Default bundle (-m model/dataset_a/)
├── model/dataset_b/        # B stacking weights only
├── Dataset/example/        # Demo FASTA + PDB
├── Dataset/Dataset_A|B/    # Train/val/test FASTA
└── requirements.txt
```

---

## Prediction inputs

**FASTA + PDB (default workflow)** — both feature blocks are computed **on the fly**; no feature CSV is read:

| Step | Input | Output |
|------|--------|--------|
| CDS | nucleotide FASTA (`-cds`) | 354-dim DNA features (`cds_feature.py`) |
| Protein structure | amino-acid FASTA + `{id}.pdb` (`-pseq`, `--pdb-dir`) | 1792-dim Graph-Mamba embeddings (`protein_feature.py`) |
| Merge + predict | 2146-dim vector | VF probability |

**Merged CSV** — use only if you already have all 2146 columns offline (not the FASTA pipeline).

**182-dim CSV** — preprocessed features; stacking only.

FASTA: ID = first token after `>` (match in protein/CDS; `|` → `_`).  
Graph-Mamba config: `--graph-mamba-config` or `DEEPGVS_GRAPH_MAMBA_CONFIG`.

Full options: `python code/prediction.py -h`

---

## Requirements

- Python ≥ 3.9, **scikit-learn 1.3.x** (`<1.4`), PyTorch ≥ 2.0 — `requirements.txt`
- FASTA + PDB path: **fair-esm**, **mamba-ssm** (match training CUDA/torch)


---

## Citation

If you use DeepGVS in published work, please cite the relevant publication(s).
