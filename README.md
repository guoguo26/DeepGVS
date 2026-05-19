# DeepGVS

DeepGVS predicts whether bacterial genes encode **virulence factors (VF)**. The pipeline combines **354-dimensional CDS sequence features** with **1792-dimensional Graph-Mamba protein structure embeddings** (2146 raw features total). After mutual-information selection and graph PCA, four base learners (**RF, SVM, XGBoost, MLP**) feed a **NAM** meta-learner that outputs the VF probability.

All commands below assume you are in the **`DeepGVS/`** directory:

```bash
cd DeepGVS
```

---

## Quick start

```bash
pip install -r requirements.txt

# Check the Dataset A model bundle
python code/verify_model.py -m model/dataset_a/

# Example: paired FASTA + PDB (structure features computed on the fly)
python code/prediction.py \
  -m model/dataset_a/ \
  -pseq Dataset/example/example_protein.fasta \
  -cds Dataset/example/example_cds.fasta \
  --protein-mode compute \
  --pdb-dir Dataset/example/PDB \
  -o results/prediction.csv
```

Output columns: `Sample_ID`, `VF_probability`, `Prediction` (`VF` or `non-VF`; threshold default `0.5`, change with `-p`).

For Graph-Mamba on-the-fly inference, install **fair-esm** and **mamba-ssm** (matching your PyTorch/CUDA build) and point `model/graph_mamba_infer.json` at your trained checkpoint and ProtT5 directory.

---

## Repository layout

```text
DeepGVS/
├── code/
│   ├── prediction.py          # Main entry: VF prediction
│   ├── verify_model.py        # Validate model/dataset_a/ files and dimensions
│   ├── model.py               # Stacking inference (preprocess + learners + NAM)
│   ├── nam.py                 # 2146→182 preprocessing and NAM definition
│   ├── cds_feature.py         # CDS → 354-dim DNA features
│   ├── protein_feature.py     # Graph-Mamba lookup or PDB inference
│   ├── graph_mamba_model.py   # Graph-Mamba network
│   ├── graph_mamba_dataset.py # PDB → kNN graph → ESM-2 node features
│   ├── pdb_graph.py           # PDB parsing, kNN graph (k=30, binary edges)
│   ├── esm2_feature_extractor.py
│   ├── bi_mamba_gcn_encoder.py
│   ├── dump_preprocess.py     # Refit preprocess.joblib (needs external feature tables)
│   └── test.py                # Evaluate on a prebuilt test tensor
│
├── model/
│   ├── dataset_a/             # Default release bundle (use -m model/dataset_a/)
│   ├── dataset_b/             # Dataset B stacking weights only (incomplete)
│   ├── graph_mamba_infer.json # On-the-fly Graph-Mamba config
│   └── README.md
│
├── Dataset/
│   ├── example/               # Demo protein/CDS FASTA + one PDB
│   ├── Dataset_A/             # Train / val / test FASTA splits (optional)
│   └── Dataset_B/
│
├── results/                   # Default prediction output
└── requirements.txt
```

**Not shipped in this repository**

| Item | Notes |
|------|--------|
| `Feature/` training tables | Large CDS / Graph-Mamba CSVs used only to **retrain** or refit `preprocess.joblib`; keep them on disk elsewhere and pass paths to `dump_preprocess.py`. |
| Precomputed protein lookup CSV | Optional; provide via `--protein-features` if you use `--protein-mode csv`. |
| Merged 2146-dim feature CSV | Any path; columns must match `model/dataset_a/feature_schema.json`. |

---

## Inference pipeline

```text
Input (exactly one mode)
  ① Merged CSV (2146 cols)   --merged-features
  ② Protein + CDS FASTA     -pseq + -cds
  ③ Preprocessed CSV (182)    --features182
              │
              ▼
  feature_schema.json  column order (modes ①②)
              │
              ▼
  preprocess.joblib  2146 → 182  (skipped in ③)
              │
              ▼
  base_learners.pkl  → 4 positive-class probabilities
              │
              ▼
  nam_meta_learner.pt  → VF probability
              │
              ▼
  results/prediction.csv
```

| Stage | Dimension |
|-------|-----------|
| Raw merged features | 2146 (1792 protein + 354 CDS) |
| After preprocess | 182 |
| Base learners → meta | 4 × 1 |
| NAM output | 1 probability |

---

## Input modes

### ① Merged feature CSV

Use when you already have all **2146** numeric features aligned with training.

```bash
python code/prediction.py \
  -m model/dataset_a/ \
  --merged-features /path/to/features.csv \
  -o results/prediction.csv
```

Required: column `sequence_id` plus every name listed in `model/dataset_a/feature_schema.json`.

### ② Paired protein FASTA + CDS FASTA

CDS features are computed in Python; protein structure features come from a lookup table and/or PDB.

```bash
python code/prediction.py \
  -m model/dataset_a/ \
  -pseq /path/to/proteins.fasta \
  -cds /path/to/cds.fasta \
  -o results/prediction.csv
```

- Header format: `>sample_id ...`; the **first token** after `>` is the ID (must match between protein and CDS; `|` → `_`).
- Protein sequence: amino acids; CDS: nucleotides.

**`--protein-mode`**

| Mode | Behavior |
|------|----------|
| `compute` | Graph-Mamba from PDB only |
| `csv` | Lookup in `--protein-features` only (error if ID missing) |
| `auto` (default) | Lookup if table given and ID found; otherwise PDB |

**Lookup table** (optional):

```bash
python code/prediction.py \
  -m model/dataset_a/ \
  -pseq Dataset/example/example_protein.fasta \
  -cds Dataset/example/example_cds.fasta \
  --protein-mode csv \
  --protein-features /path/to/graph_mamba_features.csv \
  -o results/prediction.csv
```

The table must include `sequence_id` and columns `graph_mamba_feat_0` … `graph_mamba_feat_1791`.

**On-the-fly Graph-Mamba** (included example):

1. Edit `model/graph_mamba_infer.json` (`checkpoint`, `global_prott5_dir`).
2. Place `{sequence_id}.pdb` under `--pdb-dir` (see `Dataset/example/PDB/`).
3. Install fair-esm, mamba-ssm, and a compatible PyTorch build.

```bash
python code/prediction.py \
  -m model/dataset_a/ \
  -pseq Dataset/example/example_protein.fasta \
  -cds Dataset/example/example_cds.fasta \
  --protein-mode compute \
  --pdb-dir Dataset/example/PDB \
  --gm-device cuda:0 \
  -o results/prediction.csv
```

Training-time Graph-Mamba settings: **kNN k=30**, **binary** edge weights, **GAT**, **multi_attn** pooling, **gated** fusion, **late ProtT5 concat**.

### ③ Preprocessed 182-dim CSV

Skips `preprocess.joblib` and runs base learners + NAM directly.

```bash
python code/prediction.py \
  -m model/dataset_a/ \
  --features182 /path/to/features182.csv \
  -o results/prediction.csv
```

---

## CLI reference

```bash
python code/prediction.py -h
```

| Option | Default | Description |
|--------|---------|-------------|
| `-m` | `model` | Model directory; use **`model/dataset_a/`** |
| `-o` | `results/prediction.csv` | Output CSV |
| `-p` | `0.5` | VF threshold |
| `--inference-device` | `cpu` | NAM device: `cpu`, `cuda:0`, `auto` |
| `--gm-device` | `auto` | Graph-Mamba device (`compute` / `auto`) |
| `--graph-mamba-config` | `model/graph_mamba_infer.json` | Override config JSON |
| `--skip-model-check` | off | Skip preprocess vs learner dimension check |

Model bundle details: [model/README.md](model/README.md).

---

## Refitting `preprocess.joblib`

Only needed if you retrain or replace base learners. Training feature CSVs are **not** in this repo; pass absolute paths:

```bash
python code/dump_preprocess.py \
  --cds-features /path/to/cds_features_all.csv \
  --structural-features /path/to/graph_mamba_features.csv \
  --output-joblib model/dataset_a/preprocess.joblib
```

The merged table must contain columns `split`, `label`, and all 2146 feature names. Defaults: MI `k=700`, `graph_pca_var=0.95`.

Alternatively, pass one merged file:

```bash
python code/dump_preprocess.py \
  --merged-features /path/to/merged_2146.csv \
  --output-joblib model/dataset_a/preprocess.joblib
```

---

## Dependencies

**Core (stacking + CDS features + NAM)**

```bash
pip install -r requirements.txt
```

| Package | Notes |
|---------|--------|
| Python | ≥ 3.9 |
| scikit-learn | 1.3.x (`<1.4`); weights exported under 1.3 |
| PyTorch | ≥ 2.0; GPU wheel from [pytorch.org](https://pytorch.org) if needed |
| XGBoost, pandas, numpy, scipy, joblib | See `requirements.txt` |

**Optional (FASTA + `--protein-mode compute`)**

- `fair-esm` ≥ 2.0  
- `mamba-ssm` (CUDA/torch version must match training)  
- `einops`  

Use the same environment as Graph-Mamba training when loading checkpoints.

---

## Other commands

```bash
python code/verify_model.py -m model/dataset_a/

python code/test.py \
  -m model/dataset_a/ \
  -i /path/to/test_data.pt \
  -o results/test_eval.csv
```

---

## FAQ

**`verify_model` reports preprocess vs RF dimension mismatch**  
`preprocess.joblib` and `base_learners.pkl` are from different training runs. Refit preprocess on the correct merged training table.

**`-m model` fails with missing files**  
Use **`-m model/dataset_a/`** when `model/` only contains subdirectories.

**FASTA mode: missing protein features**  
Supply `--protein-features`, or use `--protein-mode compute` / `auto` with PDB files and a valid `graph_mamba_infer.json`.

**CDS only, no protein**  
Not supported in FASTA mode; provide a full **2146-dim** merged CSV instead.

**Dataset A vs B**  
Do not mix `preprocess.joblib` / `feature_schema.json` across datasets. `model/dataset_b/` has stacking weights only.

---

## Citation

If you use DeepGVS in published work, please cite the relevant paper(s).
