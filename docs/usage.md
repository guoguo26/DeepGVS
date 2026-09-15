# Usage

All commands run from the repo root (`DeepGVS/`).

Dataset-specific preprocessing (feature transforms, column order) is resolved
automatically from each model bundle (`feature_schema.json` + `preprocess.joblib`);
no `--dataset` flag is needed — just point `--model-dir` at the right bundle.

## 1. Predict from paired FASTA + PDB (default workflow)

CDS features (354-dim) are computed in code; protein features (1792-dim Graph-Mamba
embeddings) are computed on the fly from PDB files — nothing is looked up from a table.

```bash
export DEEPGVS_GRAPH_MAMBA_CONFIG=/path/to/graph_mamba_infer.json

python scripts/predict.py \
  --model-dir model/dataset_a \
  --protein-fasta example/example_protein.fasta \
  --cds-fasta example/example_cds.fasta \
  --pdb-dir example/PDB \
  --out results/prediction.csv
```

Output columns: `Sample_ID,VF_probability,Prediction` (`VF`/`non-VF`, threshold `--threshold`, default `0.5`).
Compare with `example/expected_output.tsv`.

Notes:

- FASTA ID = first token after `>`; `|` is normalized to `_`. Only IDs present in **both**
  protein and CDS files are predicted; the rest are skipped with a warning.
- `example/PDB` ships one PDB (`VFG011519(gb_WP_002963669).pdb`) matching
  the single-record example FASTAs (mirrored from `Dataset/example/PDB/`).
- Needs `fair-esm` + `mamba-ssm` and the training checkpoint/ProtT5 paths in the JSON config.

## 2. Predict from a merged 2146-column CSV

Use when CDS + protein features were already combined offline (column order =
the bundle's `feature_schema.json`, plus `sequence_id`):

```bash
python scripts/predict.py \
  --model-dir model/dataset_b \
  --merged-features /path/to/features.csv \
  --out results/prediction_b.csv
```
## 3. Predict from a preprocessed CSV

Skips `preprocess.joblib`; the expected input dim is whatever the bundle reports
(use the dimension printed by `[1/3] model OK` at predict time):

```bash
python scripts/predict.py \
  --model-dir model/dataset_b \
  --features182 /path/to/features_preprocessed.csv \
  --out results/prediction_b.csv
```

## 4. Extract features without predicting

```bash
python scripts/extract_features.py \
  --protein-fasta example/example_protein.fasta \
  --cds-fasta example/example_cds.fasta \
  --model-dir model/dataset_a \
  --pdb-dir example/PDB \
  --out features.csv
```

## 5. Train / evaluate

```bash
python scripts/train_model.py --train-csv data/Dataset_A/train.csv --out-dir model/dataset_a --seed 42
python scripts/evaluate_model.py --model-dir model/dataset_a --test-csv data/Dataset_A/test.csv --out results/metrics.csv
```

Training CSVs need `sequence_id,label` + the 2146 feature columns. See `configs/` and
`docs/reproduction.md` for the manuscript data layout (large files live on Zenodo, not GitHub).
