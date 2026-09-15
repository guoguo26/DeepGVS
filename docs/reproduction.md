# Reproduction

This page describes how to reproduce the manuscript results. The exact manuscript
snapshot (code + models + data splits + software versions) is archived on Zenodo;
GitHub always holds the living (latest) code.

## 1. Get the manuscript snapshot

- Zenodo DOI: [`10.5281/zenodo.21756890`](https://zenodo.org/records/21756890)
- Unpack `DeepGVS-v1.0.0/` and use it instead of the GitHub checkout for reproduction.

## 2. Environment

Manuscript reference (record the exact output of `pip freeze` / `conda list` in
Zenodo `environment/software_versions.txt`):

- Python 3.10.18, PyTorch 2.1.0, CUDA 11.8, Transformers 4.30.0, random seed 42
- scikit-learn 1.3.x, fair-esm + mamba-ssm builds matching training CUDA/torch

```bash
conda env create -f environment.yml   # from the Zenodo record
conda activate deepgvs
```

## 3. Data layout (Zenodo)

```text
DeepGVS-v1.0.0/data/Dataset_A/{train,val,test}.fasta(+CDS/protein split as in configs/dataset_A.yaml)
DeepGVS-v1.0.0/data/Dataset_B/{dataSet,ind}.fasta
DeepGVS-v1.0.0/dataset_splits/*.txt   # canonical IDs; '|' -> '_' for Dataset B
DeepGVS-v1.0.0/models/{DeepGVS_DatasetA.pt,DeepGVS_DatasetB.pt}
```

`dataset_splits/` IDs in this repo snapshot were generated from the bundled FASTAs:

- `Dataset_A_train_ids.txt`: 4999, `Dataset_A_validation_ids.txt`: 800, `Dataset_A_test_ids.txt`: 1369
- `Dataset_B_train_ids.txt`: 6000 (dataSet), `Dataset_B_validation_ids.txt`: 1152 (ind)

## 4. Reproduce

```bash
bash reproducibility/reproduce_results.sh
```

which runs (see the script in the Zenodo record):

1. `python scripts/prepare_dataset.py --from-legacy --out-root data`
2. `python scripts/extract_features.py ...` per split (needs Graph-Mamba config + PDBs)
3. `python scripts/train_model.py --seed 42 ...`
4. `python scripts/evaluate_model.py ...` and diff `example/expected_output.tsv`

## 5. Minimal smoke test (GitHub checkout, no large files)

```bash
export DEEPGVS_GRAPH_MAMBA_CONFIG=/path/to/graph_mamba_infer.json
python scripts/predict.py --model-dir model/dataset_a \
  --protein-fasta example/example_protein.fasta \
  --cds-fasta example/example_cds.fasta \
  --pdb-dir example/PDB --out results/prediction.csv
diff <(tr ',' '\t' < results/prediction.csv) example/expected_output.tsv && echo SMOKE-OK
```
