# DeepGVS

DeepGVS is a bimodal deep learning framework that predicts bacterial virulence factors (VF) from coding sequences. It integrates protein-structural representations (ESM-2 residue features encoded by a Graph-Mamba network over ESMFold-predicted structures) with coding-sequence features (354-dim CDS features); the 2146-dim concatenated features are scored by a stacking ensemble (RF / SVM / XGBoost / MLP) with a NAM meta-learner. The pretrained models and the datasets analyzed in the manuscript are available at Zenodo: [10.5281/zenodo.21756890](https://zenodo.org/records/21756890).

## Installation

### Approach 1, install with mamba/conda. (Recommended)

```bash
conda env create -f environment.yml
conda activate deepgvs
```

### Approach 2, from scratch

Prerequisites

Python packages

```text
numpy==1.26.2
scipy==1.10.1
scikit-learn==1.3.2
xgboost==3.0.5
transformers==4.30.0
pandas
joblib
tqdm
pyyaml
```

Torch packages, (gpu)

Please check https://pytorch.org/get-started/previous-versions/ for installation

```text
torch==2.1.0   # CUDA 11.8 build
```

Other packages (only needed for the online FASTA + PDB path)

```text
fair-esm==2.0.0
mamba-ssm==1.2.2
causal-conv1d==1.2.2.post1
```

`mamba-ssm` / `causal-conv1d` compile CUDA kernels at install time and need a CUDA 11.8 toolkit and a C++ compiler (g++). The stacking-only path (merged-features CSV input) runs on CPU without these packages.

Install the prerequisites first, then clone the repository and enter the directory:

```bash
git clone https://github.com/guoguo26/DeepGVS
cd DeepGVS
pip install -r requirements.txt
```

## Using DeepGVS

1. Download the pretrained models from Zenodo ([10.5281/zenodo.21756890](https://zenodo.org/records/21756890)) and unpack them into the repository:

```text
model/dataset_a/    preprocess.joblib, base_learners.pkl, nam_meta_learner.pt, feature_schema.json
model/dataset_b/    same four files
models/             DeepGVS_DatasetA.pt, DeepGVS_DatasetB.pt   (Graph-Mamba checkpoints)
```

2. Config the graph-mamba JSON file (`configs/graph_mamba_infer.json` for Dataset A, `configs/graph_mamba_infer_B.json` for Dataset B), here is a demo file.

```json
{
  "checkpoint": "models/DeepGVS_DatasetA.pt",
  "global_prott5_dir": "/path/to/prott5_global_npy",
  "esm_cache_dir": "/path/to/esm_cache",
  "args": { "k": 30, "num_mamba_layers": 6, "mamba_d_state": 16, "gat_heads": 4 }
}
```

- checkpoint, Graph-Mamba weights; use the `DeepGVS_DatasetA.pt` / `DeepGVS_DatasetB.pt` files from Zenodo.
- global_prott5_dir, ProtT5 embeddings directory (`Rostlab/prot_t5_xl_half_uniref50-enc`, 1,024-dim); offline downloads are accepted.
- esm_cache_dir, ESM-2 cache directory (`facebook/esm2_t33_650M_UR50D`, layer 33).
- args, training-time graph/encoder settings (30-NN graph, 6 Mamba layers, state dim 16, 4 GAT heads); keep as shipped unless retraining.

3. Runing DeepGVS.

Online prediction from paired FASTA + PDB (all features computed on the fly):

```bash
python scripts/predict.py \
  --model-dir model/dataset_a \
  --protein-fasta example/example_protein.fasta \
  --cds-fasta example/example_cds.fasta \
  --pdb-dir example/PDB \
  --graph-mamba-config configs/graph_mamba_infer.json \
  --out results/prediction.csv
```

Dataset B: same command with `--model-dir model/dataset_b` and `--graph-mamba-config configs/graph_mamba_infer_B.json`.

CPU-only prediction from a precomputed 2146-column merged CSV:

```bash
python scripts/predict.py \
  --model-dir model/dataset_b \
  --merged-features features.csv \
  --out results/prediction_b.csv
```

FASTA ID = first token after `>` (`|` is normalized to `_`). Only IDs present in both the protein and CDS FASTA files are predicted.

## Output

`results/prediction.csv` contains three columns:

```text
Sample_ID,VF_probability,Prediction
VFG011519(gb_WP_002963669),0.7158586382865906,VF
```

- Sample_ID, input sequence ID.
- VF_probability, predicted probability of being a virulence factor (NAM meta-learner output).
- Prediction, `VF` / `non-VF` at threshold `--threshold` (default 0.5).

The shipped `example/expected_output.tsv` is the real output of the Dataset A command above. On the Dataset B independent test set (1,152 protein-CDS pairs) DeepGVS reproduces the manuscript results: AUC 0.9275, balanced accuracy 0.8750 (see `docs/reproduction.md`).

## Author

DeepGVS is developed by Prof. Guohua Wang's group at the School of Computer Science and Artificial Intelligence, Northeast Forestry University, Harbin, China. Authors: Yan Miao, Tingting Zou, Zhenyuan Sun, Yuming Zhao and Guohua Wang. Should you have any queries, please feel free to contact us by ghwang@nefu.edu.cn.

If you use DeepGVS, please cite the manuscript (see `CITATION.cff`).

## License

This project is licensed under the MIT License - see the LICENSE file for details.
