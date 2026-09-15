# Installation

## Requirements

- Python >= 3.9 (manuscript snapshot: Python 3.10.18 — record your exact version on Zenodo)
- PyTorch >= 2.0 with matching CUDA (manuscript: PyTorch 2.1.0, CUDA 11.8)
- scikit-learn 1.3.x (`>=1.3,<1.4`): base-learner pickles were exported under 1.3.x and may fail to unpickle on newer versions

## Quick install (stacking inference + FASTA/CDS features)

```bash
cd DeepGVS
pip install -r requirements.txt
```

For GPU builds, install torch with the wheel matching your CUDA first (see https://pytorch.org), e.g.:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## On-the-fly Graph-Mamba protein features (FASTA+PDB path)

The default FASTA+PDB workflow computes 1792-dim protein embeddings from PDB files at prediction
time. It needs the training-time checkpoint + ProtT5 directory (pointed to by
`DEEPGVS_GRAPH_MAMBA_CONFIG` or `--graph-mamba-config`) and two extra packages:

```bash
pip install fair-esm mamba-ssm
```

The checkpoint was trained under real Mamba; there is no LSTM fallback.
If you cannot install these, use `--protein-mode csv` with a precomputed
`graph_mamba_feat_*` table instead (see `docs/usage.md`).

## Conda (optional, matches training env)

```bash
conda env create -f environment.yml
conda activate deepgvs
```

## Verify

```bash
python scripts/predict.py --help
python -c "import deepgvs"  # after pip install -e .  (or use PYTHONPATH=src)
```
