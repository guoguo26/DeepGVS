# model/dataset_a — weights live on Zenodo

This directory on GitHub holds text-only metadata (`feature_schema.json`, `manifest.json`).
Download the full bundle from the Zenodo record (`models/dataset_a/`):

- `preprocess.joblib`
- `base_learners.pkl`
- `nam_meta_learner.pt`

into this folder, then run `scripts/predict.py --model-dir model/dataset_a ...`.
