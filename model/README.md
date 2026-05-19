# DeepGVS `model/` 模型目录

本目录存放 **Dataset A / B** 的 stacking 推理权重。日常预测请把 `-m` 指向**具体数据集子目录**（例如 `model/dataset_a/`），不要混用两套 schema 与 preprocess。

## 目录结构

```text
model/
├── dataset_a/                  # Dataset A 完整发布包（推荐默认）
│   ├── feature_schema.json     # 2146 列特征名（354 CDS + 1792 Graph-Mamba）
│   ├── preprocess.joblib       # 2146 → 182（MI + Graph PCA 等）
│   ├── base_learners.pkl       # 四个基学习器（RF / SVM / XGB / MLP）
│   ├── nam_meta_learner.pt     # NAM 元学习器
│   └── manifest.json           # 维度与文件说明（机器可读）
│
├── dataset_b/                  # Dataset B stacking 权重（勿与 A 混用）
│   ├── base_learners.pkl
│   └── nam_meta_learner.pt
│
└── README.md
```

## 推理流程（以 Dataset A 为例）

```text
原始合并特征 (2146)
    → feature_schema.json 对齐列顺序
    → preprocess.joblib (182)
    → base_learners.pkl 四个模型各输出 1 个正类概率
    → nam_meta_learner.pt 得到 VF 概率 / 标签
```

| 阶段 | 维度 |
|------|------|
| 原始合并特征 | 2146 |
| preprocess 之后 | 182 |
| 每个基学习器 → meta | 1 × 4 |
| NAM 输入 | 4 |

## 各文件是否必须

### `dataset_a/`（预测 A 集样本）

| 文件 | 必须 | 说明 |
|------|------|------|
| `feature_schema.json` | 是 | 定义 2146 列顺序 |
| `preprocess.joblib` | 是 | 须与训练时 MI/PCA 设置一致，输出 182 维 |
| `base_learners.pkl` | 是* | 含 `random_forest`、`svm_rbf`、`xgboost`、`mlp`；也可用四个独立 `rf.pkl` 等代替 |
| `nam_meta_learner.pt` | 是 | 也可用同名 `nam.pt` |

\* `code/model.py` 优先加载分文件 `rf.pkl` … `mlp.pkl`；若不存在则读 `base_learners.pkl`。

### `dataset_b/`

| 文件 | 状态 |
|------|------|
| `base_learners.pkl`、`nam_meta_learner.pt` | 已有 |
| `feature_schema.json`、`preprocess.joblib` | **未包含**，需按 B 的训练特征表自行准备后才能做完整预测 |

**不要**用 A 的 `feature_schema.json` / `preprocess.joblib` 预测 B。

## 命令示例

```bash
cd DeepGVS

# 校验模型包
python code/verify_model.py -m model/dataset_a/

# 合并特征 CSV 预测
python code/prediction.py -m model/dataset_a/ \
  --merged-features ../data/example_features.csv \
  -o results/predictions.csv

# FASTA 预测（示例：从 PDB 现场提取结构特征）
python code/prediction.py -m model/dataset_a/ \
  -pseq Dataset/example/example_protein.fasta \
  -cds Dataset/example/example_cds.fasta \
  --protein-mode compute \
  --pdb-dir Dataset/example/PDB \
  -o results/predictions.csv
```

## 基学习器权重

`dataset_a/base_learners.pkl` 已包含四个基学习器（RF / SVM / XGB / MLP）。`code/model.py` 也可分别加载 `rf.pkl`、`svm.pkl`、`xgboost.pkl`、`mlp.pkl`（若存在）。

## 注意事项

- **A 与 B 不可混用**：特征维度、schema、preprocess 均可能不同。
- **sklearn 版本**：权重多为 sklearn 1.3.x 导出，高版本加载可能有 `InconsistentVersionWarning`；建议与训练环境一致。
- **Graph-Mamba 结构特征**：本仓库**不包含**预计算蛋白特征表。FASTA 模式请任选其一：用 `--protein-features /path/to.csv` 提供含 `graph_mamba_feat_*` 的查表（`--protein-mode csv`）；或用 `--protein-mode compute` / `auto` 从 PDB 现场计算（需 `model/graph_mamba_infer.json` 与对应 `{sequence_id}.pdb`）。
