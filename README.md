# BindingAL
A scalable active learning framework for biomolecular binding prediction.

## 🚀 Getting Started

This project uses `conda` (or any virtualenv) for environment management and `pip` + `requirements.txt` for Python dependencies. The package is also configured for installation (editable or regular) via `pyproject.toml`.

### 1. Set up a Python environment
From the repository root:
```bash
conda create -n bindingal python=3.10 -y
conda activate bindingal
```

If you prefer `python -m venv`, that is also fine.

### 2. Install dependencies and the package (local clone)
```bash
pip install -r requirements.txt
pip install -e .  # install `bindingal` in editable mode
```

Once the project is published to a package index, other users will be able to install it directly with:

```bash
pip install bindingal
```

---

## 🧬 Embedding Extraction (Local)

Run the embedding extraction script on a local machine.

### CSV input (multiple sequences in one file)

```bash
python scripts/extract.py \
  --input_path path/to/input.csv \
  --output_dir_path path/to/output_parquet_dir \
  --sequence_column sequence \
  --embedding_column embedding \
  --num_actors 1 \
  --device cuda
```

### Directory or S3 of text files (one sequence per file)

```bash
python scripts/extract.py \
  --input_path s3://my-bucket/my-text-shards/ \
  --output_dir_path s3://my-bucket/my-embeddings-output/ \
  --embedding_column embedding \
  --num_actors 1 \
  --device cuda
```

Notes:
- When `input_path` is a directory or S3 prefix, the script expects **one sequence per file**.
- Outputs are written as Parquet files with one row per sequence and an embedding column.

---

## ☁️ Embedding Extraction on SageMaker

To launch the embedding extraction pipeline as a SageMaker training job, run from the repository root:

```bash
python scripts/run_sagemaker.py \
  --arn_role arn:aws:iam::<ACCOUNT_ID>:role/<SAGEMAKER_ROLE_NAME> \
  --s3_input_path s3://my-bucket/my-input-shards/ \
  --s3_embedding_output_path s3://my-bucket/my-embeddings-output/ \
  --s3_artifact_path s3://my-bucket/my-sm-artifacts/ \
  --instance_type ml.g5.12xlarge \
  --instance_count 1 \
  --num_actors 4 \
  --model_name_or_path esm3_sm_open_v1
```

Notes:
- Always run this command from the repo root so that `source_dir="."` and `entry_point="scripts/extract.py"` resolve correctly.
- `--s3_input_path` should point to a directory of sharded text files in S3 (one sequence per file).
- `--instance_type` above (`ml.g5.12xlarge`) is just an example; choose a type that matches your quota and workload.
- `--s3_embedding_output_path` is an S3 directory where Parquet files with embeddings will be written.
- `--s3_artifact_path` is where SageMaker will store model artifacts and logs.

---

## 📚 Training
### Local Training
To train the model locally with active learning:

```bash
python scripts/train.py \
  --train_data_path path/to/train.csv \
  --embed_data_path path/to/embeddings.parquet \
  --batch_size 24 \
  --query_size 96 \
  --epochs 50 \
  --learning_rate 1e-4 \
  --strategy mc_dropout \
  --report_file_path /opt/ml/model/report.json \
  --num_workers 4
```

### SageMaker Training
To launch the training pipeline as a SageMaker training job:

```bash
python scripts/run_sagemaker_train.py \
  --arn_role arn:aws:iam::<ACCOUNT_ID>:role/<SAGEMAKER_ROLE_NAME> \
  --train_data_path s3://my-bucket/my-train-data/ \
  --embed_data_path s3://my-bucket/my-embeddings/ \
  --s3_artifact_path s3://my-bucket/my-sm-artifacts/ \
  --instance_type ml.g5.12xlarge \
  --instance_count 4 \
  --batch_size 24 \
  --query_size 96 \
  --epochs 50 \
  --learning_rate 1e-4 \
  --strategy ensemble
```

## Result scripts

| File | Purpose |
|------|---------|
| `paths.py` | `dataset_csv`, `embed_dir`, `cache_npz`, `artifact_dir` — edit if your data moved. |
| `interaction_features.py` | Builds 3073-d interaction vectors from cached 4608-d concat (H\|L\|Ag). |
| `01_build_concat_cache_umap.py` | Parquet → mean-pool → concat; writes `embedding_matrix.npz`; UMAP/t-SNE figures. |
| `02_embedding_geometry_supervised.py` | Supervised vs unsupervised UMAP, RF leaf proximity, prob histograms (needs cache). |
| `03_al_concat_rf_mlp_pool.py` | Main AL loop: RF vs MLP+MC Dropout, random vs entropy; exhausts train pool. |
| `04_plot_mlp_accuracy_from_csv.py` | MLP-only accuracy plot from `al_comparison_results.csv` (after 03). |
| `05_al_salience_weighted_interaction.py` | Interaction MLP: passive, entropy, salience-weighted, 50/50 hybrid. |
| `06_al_interaction_badge.py` | Interaction MLP + BADGE-lite; batches 24 and 96; fixed round budget. |
| `07_al_interaction_exhaustive_no_badge.py` | Interaction MLP, passive vs entropy only; exhausts pool; batches 24/96. |

## Run order

1. Set paths in `paths.py`.
2. Run `01` once (heavy I/O).
3. Run `02`–`07` as needed for figures/tables; `04` depends on `03`.

## Requirements

Python 3 with `numpy`, `pandas`, `torch`, `sklearn`, `matplotlib`, `umap-learn`; parquets + `dataset_1to1.csv` at the paths you configure.

Notes:
- `--strategy` can be one of `passive`, `mc_dropout`, or `ensemble`.
- `--s3_artifact_path` is where SageMaker will store model artifacts and logs.
- Ensure that the IAM role specified in `--arn_role` has the necessary permissions for S3 and SageMaker.

---
