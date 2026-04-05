# BindingAL
A scalable active learning framework for biomolecular binding prediction.

## 🚀 Getting Started

This project uses `conda` (or any virtualenv) for environment management and `pip` + `requirements.txt` for Python dependencies.

### 1. Set up a Python environment
From the repository root:
```bash
conda create -n bindingal python=3.10 -y
conda activate bindingal
```

If you prefer `python -m venv`, that is also fine.

### 2. Install dependencies
```bash
pip install -r requirements.txt
pip install -e .  # optional but recommended for `import bindingal`
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

TODO