# BindingAL
A scalable active learning framework for biomolecular binding prediction.

## 🚀 Getting Started (Development Setup)

This project uses [`uv`](https://github.com/astral-sh/uv) for fast and reproducible dependency management, and `ruff` for code linting and formatting.

### 1. Install `uv`
If you haven't installed `uv` yet, run:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create a virtual environment
From the repository root:
```bash
uv venv
```

### 3. Install dependencies (including dev tools)
```bash
uv sync --extra dev
```

### 4. Set up pre-commit (Ruff auto-fix on commit)
This repo is configured to run `ruff --fix` via pre-commit.

```bash
uv run pre-commit install
```

(Optional) Run it on all files once:
```bash
uv run pre-commit run --all-files
```

### 5. Run Ruff manually
```bash
uv run ruff check --fix .
```

### 6. Running scripts
> Note: the training/extraction scripts are currently placeholders.

```bash
uv run python -m scripts.extract
uv run python -m scripts.train
```