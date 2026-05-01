"""
Central locations for data, cached features, and figures.
Edit dataset_csv / embed_dir in paths.py if your files live elsewhere.
"""
from pathlib import Path

_pkg = Path(__file__).resolve().parent
project_root = _pkg.parent  # .../Automation/Project

# input: paired sequences + labels (your pipeline export)
dataset_csv = Path.home() / "Downloads" / "embeddings" / "dataset_1to1.csv"
embed_dir = Path.home() / "Downloads" / "embeddings"  # esm3-sm parquet shards live here

# heavy npz + all experiment csv/png go next to the original Projectscripts run
artifact_dir = project_root / "Projectscripts"
cache_npz = artifact_dir / "embedding_matrix.npz"  # written by 01_, read by everything else
