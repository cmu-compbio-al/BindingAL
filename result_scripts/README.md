
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
