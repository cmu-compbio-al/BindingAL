# reads dataset_1to1.csv + esm3 parquets, mean-pools tokens, concatenates h|l|ag -> 4608
# writes embedding_matrix.npz for every downstream script and saves umap/tsne figures

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import paths  # noqa: E402


def load_pooled_embedding_map(embed_dir: Path) -> dict[str, np.ndarray]:
    # one vector per raw sequence string; later we look up by exact seq from csv
    mapping: dict[str, np.ndarray] = {}
    parquets = sorted(embed_dir.glob("*.parquet"))
    for i, fp in enumerate(parquets):
        df = pd.read_parquet(fp)
        for _, row in df.iterrows():
            text = row["text"]
            emb = row["embedding"]
            arr = np.array(emb["data"], dtype=np.float32).reshape(emb["shape"])
            mapping[str(text)] = arr.mean(axis=0)  # token dim -> single embedding
        if (i + 1) % 20 == 0 or i == len(parquets) - 1:
            print(f"parquet files read: {i + 1}/{len(parquets)} ({len(mapping)} seqs)")
    return mapping


def main() -> None:
    out = paths.artifact_dir
    out.mkdir(parents=True, exist_ok=True)

    print("loading csv")
    dataset = pd.read_csv(paths.dataset_csv)
    print(f"rows {len(dataset)} labels {dataset['label'].value_counts().to_dict()}")

    print("loading parquets (slow first time)")
    emb_map = load_pooled_embedding_map(paths.embed_dir)
    emb_dim = len(next(iter(emb_map.values())))
    zero = np.zeros(emb_dim, dtype=np.float32)

    rows: list[np.ndarray] = []
    valid_idx: list[int] = []
    skipped = 0
    for idx, row in dataset.iterrows():
        heavy = str(row["heavy_seq"]) if pd.notna(row["heavy_seq"]) else None
        light = str(row["light_seq"]) if pd.notna(row["light_seq"]) else None
        antigen = str(row["antigen_seq"]) if pd.notna(row["antigen_seq"]) else None
        h = emb_map.get(heavy, zero) if heavy else zero
        l = emb_map.get(light, zero) if light else zero
        a = emb_map.get(antigen, zero) if antigen else zero
        # keep row if at least one chain hit the map (parquet coverage gaps happen)
        ok = (heavy and heavy in emb_map) or (light and light in emb_map) or (antigen and antigen in emb_map)
        if ok:
            rows.append(np.concatenate([h, l, a]))
            valid_idx.append(idx)
        else:
            skipped += 1
        if (idx + 1) % 2000 == 0:
            print(f"csv rows {idx + 1}/{len(dataset)}")

    x = np.asarray(rows, dtype=np.float32)
    labels = dataset.loc[valid_idx, "label"].values
    print(f"matrix {x.shape} skipped_no_emb {skipped}")

    print("umap 2d")
    u = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3, metric="cosine")
    x_umap = u.fit_transform(x)  # global structure of frozen concat features
    print("tsne 2d")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    x_tsne = tsne.fit_transform(x)

    colors = {0: "#e74c3c", 1: "#2ecc71"}
    names = {0: "negative", 1: "positive"}
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax, xy, title in zip(axes, [x_umap, x_tsne], ["UMAP", "t-SNE"]):
        for lab in (0, 1):
            m = labels == lab
            ax.scatter(xy[m, 0], xy[m, 1], c=colors[lab], alpha=0.4, s=8, label=names[lab])
        ax.set_title(title)
        ax.legend(markerscale=3)
    fig.tight_layout()
    p1 = out / "embeddings_pos_vs_neg.png"
    fig.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p1}")

    # optional split views (same run, quick sanity on where signal might hide)
    fig2, axes2 = plt.subplots(1, 3, figsize=(24, 7))
    for lab in (0, 1):
        m = labels == lab
        axes2[0].scatter(x_umap[m, 0], x_umap[m, 1], c=colors[lab], alpha=0.35, s=6, label=names[lab])
    axes2[0].set_title("concat H+L+Ag")
    axes2[0].legend(markerscale=3, fontsize=10)

    x_ag = np.array([emb_map.get(str(dataset.loc[i, "antigen_seq"]), zero) for i in valid_idx], dtype=np.float32)
    x_ag_umap = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3, metric="cosine").fit_transform(x_ag)
    for lab in (0, 1):
        m = labels == lab
        axes2[1].scatter(x_ag_umap[m, 0], x_ag_umap[m, 1], c=colors[lab], alpha=0.35, s=6, label=names[lab])
    axes2[1].set_title("antigen only")
    axes2[1].legend(markerscale=3, fontsize=10)

    x_ab = np.array(
        [
            np.concatenate(
                [
                    emb_map.get(str(dataset.loc[i, "heavy_seq"]), zero),
                    emb_map.get(str(dataset.loc[i, "light_seq"]), zero)
                    if pd.notna(dataset.loc[i, "light_seq"])
                    else zero,
                ]
            )
            for i in valid_idx
        ],
        dtype=np.float32,
    )
    x_ab_umap = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3, metric="cosine").fit_transform(x_ab)
    for lab in (0, 1):
        m = labels == lab
        axes2[2].scatter(x_ab_umap[m, 0], x_ab_umap[m, 1], c=colors[lab], alpha=0.35, s=6, label=names[lab])
    axes2[2].set_title("antibody H+L")
    axes2[2].legend(markerscale=3, fontsize=10)
    fig2.tight_layout()
    p2 = out / "embeddings_by_component.png"
    fig2.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"wrote {p2}")

    # valid_indices maps matrix rows back to original csv row index if you need joins
    np.savez_compressed(paths.cache_npz, x=x, labels=labels, valid_indices=np.array(valid_idx, dtype=np.int64))
    print(f"wrote {paths.cache_npz}")


if __name__ == "__main__":
    main()
