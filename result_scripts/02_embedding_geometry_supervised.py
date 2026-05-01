# uses cached 4608-d concat + simple interaction augments for a richer feature block
# trains one rf, then plots: supervised umap, rf leaf-distance umap, prob histograms, raw umap

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import interaction_features  # noqa: E402
import paths  # noqa: E402


def main() -> None:
    out = paths.artifact_dir
    data = np.load(paths.cache_npz)
    x_all = data["x"]
    labels = data["labels"]
    print(f"loaded {x_all.shape}")

    x_inter = interaction_features.build_interaction_features(x_all)
    x_feat = np.hstack([x_all, x_inter])  # concat + cheap pairwise terms; rf can use both
    print(f"feature block {x_feat.shape}")

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_feat)

    # labels enter umap as targets — shows separability a linear read might miss in raw space
    print("supervised umap")
    sup = umap.UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=30,
        min_dist=0.3,
        metric="cosine",
        target_metric="categorical",
    )
    xy_sup = sup.fit_transform(x_scaled, y=labels)

    x_train, x_test, y_train, y_test = train_test_split(x_feat, labels, test_size=0.2, random_state=42, stratify=labels)
    sc2 = StandardScaler()
    x_train_s = sc2.fit_transform(x_train)
    x_test_s = sc2.transform(x_test)

    print("training rf")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(x_train_s, y_train)
    y_prob = rf.predict_proba(x_test_s)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    print(f"test acc {accuracy_score(y_test, y_pred):.3f} f1 {f1_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred, target_names=["neg", "pos"]))

    rng = np.random.RandomState(42)
    n = len(x_feat)
    sample_idx = rng.choice(n, min(5000, n), replace=False)
    leaf = rf.apply(sc2.transform(x_feat[sample_idx]))
    prox = np.zeros((len(sample_idx), len(sample_idx)), dtype=np.float32)
    for t in range(leaf.shape[1]):
        col = leaf[:, t]
        prox += (col[:, None] == col[None, :]).astype(np.float32)  # same leaf = similar rf routing
    prox /= leaf.shape[1]
    dist = 1.0 - prox
    print("umap on rf leaf distance")
    xy_prox = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3, metric="precomputed").fit_transform(dist)

    y_prob_all = rf.predict_proba(sc2.transform(x_feat))[:, 1]  # for histogram: can model shift mass?

    print("unsupervised umap (same scaled concat+inter)")
    xy_raw = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3, metric="cosine").fit_transform(
        x_scaled
    )  # no labels: should overlap more if raw geometry is messy

    colors = {0: "#e74c3c", 1: "#2ecc71"}
    names = {0: "negative", 1: "positive"}
    lab_s = labels[sample_idx]

    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    for lab in (0, 1):
        m = labels == lab
        axes[0, 0].scatter(xy_sup[m, 0], xy_sup[m, 1], c=colors[lab], alpha=0.4, s=8, label=names[lab])
    axes[0, 0].set_title("supervised umap")
    axes[0, 0].legend(markerscale=3)

    for lab in (0, 1):
        m = lab_s == lab
        axes[0, 1].scatter(xy_prox[m, 0], xy_prox[m, 1], c=colors[lab], alpha=0.4, s=8, label=names[lab])
    axes[0, 1].set_title("rf leaf proximity -> umap (5k subsample)")
    axes[0, 1].legend(markerscale=3)

    for lab in (0, 1):
        m = labels == lab
        axes[1, 0].hist(y_prob_all[m], bins=80, alpha=0.6, color=colors[lab], label=names[lab], density=True)
    axes[1, 0].axvline(0.5, color="black", ls="--", alpha=0.5)
    axes[1, 0].set_title("rf p(binding) on full matrix")
    axes[1, 0].legend()

    for lab in (0, 1):
        m = labels == lab
        axes[1, 1].scatter(xy_raw[m, 0], xy_raw[m, 1], c=colors[lab], alpha=0.4, s=8, label=names[lab])
    axes[1, 1].set_title("unsupervised umap (same inputs)")
    axes[1, 1].legend(markerscale=3)

    fig.suptitle("raw geometry vs what a shallow ensemble sees", fontsize=15, y=1.01)
    fig.tight_layout()
    fp = out / "supervised_vs_unsupervised_embeddings.png"
    fig.savefig(fp, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fp}")


if __name__ == "__main__":
    main()
