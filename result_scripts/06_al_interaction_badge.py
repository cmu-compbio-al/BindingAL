# interaction 3073 + deeper mlp: passive vs mc entropy vs lightweight badge (kmeans on uncertain slice)
# two batch sizes (24, 96); fixed round cap 40

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import interaction_features  # noqa: E402
import paths  # noqa: E402

n_seeds = 5
seed_size = 100
n_rounds = 40  # fixed horizon — not trying to empty pool here
test_fraction = 0.2
mc_samples = 30
dropout_rate = 0.3
hidden_dim = 512
n_epochs = 80
lr = 5e-4
batch_sizes = (24, 96)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class MCDropoutMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int, drop: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden // 2, hidden // 4),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp(model: nn.Module, x: np.ndarray, y: np.ndarray, epochs: int, lr_: float, dev: torch.device) -> None:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr_, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    pos_w = torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)], dtype=torch.float32, device=dev)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    x_t = torch.tensor(x, dtype=torch.float32, device=dev)
    y_t = torch.tensor(y, dtype=torch.float32, device=dev).unsqueeze(1)
    bs = min(256, len(x))
    n_batches = max(1, len(x) // bs)
    for _ in range(epochs):
        perm = torch.randperm(len(x_t))
        for b in range(n_batches):
            idx = perm[b * bs : (b + 1) * bs]
            opt.zero_grad()
            crit(model(x_t[idx]), y_t[idx]).backward()
            opt.step()
        sched.step()


def mc_predict(model: nn.Module, x: np.ndarray, passes: int, dev: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.train()
    x_t = torch.tensor(x, dtype=torch.float32, device=dev)
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for _ in range(passes):
            logits = model(x_t)
            preds.append(torch.sigmoid(logits).cpu().numpy().ravel())
    stacked = np.array(preds)
    mean_p = stacked.mean(axis=0)
    std_p = stacked.std(axis=0)
    ent = -(mean_p * np.log(mean_p + 1e-10) + (1 - mean_p) * np.log(1 - mean_p + 1e-10))
    return mean_p, ent, std_p


def badge_select(model: nn.Module, x_pool_u: np.ndarray, batch_sz: int, dev: torch.device) -> np.ndarray:
    # cheap badge flavor: narrow to top uncertain slice, then kmeans for diversity
    _, ent, _ = mc_predict(model, x_pool_u, mc_samples, dev)
    cand_n = min(batch_sz * 5, len(x_pool_u))
    top_local = np.argsort(ent)[-cand_n:]
    candidates = x_pool_u[top_local]
    km = MiniBatchKMeans(n_clusters=batch_sz, random_state=42, n_init=3)
    km.fit(candidates)
    chosen: list[int] = []
    for c in range(batch_sz):
        mask = km.labels_ == c
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue
        d = np.linalg.norm(candidates[idxs] - km.cluster_centers_[c], axis=1)
        chosen.append(int(idxs[np.argmin(d)]))  # closest point to each cluster center
    return top_local[np.array(chosen, dtype=np.int64)]  # indices relative to full u slice passed in


def evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_hat = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_hat),
        "f1": f1_score(y_true, y_hat, zero_division=0),
        "auprc": average_precision_score(y_true, y_prob),
        "auroc": roc_auc_score(y_true, y_prob),
    }


def run_al(
    x_pool: np.ndarray,
    y_pool: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    strategy: str,
    seed: int,
    scaler: StandardScaler,
    batch_sz: int,
) -> list[dict]:
    rng = np.random.RandomState(seed)
    n = len(y_pool)
    pos = np.where(y_pool == 1)[0]
    neg = np.where(y_pool == 0)[0]
    labeled = set(
        rng.choice(pos, seed_size // 2, replace=False).tolist() + rng.choice(neg, seed_size // 2, replace=False).tolist()
    )
    unlabeled = set(range(n)) - labeled
    rows: list[dict] = []
    for r in range(n_rounds + 1):
        li = sorted(labeled)
        x_l = scaler.transform(x_pool[li])
        y_l = y_pool[li]
        torch.manual_seed(seed + r)
        model = MCDropoutMLP(x_l.shape[1], hidden_dim, dropout_rate).to(device)
        train_mlp(model, x_l, y_l, n_epochs, lr, device)
        mean_p, _, _ = mc_predict(model, scaler.transform(x_test), mc_samples, device)
        m = evaluate(y_test, mean_p)
        m.update({"round": r, "n_labeled": len(labeled), "strategy": strategy, "seed": seed, "batch_size": batch_sz})
        rows.append(m)
        if r == n_rounds or not unlabeled:
            break
        u = sorted(unlabeled)
        b = min(batch_sz, len(u))
        if strategy == "passive":
            pick = rng.choice(u, b, replace=False)
        elif strategy == "uncertainty":
            x_u = scaler.transform(x_pool[u])
            _, ent, _ = mc_predict(model, x_u, mc_samples, device)
            pick = np.array(u)[np.argsort(ent)[-b:]]
        else:
            x_u = scaler.transform(x_pool[u])
            bi = badge_select(model, x_u, b, device)  # indices into x_u / u order
            pick = np.array(u)[bi]
        for idx in pick:
            labeled.add(int(idx))
            unlabeled.discard(int(idx))
    return rows


def main() -> None:
    out = paths.artifact_dir
    data = np.load(paths.cache_npz)
    x_int = interaction_features.build_interaction_features(data["x"])
    labels = data["labels"]
    print(f"device {device} features {x_int.shape}")

    all_rows: list[dict] = []
    for batch_sz in batch_sizes:
        print(f"batch {batch_sz}")
        for seed in range(n_seeds):
            x_pool, x_test, y_pool, y_test = train_test_split(
                x_int, labels, test_size=test_fraction, random_state=seed, stratify=labels
            )
            scaler = StandardScaler().fit(x_pool)
            for strat in ("passive", "uncertainty", "badge"):
                print(f"  seed {seed} {strat}", end=" ", flush=True)
                r = run_al(x_pool, y_pool, x_test, y_test, strat, seed, scaler, batch_sz)
                all_rows.extend(r)
                print(f"f1 {r[-1]['f1']:.3f}")

    df = pd.DataFrame(all_rows)
    df.to_csv(out / "al_improved_results.csv", index=False)
    print(f"wrote {out / 'al_improved_results.csv'}")

    for batch_sz in batch_sizes:
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        # one png per batch size so slides can pick plate layout vs finer queries
        style_map = {
            "passive": {"color": "#95a5a6", "ls": "-", "label": "passive"},
            "uncertainty": {"color": "#e74c3c", "ls": "-", "label": "MC entropy"},
            "badge": {"color": "#2ecc71", "ls": "-", "label": "BADGE-lite"},
        }
        sub = df[df["batch_size"] == batch_sz]
        for ax, (metric, name) in zip(axes, [("accuracy", "Accuracy"), ("f1", "F1"), ("auprc", "AUPRC")]):
            for strat, style in style_map.items():
                g = sub[sub["strategy"] == strat].groupby("n_labeled")[metric].agg(["mean", "std"]).reset_index()
                ax.plot(g["n_labeled"], g["mean"], color=style["color"], ls=style["ls"], lw=2, label=style["label"])
                ax.fill_between(g["n_labeled"], g["mean"] - g["std"], g["mean"] + g["std"], color=style["color"], alpha=0.12)
            ax.set_xlabel("labeled samples")
            ax.set_ylabel(name)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"interaction MLP AL (batch={batch_sz})", fontsize=14)
        fig.tight_layout()
        fp = out / f"al_improved_batch{batch_sz}.png"
        fig.savefig(fp, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {fp}")


if __name__ == "__main__":
    main()
