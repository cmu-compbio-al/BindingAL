# round-based al on frozen 4608-d concat: rf vs mlp+mc dropout, random vs entropy
# runs until the 80% train pool is empty (~6913 labels per seed)

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import paths  # noqa: E402

n_seeds = 5
seed_size = 100  # balanced cold start: half from each class
batch_size = 96  # plate-sized batch in the wet-lab story
max_rounds = 500  # safety cap; loop exits when unlabeled empty (~6913 pool labels)
test_fraction = 0.2
mc_samples = 20  # forward passes for predictive entropy (gal & ghahramani style)
dropout_rate = 0.3
hidden_dim = 256
n_epochs = 30
lr = 1e-3
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class MCDropoutMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp(model: nn.Module, x: np.ndarray, y: np.ndarray, epochs: int, lr_: float, dev: torch.device) -> None:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr_, weight_decay=1e-4)
    pos_w = torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)], dtype=torch.float32, device=dev)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)  # mild class rebalance on logits
    x_t = torch.tensor(x, dtype=torch.float32, device=dev)
    y_t = torch.tensor(y, dtype=torch.float32, device=dev).unsqueeze(1)
    for _ in range(epochs):
        opt.zero_grad()
        crit(model(x_t), y_t).backward()
        opt.step()


def mc_predict(model: nn.Module, x: np.ndarray, passes: int, dev: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.train()  # dropout on: each pass is a different bernoulli mask
    x_t = torch.tensor(x, dtype=torch.float32, device=dev)
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for _ in range(passes):
            logits = model(x_t)
            preds.append(torch.sigmoid(logits).cpu().numpy().ravel())
    stacked = np.array(preds)
    mean_p = stacked.mean(axis=0)
    ent = -(mean_p * np.log(mean_p + 1e-10) + (1 - mean_p) * np.log(1 - mean_p + 1e-10))
    return mean_p, ent  # binary entropy per point


def evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_hat = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_hat),
        "f1": f1_score(y_true, y_hat),
        "auprc": average_precision_score(y_true, y_prob),
        "auroc": roc_auc_score(y_true, y_prob),
    }


def run_al_mlp(
    x_pool: np.ndarray,
    y_pool: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    strategy: str,
    seed: int,
    scaler: StandardScaler,
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
    r = 0
    while r < max_rounds:
        li = sorted(labeled)
        x_l = scaler.transform(x_pool[li])
        y_l = y_pool[li]
        torch.manual_seed(seed + r)  # new mlp each round — avoids weight inertia across acquisitions
        model = MCDropoutMLP(x_l.shape[1], hidden_dim, dropout_rate).to(device)
        train_mlp(model, x_l, y_l, n_epochs, lr, device)
        mean_p, _ = mc_predict(model, scaler.transform(x_test), mc_samples, device)
        m = evaluate(y_test, mean_p)
        m.update({"round": r, "n_labeled": len(labeled), "strategy": strategy, "seed": seed, "model": "mlp_mcdropout"})
        rows.append(m)
        r += 1
        if not unlabeled:
            break
        u = sorted(unlabeled)
        b = min(batch_size, len(u))
        if strategy == "random":
            pick = rng.choice(u, b, replace=False)
        else:
            x_u = scaler.transform(x_pool[u])
            _, ent = mc_predict(model, x_u, mc_samples, device)
            pick = np.array(u)[np.argsort(ent)[-b:]]  # take noisiest points in the sense of predictive entropy
        for idx in pick:
            labeled.add(int(idx))
            unlabeled.discard(int(idx))
    return rows


def run_al_rf(
    x_pool: np.ndarray,
    y_pool: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    strategy: str,
    seed: int,
    scaler: StandardScaler,
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
    r = 0
    while r < max_rounds:
        li = sorted(labeled)
        x_l = scaler.transform(x_pool[li])
        y_l = y_pool[li]
        clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=-1)
        clf.fit(x_l, y_l)  # rf refit from scratch on current labeled set only
        y_prob = clf.predict_proba(scaler.transform(x_test))[:, 1]
        m = evaluate(y_test, y_prob)
        m.update({"round": r, "n_labeled": len(labeled), "strategy": strategy, "seed": seed, "model": "rf"})
        rows.append(m)
        r += 1
        if not unlabeled:
            break
        u = sorted(unlabeled)
        b = min(batch_size, len(u))
        if strategy == "random":
            pick = rng.choice(u, b, replace=False)
        else:
            pr = clf.predict_proba(scaler.transform(x_pool[u]))
            ent = -np.sum(pr * np.log(pr + 1e-10), axis=1)  # shannon entropy of {p(neg), p(pos)}
            pick = np.array(u)[np.argsort(ent)[-b:]]
        for idx in pick:
            labeled.add(int(idx))
            unlabeled.discard(int(idx))
    return rows


def main() -> None:
    out = paths.artifact_dir
    data = np.load(paths.cache_npz)
    x_all = data["x"]
    y_all = data["labels"]
    print(f"device {device} matrix {x_all.shape}")

    all_rows: list[dict] = []
    for seed in range(n_seeds):
        print(f"seed {seed + 1}/{n_seeds}")
        # stratify keeps pos/neg ratio stable in both pool and test across seeds
        x_pool, x_test, y_pool, y_test = train_test_split(
            x_all, y_all, test_size=test_fraction, random_state=seed, stratify=y_all
        )
        scaler = StandardScaler().fit(x_pool)  # fit only on pool — no test leakage
        for strat in ("random", "uncertainty"):
            print(f"  rf {strat}", end=" ", flush=True)
            r = run_al_rf(x_pool, y_pool, x_test, y_test, strat, seed, scaler)
            all_rows.extend(r)
            print(f"f1 {r[-1]['f1']:.3f}")
        for strat in ("random", "uncertainty"):
            print(f"  mlp {strat}", end=" ", flush=True)
            r = run_al_mlp(x_pool, y_pool, x_test, y_test, strat, seed, scaler)
            all_rows.extend(r)
            print(f"f1 {r[-1]['f1']:.3f}")

    df = pd.DataFrame(all_rows)
    csv_path = out / "al_comparison_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")

    style_map = {
        ("rf", "random"): {"color": "#bdc3c7", "ls": "--", "label": "RF + random"},
        ("rf", "uncertainty"): {"color": "#e74c3c", "ls": "--", "label": "RF + entropy"},
        ("mlp_mcdropout", "random"): {"color": "#3498db", "ls": "-", "label": "MLP + random"},
        ("mlp_mcdropout", "uncertainty"): {"color": "#2ecc71", "ls": "-", "label": "MLP + MC entropy"},
    }
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    for ax, (metric, title) in zip(
        axes.ravel(),
        [("accuracy", "Accuracy"), ("f1", "F1"), ("auprc", "AUPRC"), ("auroc", "AUROC")],
    ):
        for (model, strat), style in style_map.items():
            sub = df[(df["model"] == model) & (df["strategy"] == strat)]
            g = sub.groupby("n_labeled")[metric].agg(["mean", "std"]).reset_index()  # mean/std over seeds
            ax.plot(g["n_labeled"], g["mean"], color=style["color"], ls=style["ls"], lw=2, label=style["label"])
            ax.fill_between(g["n_labeled"], g["mean"] - g["std"], g["mean"] + g["std"], color=style["color"], alpha=0.1)
        ax.set_xlabel("labeled pool size")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle("AL: RF vs MLP+MC Dropout (concat features)", fontsize=15)
    fig.tight_layout()
    fig_path = out / "al_rf_vs_mlp_comparison.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
