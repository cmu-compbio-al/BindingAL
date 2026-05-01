# mlp on 3073-d interaction features: passive, mc entropy, mild hadamard reweight, 50/50 hybrid
# paper’s salience term is ||Ab⊙Ag|| (z-scored in-batch) times mc entropy with small w

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
batch_size = 96
n_rounds = 25
test_fraction = 0.2
mc_samples = 30
dropout_rate = 0.3
hidden_dim = 512
n_epochs = 40
lr = 5e-4
hadamard_w = 0.12  # small multiplier so salience nudges but doesn’t drown entropy
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


def mc_predict(model: nn.Module, x: np.ndarray, passes: int, dev: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.train()
    x_t = torch.tensor(x, dtype=torch.float32, device=dev)
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for _ in range(passes):
            logits = model(x_t)
            preds.append(torch.sigmoid(logits).cpu().numpy().ravel())
    stacked = np.array(preds)
    mean_p = stacked.mean(axis=0)
    ent = -(mean_p * np.log(mean_p + 1e-10) + (1 - mean_p) * np.log(1 - mean_p + 1e-10))
    return mean_p, ent


def evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_hat = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_hat),
        "f1": f1_score(y_true, y_hat, zero_division=0),
        "auprc": average_precision_score(y_true, y_prob),
        "auroc": roc_auc_score(y_true, y_prob),
    }


def hadamard_block_norm(x_raw: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x_raw[:, :1536], axis=1)  # first 1536 cols are Ab⊙Ag in our layout


def weighted_score(ent: np.ndarray, raw_u: np.ndarray) -> np.ndarray:
    mag = hadamard_block_norm(raw_u)
    z = (mag - mag.mean()) / (mag.std() + 1e-8)  # compare salience within this round’s candidate set only
    return ent * (1.0 + hadamard_w * np.tanh(z))  # bounded tweak via tanh


def run_al(
    x_pool_raw: np.ndarray,
    x_pool_scaled: np.ndarray,
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
    for r in range(n_rounds + 1):
        li = sorted(labeled)
        x_l = scaler.transform(x_pool_scaled[li])
        y_l = y_pool[li]
        torch.manual_seed(seed + r)
        model = MCDropoutMLP(x_l.shape[1], hidden_dim, dropout_rate).to(device)
        train_mlp(model, x_l, y_l, n_epochs, lr, device)
        mean_p, _ = mc_predict(model, scaler.transform(x_test), mc_samples, device)
        m = evaluate(y_test, mean_p)
        m.update({"round": r, "n_labeled": len(labeled), "strategy": strategy, "seed": seed})
        rows.append(m)
        if r == n_rounds or not unlabeled:
            break
        u = sorted(unlabeled)
        b = min(batch_size, len(u))
        if strategy == "passive":
            pick = rng.choice(u, b, replace=False)
        else:
            x_u = scaler.transform(x_pool_scaled[u])
            raw_u = x_pool_raw[np.array(u)]  # salience uses unscaled interaction rows
            _, ent = mc_predict(model, x_u, mc_samples, device)
            if strategy == "uncertainty":
                pick = np.array(u)[np.argsort(ent)[-b:]]
            elif strategy == "hybrid":
                n_rand = b // 2
                n_w = b - n_rand
                rand_idx = rng.choice(u, n_rand, replace=False)
                rem = [i for i in u if i not in set(rand_idx)]
                if n_w <= 0:
                    pick = list(rand_idx)
                elif len(rem) <= n_w:
                    pick = list(rand_idx) + rem
                else:
                    x_w = scaler.transform(x_pool_scaled[rem])
                    raw_w = x_pool_raw[np.array(rem)]
                    _, ent_w = mc_predict(model, x_w, mc_samples, device)
                    sc = weighted_score(ent_w, raw_w)
                    pick = list(rand_idx) + list(np.array(rem)[np.argsort(sc)[-n_w:]])  # weighted half after random half
            else:
                sc = weighted_score(ent, raw_u)
                pick = np.array(u)[np.argsort(sc)[-b:]]  # interaction_weighted
        for idx in pick:
            labeled.add(int(idx))
            unlabeled.discard(int(idx))
    return rows


def main() -> None:
    out = paths.artifact_dir
    data = np.load(paths.cache_npz)
    x_raw = data["x"]
    labels = data["labels"]
    x_int = interaction_features.build_interaction_features(x_raw)
    print(f"device {device} interaction shape {x_int.shape}")

    all_rows: list[dict] = []
    for seed in range(n_seeds):
        print(f"seed {seed}")
        x_pool, x_test, y_pool, y_test = train_test_split(
            x_int, labels, test_size=test_fraction, random_state=seed, stratify=labels
        )
        scaler = StandardScaler().fit(x_pool)
        for strat in ("passive", "uncertainty", "interaction_weighted", "hybrid"):
            print(f"  {strat}", end=" ", flush=True)
            r = run_al(x_pool, x_pool, y_pool, x_test, y_test, strat, seed, scaler)
            all_rows.extend(r)
            print(f"acc {r[-1]['accuracy']:.3f}")

    df = pd.DataFrame(all_rows)
    csv_out = out / "al_interaction_weighted_v2_results.csv"
    df.to_csv(csv_out, index=False)
    print(f"wrote {csv_out}")

    styles = {
        "passive": {"c": "#95a5a6", "ls": "-", "label": "passive"},
        "uncertainty": {"c": "#e74c3c", "ls": "-", "label": "MC entropy"},
        "interaction_weighted": {"c": "#2980b9", "ls": "-", "label": f"weighted w={hadamard_w}"},
        "hybrid": {"c": "#8e44ad", "ls": "-", "label": "50% random + 50% weighted"},
    }
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for strat, st in styles.items():
        g = df[df["strategy"] == strat].groupby("n_labeled")["accuracy"].agg(["mean", "std"]).reset_index()
        ax.plot(g["n_labeled"], g["mean"], color=st["c"], ls=st["ls"], lw=2, label=st["label"])
        ax.fill_between(g["n_labeled"], g["mean"] - g["std"], g["mean"] + g["std"], color=st["c"], alpha=0.1)
    ax.set_xlabel("labeled training samples")
    ax.set_ylabel("test accuracy")
    ax.set_title("interaction + MLP (salience reweight + hybrid)")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p1 = out / "mlp_interaction_weighted_mcdropout_v2_accuracy.png"
    fig.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p1}")

    fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (metric, name) in zip(axes, [("f1", "F1"), ("auprc", "AUPRC")]):
        for strat, st in styles.items():
            g = df[df["strategy"] == strat].groupby("n_labeled")[metric].agg(["mean", "std"]).reset_index()
            ax.plot(g["n_labeled"], g["mean"], color=st["c"], ls=st["ls"], lw=2, label=st["label"])
            ax.fill_between(g["n_labeled"], g["mean"] - g["std"], g["mean"] + g["std"], color=st["c"], alpha=0.1)
        ax.set_xlabel("labeled training samples")
        ax.set_ylabel(name)
        ax.legend(fontsize=6, loc="lower right")
        ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    p2 = out / "mlp_interaction_weighted_mcdropout_v2_f1_auprc.png"
    fig2.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"wrote {p2}")


if __name__ == "__main__":
    main()
