# thin wrapper: reads al_comparison_results.csv, plots mlp-only accuracy vs labels

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import paths  # noqa: E402


def main() -> None:
    out = paths.artifact_dir
    csv_path = out / "al_comparison_results.csv"
    df = pd.read_csv(csv_path)
    mlp = df[df["model"] == "mlp_mcdropout"]  # strip rf rows — paper figure is mlp-only
    fig, ax = plt.subplots(figsize=(9, 5.5))
    styles = {
        "random": {"color": "#3498db", "label": "MLP + passive (random)"},
        "uncertainty": {"color": "#2ecc71", "label": "MLP + MC Dropout uncertainty"},
    }
    for strat in ("random", "uncertainty"):
        sub = mlp[mlp["strategy"] == strat]
        g = sub.groupby("n_labeled")["accuracy"].agg(["mean", "std"]).reset_index()
        st = styles[strat]
        ax.plot(g["n_labeled"], g["mean"], color=st["color"], lw=2, label=st["label"])
        ax.fill_between(g["n_labeled"], g["mean"] - g["std"], g["mean"] + g["std"], color=st["color"], alpha=0.12)  # ±1 std over seeds
    ax.set_xlabel("labeled training samples")
    ax.set_ylabel("test accuracy")
    ax.set_title("MLP + MC Dropout: passive vs uncertainty")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.suptitle("5 seeds | 4608-d | seed 100 | +96/round | pool exhausted", fontsize=9, y=0.02)
    fig.tight_layout()
    fp = out / "mlp_mcdropout_vs_random.png"
    fig.savefig(fp, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fp}")


if __name__ == "__main__":
    main()
