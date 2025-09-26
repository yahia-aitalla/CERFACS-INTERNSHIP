# losses_grid_fixed.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------- Style "paper" ----------
def apply_paper_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 300,
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.frameon": False,
        "grid.linestyle": ":",
        "grid.linewidth": 0.45,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    })

# ---------- I/O ----------
def read_metrics_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"global_epoch", "stage_n", "epoch_in_stage", "loss_epoch"}
    miss = need.difference(df.columns)
    if miss:
        raise ValueError(f"{path} missing columns: {sorted(miss)}")
    return df.sort_values("global_epoch").reset_index(drop=True)

# ---------- Plot helpers ----------
def plot_loss(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    ax.plot(df["global_epoch"], df["loss_epoch"], lw=1.0, alpha=0.95)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, which="both", alpha=0.55)

def plot_curriculum(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    # couleurs fixes par stage
    stage_colors = {1: "#1f77b4", 2: "#2ca02c", 4: "#ff7f0e", 8: "#d62728"}
    e = df["global_epoch"].to_numpy()
    y = df["loss_epoch"].to_numpy()
    s = df["stage_n"].astype(int).to_numpy()

    # tracer en segments continus par stage
    start = 0
    for i in range(1, len(df) + 1):
        if i == len(df) or s[i] != s[i-1]:
            st = int(s[start])
            ax.plot(e[start:i], y[start:i], lw=1.1, color=stage_colors.get(st, "k"),
                    label=f"Stage {st}")
            start = i

    # marquer ruptures de stage (fines lignes)
    edges = df.index[df["stage_n"].diff().fillna(0) != 0].tolist()[1:]
    for k in edges:
        ax.axvline(e[k], color="0.75", lw=0.6, ls="--", alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, which="both", alpha=0.55)
    # légende ordonnée
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: int(labels[i].split()[-1]))
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="best", ncol=4, handlelength=2.2)

# ---------- Main figure ----------
def make_losses_grid(
    p64: str | Path = "/scratch/algo/aitalla/StageGitlab/runs/offline/decaying8step/logs/metrics_epoch.csv",
    p256: str | Path = "/scratch/algo/aitalla/StageGitlab/runs/offline/256decaying8step/logs/metrics_epoch.csv",
    pcurr: str | Path = "/scratch/algo/aitalla/StageGitlab/runs/offline/256decayingCurric/logs/metrics_epoch.csv",
    out_path: str | Path = "losses_grid.png",
    figsize=(12.0, 6.8),
) -> Path:
    apply_paper_style()

    df64  = read_metrics_csv(p64)     # N_ut=8, 64x64, no curriculum
    df256 = read_metrics_csv(p256)    # N_ut=8, 256x256, no curriculum
    dfcur = read_metrics_csv(pcurr)   # N_ut=8, 256x256, curriculum stages 1/2/4/8

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.38, wspace=0.28)

    ax64  = fig.add_subplot(gs[0, 0])
    ax256 = fig.add_subplot(gs[0, 1])
    axcur = fig.add_subplot(gs[1, :])     # <- occupe toute la largeur, centré et large

    plot_loss(
        ax64, df64,
        r"Decaying turbulence — $64\times64$, $N_{\mathrm{ut}}=8$ (training loss)"
    )
    plot_loss(
        ax256, df256,
        r"Decaying turbulence — $256\times256$, $N_{\mathrm{ut}}=8$ (training loss)"
    )
    plot_curriculum(
        axcur, dfcur,
        r"Decaying turbulence — $256\times256$ with curriculum (stages $1\!\to\!2\!\to\!4\!\to\!8$)"
    )

    out_path = Path(out_path)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path

if __name__ == "__main__":
    make_losses_grid()
    print("Saved: losses_grid.png")
