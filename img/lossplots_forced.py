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

###################################
def plot_curriculum(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    # Couleurs par stage (tu gardes les tiennes)
    stage_colors = {1: "#d627ca", 2: "#2ca02c", 4: "#ff7f0e", 8: "#1f77b4"}

    e = df["global_epoch"].to_numpy()
    y = df["loss_epoch"].to_numpy()
    s = df["stage_n"].astype(int).to_numpy()

    lw = 1.1

    # Tracer des segments *continus* : on inclut le point de frontière dans le segment suivant
    start = 0
    for i in range(1, len(df)):
        if s[i] != s[i-1]:
            st = int(s[start])
            # inclut e[i] et y[i] pour éviter toute discontinuité visuelle
            ax.plot(e[start:i+1], y[start:i+1], lw=lw, color=stage_colors.get(st, "k"),
                    label=f"Stage {st}")
            start = i
    # Dernier segment jusqu'à la fin
    st = int(s[start])
    ax.plot(e[start:], y[start:], lw=lw, color=stage_colors.get(st, "k"),
            label=f"Stage {st}")

    # Plus de lignes verticales (segments déjà continus)
    # axvline supprimé intentionnellement

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, which="both", alpha=0.55)

    # Légende ordonnée par numéro de stage
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: int(labels[i].split()[-1]))
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="best", ncol=4, handlelength=2.2)

##################
def plot_curriculum_1(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    stage_colors = {1: "#d627ca", 2: "#2ca02c", 4: "#ff7f0e", 8: "#1f77b4"}
    e = df["global_epoch"].to_numpy()
    y = df["loss_epoch"].to_numpy()
    s = df["stage_n"].astype(int).to_numpy()

    start = 0
    for i in range(1, len(df) + 1):
        if i == len(df) or s[i] != s[i-1]:
            st = int(s[start])
            ax.plot(e[start:i], y[start:i], lw=1.1, color=stage_colors.get(st, "k"),
                    label=f"Stage {st}")
            start = i

    edges = df.index[df["stage_n"].diff().fillna(0) != 0].tolist()[1:]
    for k in edges:
        ax.axvline(e[k], color="0.75", lw=0.6, ls="--", alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, which="both", alpha=0.55)
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: int(labels[i].split()[-1]))
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="best", ncol=4, handlelength=2.2)




#################
def color_title_fragment(ax: plt.Axes, whole_title_prefix: str, colored_piece: str,
                         whole_title_suffix: str, color: str,
                         y: float = 1.01, size: int = 11) -> None:
    """
    Ré-écrit le titre comme trois fragments, centrés au-dessus de l'axe.
    Seul `colored_piece` est coloré. Taille de police légèrement réduite.
    """
    ax.set_title("")  # efface le titre existant
    fig = ax.figure
    renderer = fig.canvas.get_renderer()

    # Mesurer les largeurs en coord. figure pour centrer précisément
    # (on crée des textes temporaires hors champ, puis on supprime)
    tmp1 = ax.text(0, -10, whole_title_prefix, fontsize=size, transform=ax.transAxes)
    tmp2 = ax.text(0, -10, colored_piece,       fontsize=size, transform=ax.transAxes)
    tmp3 = ax.text(0, -10, whole_title_suffix,  fontsize=size, transform=ax.transAxes)
    fig.canvas.draw_idle()
    w1 = tmp1.get_window_extent(renderer).width / ax.bbox.width
    w2 = tmp2.get_window_extent(renderer).width / ax.bbox.width
    w3 = tmp3.get_window_extent(renderer).width / ax.bbox.width
    for t in (tmp1, tmp2, tmp3):
        t.remove()

    total_w = w1 + w2 + w3
    x0 = 0.5 - 0.5 * total_w  # point de départ pour centrer

    # Dessin définitif, centré
    t1 = ax.text(x0, y, whole_title_prefix, transform=ax.transAxes,
                 fontsize=size, va="bottom", ha="left")
    x2 = x0 + w1
    t2 = ax.text(x2, y, colored_piece, transform=ax.transAxes,
                 fontsize=size, va="bottom", ha="left", color=color)
    x3 = x2 + w2
    ax.text(x3, y, whole_title_suffix, transform=ax.transAxes,
            fontsize=size, va="bottom", ha="left")
########################@







# ---------- (Ajout minime) : colorer juste un fragment du titre sans rien changer d'autre ----------
def color_title_fragment1(ax: plt.Axes, whole_title_prefix: str, colored_piece: str,
                         whole_title_suffix: str, color: str, y: float = 1.02, size: int = 12) -> None:
    """Ré-écrit le titre comme trois fragments pour colorer seulement `colored_piece`."""
    ax.set_title("")
    fig = ax.figure

    t1 = ax.text(0.02, y, whole_title_prefix, transform=ax.transAxes,
                 fontsize=size, va="bottom", ha="left")
    fig.canvas.draw()
    w1 = t1.get_window_extent(fig.canvas.get_renderer()).width / ax.bbox.width

    x2 = 0.02 + w1
    t2 = ax.text(x2, y, colored_piece, transform=ax.transAxes,
                 fontsize=size, va="bottom", ha="left", color=color)
    fig.canvas.draw()
    w2 = t2.get_window_extent(fig.canvas.get_renderer()).width / ax.bbox.width

    x3 = 0.02 + w1 + w2
    ax.text(x3, y, whole_title_suffix, transform=ax.transAxes,
            fontsize=size, va="bottom", ha="left")

# ---------- Main figure ----------
def make_losses_grid(
    p64: str | Path = "/scratch/algo/aitalla/StageGitlab/runs/offline/forced8step/logs/metrics_epoch.csv",
    p256: str | Path = "/scratch/algo/aitalla/StageGitlab/runs/offline/256forced8step/logs/metrics_epoch.csv",
    pcurr: str | Path = "/scratch/algo/aitalla/StageGitlab/metrics_epoch copy.csv",
    out_path: str | Path = "losses_grid_forced2.png",
    figsize=(12.0, 6.8),
) -> Path:
    apply_paper_style()

    df64  = read_metrics_csv(p64)
    df256 = read_metrics_csv(p256)
    dfcur = read_metrics_csv(pcurr)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.38, wspace=0.28)

    ax64  = fig.add_subplot(gs[0, 0])
    ax256 = fig.add_subplot(gs[0, 1])
    axcur = fig.add_subplot(gs[1, :])
  # <-- si cette ligne provoque une erreur, remets fig.add_subplot

    title64  = r"Forced turbulence — $64\times64$, $N_{\mathrm{ut}}=8$ (training loss)"
    title256 = r"Forced turbulence — $256\times256$, $N_{\mathrm{ut}}=8$ (training loss)"
    plot_loss(ax64,  df64,  title64)
    plot_loss(ax256, df256, title256)
    plot_curriculum(
        axcur, dfcur,
        r"Forced turbulence — $256\times256$ with curriculum (stages $1\!\to\!2\!\to\!4\!\to\!8$)"
    )

    # *** seule modif : recoloration du fragment numérique dans les titres ***
    col64  = "#0c090f"  # bleu pour 64×64
    col256 = "#e90707"  # rouge pour 256×256

    color_title_fragment(
        ax64,
        whole_title_prefix="Forced turbulence — ",
        colored_piece=r"$64\times64$",
        whole_title_suffix=r", $N_{\mathrm{ut}}=8$ (training loss)",
        color=col64,
    )
    color_title_fragment(
        ax256,
        whole_title_prefix="Forced turbulence — ",
        colored_piece=r"$256\times256$",
        whole_title_suffix=r", $N_{\mathrm{ut}}=8$ (training loss)",
        color=col256,
    )
    # === AJOUT pour le curriculum : même couleur que le 256×256 sans curriculum ===
    color_title_fragment(
        axcur,
        whole_title_prefix="Forced turbulence — ",
        colored_piece=r"$256\times256$",
        whole_title_suffix=r" with curriculum (stages $1\!\to\!2\!\to\!4\!\to\!8$)",
        color=col256,
    )

    out_path = Path(out_path)
    fig.savefig(out_path, dpi=350)
    plt.close(fig)
    return out_path

if __name__ == "__main__":
    make_losses_grid()
    print("Saved: losses_grid.png")
