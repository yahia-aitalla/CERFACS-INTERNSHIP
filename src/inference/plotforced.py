# tke_rows_forced_free.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt

# --- Style identique ---
from inference.plotting import apply_default_style
# --- Fonctions utilisateur (signatures respectées) ---
from inference.metrics import tke_from_vorticity_sequence

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Couleurs
COL_REAL = "#264653"   # Simulation (réel)
COL_PRED = "#E9C46A"   # Prediction
COL_INJ  = "#BC3754"   # Injections (graduations)

# -----------------------------
# IO helper
# -----------------------------
def read_vorticity_domain_time_inj(
    h5_path: str | Path,
    key: str = "vorticity",
    *,
    expect_time: bool = False,
):
    """
    Retourne:
      omega (T,H,W), domain=(x_min,x_max,y_min,y_max),
      time (T,) ou None, injection_indices (K,) ou None.
    """
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        if key not in f:
            raise KeyError(f"Key '{key}' not in {h5_path}. Available: {list(f.keys())}")
        dset = f[key]
        arr = np.asarray(dset[...])
        if arr.ndim == 4 and arr.shape[1] == 1:
            arr = arr[:, 0, :, :]
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.ndim != 3:
            raise ValueError(f"Expected (T,H,W) or (T,1,H,W). Got {arr.shape}")

        attrs = dset.attrs
        if all(k in attrs for k in ("x_min","x_max","y_min","y_max")):
            domain = (float(attrs["x_min"]), float(attrs["x_max"]),
                      float(attrs["y_min"]), float(attrs["y_max"]))
        else:
            domain = (0.0, 2*np.pi, 0.0, 2*np.pi)

        time = None
        if "time" in attrs:
            time = np.asarray(attrs["time"], dtype=float)
        elif expect_time:
            raise KeyError(f"'time' attribute missing in {h5_path} /{key}")

        inj = None
        if "injection_indices" in attrs:
            inj = np.asarray(attrs["injection_indices"], dtype=int)

    return arr.astype(np.float32), domain, time, inj


# -----------------------------
# Calcul TKE (appel ta fonction)
# -----------------------------
def compute_tke_series(
    omega: np.ndarray,
    domain: tuple[float,float,float,float],
) -> np.ndarray:
    tke_t, _ = tke_from_vorticity_sequence(
        omega, domain=domain, device=DEVICE, area_weighted_total=False
    )
    return np.asarray(tke_t, dtype=float)


# -----------------------------
# Rug des injections
# -----------------------------
def draw_injections_as_rug(ax: plt.Axes, times: np.ndarray, inj_idx: np.ndarray | None) -> None:
    """Graduations rouges fines au bas de l'axe du temps, si indices disponibles."""
    if inj_idx is None or len(inj_idx) == 0:
        return
    inj_idx = np.asarray(inj_idx, dtype=int)
    inj_idx = inj_idx[(inj_idx >= 0) & (inj_idx < len(times))]
    if inj_idx.size == 0:
        return
    inj_times = times[inj_idx]
    trans = ax.get_xaxis_transform()  # x en data, y en [0,1]
    ax.vlines(
        inj_times, 0.0, 0.015,
        transform=trans,
        colors=COL_INJ,
        linewidths=0.2,
        alpha=0.85,
        zorder=5,
        clip_on=False,
    )



def plot_tke_rows_forced_free(
    real_h5: str,
    pred_h5_list: List[Tuple[str, str]],  # [(label_row, path_h5)] pour N_ut=1,2,4,8 (dans cet ordre)
    out_path: str = "tke_rows_forced_free.png",
    width_scale: float = 1.7,             # élargir la figure (1.7~2.2 pour des séries très serrées)
) -> None:
    from matplotlib.lines import Line2D
    apply_default_style()

    # --- Données réelles (avec 'time') ---
    real_omega, real_dom, real_time, real_inj = read_vorticity_domain_time_inj(real_h5, expect_time=True)
    if real_time is None:
        raise RuntimeError("L'attribut 'time' doit exister dans la base réelle.")
    tke_real = compute_tke_series(real_omega, real_dom)

    # --- Figure large : 4 rangées × 1 colonne ---
    nrows = len(pred_h5_list)
    ncols = 1
    base_w, base_h = 7.5, 1.7
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(width_scale * base_w, nrows * base_h),
        sharex=False, sharey=False
    )
    axes = np.asarray(axes).reshape(nrows, ncols)

    # Styles des courbes (visibilité)
    lw, ms, mew, alp = 0.45, 0.9, 0.7, 0.95

    # Étiquettes math
    row_math = [r"$N_{\mathrm{ut}}=1$", r"$N_{\mathrm{ut}}=2$", r"$N_{\mathrm{ut}}=4$", r"$N_{\mathrm{ut}}=8$"]

    for i, (row_label, pred_h5) in enumerate(pred_h5_list):
        ax = axes[i, 0]

        # Réel
        ax.plot(real_time, tke_real, color=COL_REAL, linewidth=lw,
                marker='x', markersize=ms, markeredgewidth=mew, markevery=1, alpha=alp)
        draw_injections_as_rug(ax, real_time, real_inj)

        # Prédiction
        try:
            pred_omega, pred_dom, pred_time, pred_inj = read_vorticity_domain_time_inj(pred_h5)
            tke_pred = compute_tke_series(pred_omega, pred_dom)
            Tm = min(len(real_time), len(tke_pred))
            ax.plot(real_time[:Tm], tke_pred[:Tm], color=COL_PRED, linewidth=lw,
                    marker='x', markersize=ms, markeredgewidth=mew, markevery=1, alpha=alp)
            draw_injections_as_rug(ax, real_time[:Tm], pred_inj)
        except Exception as e:
            print(f"[TKE row {row_label}] {pred_h5}: {e}")
            ax.text(0.02, 0.80, "ERR", transform=ax.transAxes, fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.9, edgecolor="none"))

        # >>> Étiquette de ligne : légèrement AU-DESSUS de l'axe (plus haut, pas de chevauchement)
        ax.text(-0.01, 1.04, row_math[i], transform=ax.transAxes,
                ha="left", va="bottom", fontsize=9, clip_on=False)

        # Axes : xlabel seulement sur le dernier
        if i == nrows - 1:
            ax.set_xlabel("time(s)")
        ax.set_ylabel("TKE")

        ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.5)
        ax.minorticks_on()

    # Légende globale (styles identiques à la courbe)
    legend_handles = [
        Line2D([], [], color=COL_REAL, marker='x', markersize=ms, markeredgewidth=mew,
               linewidth=lw, alpha=alp, label="Simulation"),
        Line2D([], [], color=COL_PRED, marker='x', markersize=ms, markeredgewidth=mew,
               linewidth=lw, alpha=alp, label="Prediction"),
        #Line2D([], [], color=COL_INJ, linestyle="None", marker='|',
        #       markersize=7, markeredgewidth=0.9, label="injections"),
    ]
    fig.legend(
        legend_handles, [h.get_label() for h in legend_handles],
        loc="upper center", bbox_to_anchor=(0.5, 1.01),
        ncol=3, frameon=False, fontsize=10,
        borderpad=0.1, handletextpad=0.6, columnspacing=1.2, handlelength=2.2
    )

    # Pas de suptitle ; marges : un peu d’air en haut pour la légende et les étiquettes relevées
    fig.subplots_adjust(left=0.10, right=0.995, bottom=0.08, top=0.94, hspace=0.30)

    fig.savefig(out_path, dpi=350)
    plt.close(fig)

######################################################
def plot_tke_rows_forced_free_III(
    real_h5: str,
    pred_h5_list: List[Tuple[str, str]],  # [(label_row, path_h5)] pour N_ut=1,2,4,8 (dans cet ordre)
    out_path: str = "tke_rows_forced_free.png",
    width_scale: float = 1.7,             # élargir la figure (1.7~2.2 pour des séries très serrées)
) -> None:
    from matplotlib.lines import Line2D

    apply_default_style()

    # --- Données réelles (avec 'time') ---
    real_omega, real_dom, real_time, real_inj = read_vorticity_domain_time_inj(real_h5, expect_time=True)
    if real_time is None:
        raise RuntimeError("L'attribut 'time' doit exister dans la base réelle.")
    tke_real = compute_tke_series(real_omega, real_dom)

    # --- Figure large : 4 rangées × 1 colonne ---
    nrows = len(pred_h5_list)
    ncols = 1
    base_w, base_h = 7.5, 1.7                      # taille par axe (en pouces)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(width_scale * base_w, nrows * base_h),
        sharex=False, sharey=False
    )
    axes = np.asarray(axes).reshape(nrows, ncols)

    # Styles des courbes (visibilité améliorée)
    lw = 0.45
    ms = 0.9
    mew = 0.7
    alp = 0.95

    # Étiquettes math des lignes
    row_math = [r"$N_{\mathrm{ut}}=1$", r"$N_{\mathrm{ut}}=2$", r"$N_{\mathrm{ut}}=4$", r"$N_{\mathrm{ut}}=8$"]

    for i, (row_label, pred_h5) in enumerate(pred_h5_list):
        ax = axes[i, 0]

        # --- Réel ---
        ax.plot(
            real_time, tke_real,
            color=COL_REAL, linewidth=lw, marker='x', markersize=ms, markeredgewidth=mew,
            markevery=1, alpha=alp
        )
        draw_injections_as_rug(ax, real_time, real_inj)

        # --- Prédiction (alignée temps réel) ---
        try:
            pred_omega, pred_dom, pred_time, pred_inj = read_vorticity_domain_time_inj(pred_h5)
            tke_pred = compute_tke_series(pred_omega, pred_dom)
            Tm = min(len(real_time), len(tke_pred))
            ax.plot(
                real_time[:Tm], tke_pred[:Tm],
                color=COL_PRED, linewidth=lw, marker='x', markersize=ms, markeredgewidth=mew,
                markevery=1, alpha=alp
            )
            draw_injections_as_rug(ax, real_time[:Tm], pred_inj)
        except Exception as e:
            print(f"[TKE row {row_label}] {pred_h5}: {e}")
            ax.text(0.02, 0.80, "ERR", transform=ax.transAxes, fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.9, edgecolor="none"))

        # Titre de ligne à gauche
        ax.set_title(row_math[i], fontsize=9, pad=1.2, loc="left")

        # Axes : ylabel partout, xlabel seulement sur le DERNIER
        if i == nrows - 1:
            ax.set_xlabel("time(s)")
        ax.set_ylabel("TKE")

        ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.5)
        ax.minorticks_on()

    # --- Légende globale : styles EXACTS via Line2D ---
    legend_handles = [
        Line2D([], [], color=COL_REAL, marker='x', markersize=ms, markeredgewidth=mew,
               linewidth=lw, alpha=alp, label="Simulation"),
        Line2D([], [], color=COL_PRED, marker='x', markersize=ms, markeredgewidth=mew,
               linewidth=lw, alpha=alp, label="Prediction"),
        Line2D([], [], color=COL_INJ, linestyle="None", marker='|',
               markersize=7, markeredgewidth=0.9, label="injections"),
    ]
    fig.legend(
        legend_handles, [h.get_label() for h in legend_handles],
        loc="upper center", bbox_to_anchor=(0.5, 1.01),
        ncol=3, frameon=False, fontsize=10,
        borderpad=0.1, handletextpad=0.6, columnspacing=1.2, handlelength=2.2
    )

    # Pas de suptitle (demandé)
    # Ajustement des marges (un peu d’air en haut pour la légende)
    fig.subplots_adjust(left=0.10, right=0.995, bottom=0.08, top=0.94, hspace=0.30)

    fig.savefig(out_path, dpi=350)
    plt.close(fig)

#########################################################

def plot_tke_rows_forced_free_II(
    real_h5: str,
    pred_h5_list: List[Tuple[str, str]],  # [(label_row, path_h5)] pour N_ut = 1,2,4,8
    out_path: str = "tke_rows_forced_free.png",
    width_scale: float = 1.8,            # plus large pour des séries serrées/bruitées
) -> None:
    apply_default_style()

    # --- Réel ---
    real_omega, real_dom, real_time, real_inj = read_vorticity_domain_time_inj(real_h5, expect_time=True)
    if real_time is None:
        raise RuntimeError("L'attribut 'time' doit exister dans la base réelle.")
    tke_real = compute_tke_series(real_omega, real_dom)

    # --- Figure 4x1 large ---
    nrows = len(pred_h5_list)
    ncols = 1
    base_w, base_h = 7.2, 1.65  # un peu plus grand qu'avant
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(width_scale * base_w, nrows * base_h),
        sharex=True, sharey=False
    )
    axes = np.asarray(axes).reshape(nrows, ncols)

    # Légende globale (proxies)
    inj_proxy = None
    line_real_proxy = None
    line_pred_proxy = None

    row_math = [r"$N_{\mathrm{ut}}=1$", r"$N_{\mathrm{ut}}=2$", r"$N_{\mathrm{ut}}=4$", r"$N_{\mathrm{ut}}=8$"]

    for i, (row_label, pred_h5) in enumerate(pred_h5_list):
        ax = axes[i, 0]

        # --- Simulation (réel) : lignes un peu plus visibles ---
        ax.plot(
            real_time, tke_real,
            color=COL_REAL, linewidth=0.35, marker='x', markersize=0.6, markevery=1, alpha=0.95
        )
        draw_injections_as_rug(ax, real_time, real_inj)

        # --- Prédiction ---
        try:
            pred_omega, pred_dom, pred_time, pred_inj = read_vorticity_domain_time_inj(pred_h5)
            tke_pred = compute_tke_series(pred_omega, pred_dom)
            Tm = min(len(real_time), len(tke_pred))
            ax.plot(
                real_time[:Tm], tke_pred[:Tm],
                color=COL_PRED, linewidth=0.35, marker='x', markersize=0.6, markevery=1, alpha=0.95
            )
            draw_injections_as_rug(ax, real_time[:Tm], pred_inj)
        except Exception as e:
            print(f"[TKE row {row_label}] {pred_h5}: {e}")
            ax.text(0.02, 0.82, "ERR", transform=ax.transAxes, fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.9, edgecolor="none"))

        # Titre de ligne à gauche (en maths)
        ax.set_title(row_math[i], fontsize=9, pad=1.0, loc="left")

        # Axes : Y sur chaque rangée, X seulement sur la dernière
        ax.set_ylabel("TKE")
        if i == nrows - 1:
            ax.set_xlabel("time(s)")
            ax.tick_params(labelbottom=True)
        else:
            ax.set_xlabel("")           # pas de label
            ax.tick_params(labelbottom=False)

        ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.55)
        ax.minorticks_on()

        # Proxies pour légende globale (sans changer les courbes)
        if line_real_proxy is None:
            line_real_proxy, = ax.plot([], [], color=COL_REAL, linewidth=1.0, label="Simulation")
        if line_pred_proxy is None:
            line_pred_proxy, = ax.plot([], [], color=COL_PRED, linewidth=1.0, label="Prediction")
        if inj_proxy is None:
            inj_proxy, = ax.plot([], [], linestyle="None", marker="|", markersize=7,
                                 markeredgewidth=0.9, color=COL_INJ, label="injections")

    # Légende globale en haut
    handles = [line_real_proxy, line_pred_proxy, inj_proxy]
    labels  = ["Simulation", "Prediction", "injections"]
    leg = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01),
                     ncol=3, frameon=False, fontsize=10,
                     borderpad=0.1, handletextpad=0.6, columnspacing=1.2)

    # Pas de titre global demandé -> on ne met pas fig.suptitle

    # Marges/espaces : rapprocher les rangées, laisser un peu de place à la légende
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.08, top=0.94, hspace=0.18)

    fig.savefig(out_path, dpi=350)
    plt.close(fig)
##############################################################
# -----------------------------
# Figure 4×1 (large)
# -----------------------------
def plot_tke_rows_forced_free_I(
    real_h5: str,
    pred_h5_list: List[Tuple[str, str]],  # [(label_row, path_h5), ...] pour N_ut = 1,2,4,8
    out_path: str = "tke_rows_forced_free.png",
    width_scale: float = 1.3,            # >1 : plus large ; <1 : plus compact
) -> None:
    """
    Crée une image large avec 4 sous-plots empilés (N_ut = 1,2,4,8).
    Chaque sous-plot : TKE réelle vs TKE prédite (free forecast).
    """
    apply_default_style()

    # Réel (avec temps depuis l'attribut 'time')
    real_omega, real_dom, real_time, real_inj = read_vorticity_domain_time_inj(real_h5, expect_time=True)
    if real_time is None:
        raise RuntimeError("L'attribut 'time' doit exister dans la base réelle.")
    tke_real = compute_tke_series(real_omega, real_dom)

    # Figure large : 4 lignes, 1 colonne
    nrows = len(pred_h5_list)
    ncols = 1
    base_w, base_h = 6.5, 1.6   # base par axe (en pouces)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(width_scale * base_w, nrows * base_h),
        sharex=False, sharey=False
    )
    axes = np.asarray(axes).reshape(nrows, ncols)

    # Légende globale (proxies)
    inj_proxy = None
    line_real_proxy = None
    line_pred_proxy = None

    # Étiquettes de lignes en maths
    row_math = [r"$N_{\mathrm{ut}}=1$", r"$N_{\mathrm{ut}}=2$", r"$N_{\mathrm{ut}}=4$", r"$N_{\mathrm{ut}}=8$"]

    for i, (row_label, pred_h5) in enumerate(pred_h5_list):
        ax = axes[i, 0]

        # réel
        ax.plot(real_time, tke_real, linewidth=0.15, marker='x', markersize=0.3, markevery=1,
                color=COL_REAL, label=None)
        draw_injections_as_rug(ax, real_time, real_inj)

        # prédiction (alignée sur temps réel)
        try:
            pred_omega, pred_dom, pred_time, pred_inj = read_vorticity_domain_time_inj(pred_h5)
            tke_pred = compute_tke_series(pred_omega, pred_dom)
            Tm = min(len(real_time), len(tke_pred))
            ax.plot(real_time[:Tm], tke_pred[:Tm], linewidth=0.15, marker='x', markersize=0.3, markevery=1,
                    color=COL_PRED, label=None)
            draw_injections_as_rug(ax, real_time[:Tm], pred_inj)
        except Exception as e:
            print(f"[TKE row {row_label}] {pred_h5}: {e}")
            ax.text(0.02, 0.80, "ERR", transform=ax.transAxes, fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.9, edgecolor="none"))

        # titre de ligne (math)
        ax.set_title(row_math[i], fontsize=9, pad=1.2, loc="left")

        # axes
        ax.set_xlabel("time(s)")
        ax.set_ylabel("TKE")

        ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.5)
        ax.minorticks_on()

        # proxies pour légende globale
        if line_real_proxy is None:
            line_real_proxy, = ax.plot([], [], linewidth=0.15, marker='x', markersize=0.3,
                                       color=COL_REAL, label="Simulation")
        if line_pred_proxy is None:
            line_pred_proxy, = ax.plot([], [], linewidth=0.15, marker='x', markersize=0.3,
                                       color=COL_PRED, label="Prediction")
        if inj_proxy is None:
            inj_proxy, = ax.plot([], [], linestyle="None", marker="|", markersize=6,
                                 markeredgewidth=0.6, color=COL_INJ, label="injections")

    # Légende globale en haut
    handles = [line_real_proxy, line_pred_proxy, inj_proxy]
    labels  = ["Simulation", "Prediction", "injections"]
    leg = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
                     ncol=3, frameon=False, fontsize=10,
                     borderpad=0.1, handletextpad=0.5, columnspacing=1.0)
    handles_for_tweak = getattr(leg, "legendHandles", None) or getattr(leg, "legend_handles", [])
    for h in handles_for_tweak:
        if hasattr(h, "set_linewidth"):
            try: h.set_linewidth(1.2)
            except Exception: pass
        if hasattr(h, "set_markersize"):
            try: h.set_markersize(8)
            except Exception: pass
        if hasattr(h, "set_markeredgewidth"):
            try: h.set_markeredgewidth(0.8)
            except Exception: pass

    # Titre global (optionnel)
    fig.suptitle("TKE — free forecast (no truth injection) per training rollout length", y=1.04, fontsize=11)

    # Marges : un peu plus de place à gauche/droite pour confort
    fig.subplots_adjust(left=0.10, right=0.995, bottom=0.08, top=0.92, hspace=0.28)

    fig.savefig(out_path, dpi=350)
    plt.close(fig)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    REAL_H5 = "/scratch/algo/aitalla/StageGitlab/data/ForcedTurbulence/forced256/vorticity.h5"
    PRED_LIST = [
        ("N_ut=1", "/scratch/algo/aitalla/StageGitlab/predictions/256forced_1ut_free/preds/vorticity.h5"),
        ("N_ut=2", "/scratch/algo/aitalla/StageGitlab/predictions/256forced_2ut_free/preds/vorticity.h5"),
        ("N_ut=4", "/scratch/algo/aitalla/StageGitlab/predictions/256forced_4ut_free/preds/vorticity.h5"),
        ("N_ut=8", "/scratch/algo/aitalla/StageGitlab/predictions/256forced_8ut_free/preds/vorticity.h5"),
    ]
    plot_tke_rows_forced_free(
        REAL_H5,
        PRED_LIST,
        out_path="tke_rows_forced_free.png",
        width_scale=1.6,  # mettre 1.8 ou 2.0 si tu veux encore plus large
    )
    print("Saved: tke_rows_forced_free.png")
