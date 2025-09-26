# grid_compare_tke_spectrum.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt

# --- Ton module style/plots (doit être importable) ---
from inference.plotting import apply_default_style
# --- Ton module metrics (signatures respectées) ---
from inference.metrics import (
    tke_from_vorticity_sequence,
    energy_spectrum_from_vorticity_sequence,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Couleurs compatibles avec ton style
COL_REAL = "#264653"   # Simulation (réel)
COL_PRED = "#E9C46A"   # Prediction
COL_INJ  = "#BC3754"   # Injections (graduations)

# -----------------------------
# IO helpers
# -----------------------------
def read_vorticity_domain_time_inj(
    h5_path: str | Path,
    key: str = "vorticity",
    *,
    expect_time: bool = False,
) -> tuple[np.ndarray, tuple[float,float,float,float], np.ndarray | None, np.ndarray | None]:
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
        # (T,1,H,W) -> (T,H,W)
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
# Calculs (appels à TES fonctions)
# -----------------------------
def compute_tke_series(
    omega: np.ndarray,
    domain: tuple[float,float,float,float],
) -> np.ndarray:
    """Retourne tke_t (T,) en appelant TA fonction."""
    tke_t, _ = tke_from_vorticity_sequence(
        omega, domain=domain, device=DEVICE, area_weighted_total=False
    )
    return np.asarray(tke_t, dtype=float)

def compute_spectrum(
    omega: np.ndarray,
    domain: tuple[float,float,float,float],
) -> tuple[np.ndarray, np.ndarray]:
    """Retourne (k, E_k) temps-moyenné via TA fonction."""
    k, Ek = energy_spectrum_from_vorticity_sequence(
        omega, domain=domain, device=DEVICE,
        nbins=64, subtract_time_mean=True, density=True
    )
    return np.asarray(k, dtype=float), np.asarray(Ek, dtype=float)


# -----------------------------
# Grille (chemins)
# -----------------------------
REAL_H5 = "/scratch/algo/aitalla/StageGitlab/data/DecayingTurbulence/decaying256/vorticity.h5"

GRID_PATHS: List[List[Tuple[str, str]]] = [
    # 1ut
    [
        ("1ut / free",  "/scratch/algo/aitalla/StageGitlab/predictions/256deacying_1ut_free/preds/vorticity.h5"),
        ("1ut / 1",     "/scratch/algo/aitalla/StageGitlab/predictions/256deacying_1ut_1inject/preds/vorticity.h5"),
        ("1ut / 4",     "/scratch/algo/aitalla/StageGitlab/predictions/256deacying_1ut_4inject/preds/vorticity.h5"),
        ("1ut / 8",     "/scratch/algo/aitalla/StageGitlab/predictions/256deacying_1ut_8inject/preds/vorticity.h5"),
        ("1ut / 16",    "/scratch/algo/aitalla/StageGitlab/predictions/256deacying_1ut_16inject/preds/vorticity.h5"),
        ("1ut / 32",    "/scratch/algo/aitalla/StageGitlab/predictions/256deacying_1ut_32inject/preds/vorticity.h5"),
    ],
    # 2ut
    [
        ("2ut / free",  "/scratch/algo/aitalla/StageGitlab/predictions/256deacying_2ut_free/preds/vorticity.h5"),
        ("2ut / 1",     "/scratch/algo/aitalla/StageGitlab/predictions/256deacying_2ut_1inject/preds/vorticity.h5"),
        ("2ut / 4",     "/scratch/algo/aitalla/StageGitlab/predictions/256deacying_2ut_4inject/preds/vorticity.h5"),
        ("2ut / 8",     "/scratch/algo/aitalla/StageGitlab/predictions/256deacying_2ut_8inject/preds/vorticity.h5"),
        ("2ut / 16",    "/scratch/algo/aitalla/StageGitlab/predictions/256deacying_2ut_16inject/preds/vorticity.h5"),
        ("2ut / 32",    "/scratch/algo/aitalla/StageGitlab/predictions/256deacying_2ut_32inject/preds/vorticity.h5"),
    ],
    # 4ut
    [
        ("4ut / free",  "/scratch/algo/aitalla/StageGitlab/predictions/256deacying_4ut_free/preds/vorticity.h5"),
        ("4ut / 1",     "/scratch/algo/aitalla/StageGitlab/predictions/256deacying_4ut_1inject/preds/vorticity.h5"),
        ("4ut / 4",     "/scratch/algo/aitalla/StageGitlab/predictions/256deacying_4ut_4inject/preds/vorticity.h5"),
        ("4ut / 8",     "/scratch/algo/aitalla/StageGitlab/predictions/256deacying_4ut_8inject/preds/vorticity.h5"),
        ("4ut / 16",    "/scratch/algo/aitalla/StageGitlab/predictions/256deacying_4ut_16inject/preds/vorticity.h5"),
        ("4ut / 32",    "/scratch/algo/aitalla/StageGitlab/predictions/256deacying_4ut_32inject/preds/vorticity.h5"),
    ],
    # 8ut
    [
        ("8ut / free",  "/scratch/algo/aitalla/StageGitlab/predictions/256deacying_8ut_free/preds/vorticity.h5"),
        ("8ut / 1",     "/scratch/algo/aitalla/StageGitlab/predictions/256deacying_8ut_1inject/preds/vorticity.h5"),
        ("8ut / 4",     "/scratch/algo/aitalla/StageGitlab/predictions/256deacying_8ut_4inject/preds/vorticity.h5"),
        ("8ut / 8",     "/scratch/algo/aitalla/StageGitlab/predictions/256deacying_8ut_8inject/preds/vorticity.h5"),
        ("8ut / 16",    "/scratch/algo/aitalla/StageGitlab/predictions/256deacying_8ut_16inject/preds/vorticity.h5"),
        ("8ut / 32",    "/scratch/algo/aitalla/StageGitlab/predictions/256deacying_8ut_32inject/preds/vorticity.h5"),
    ],
    # Curriculum 8ut
    [
        ("Cur8 / free", "/scratch/algo/aitalla/StageGitlab/predictions/256_decaying_Curriculum_8ut_free/preds/vorticity.h5"),
        ("Cur8 / 1",    "/scratch/algo/aitalla/StageGitlab/predictions/256_decaying_Curriculum_8ut_1inject/preds/vorticity.h5"),
        ("Cur8 / 4",    "/scratch/algo/aitalla/StageGitlab/predictions/256_decaying_Curriculum_8ut_4inject/preds/vorticity.h5"),
        ("Cur8 / 8",    "/scratch/algo/aitalla/StageGitlab/predictions/256_decaying_Curriculum_8ut_8inject/preds/vorticity.h5"),
        ("Cur8 / 16",   "/scratch/algo/aitalla/StageGitlab/predictions/256_decaying_Curriculum_8ut_16inject/preds/vorticity.h5"),
        ("Cur8 / 32",   "/scratch/algo/aitalla/StageGitlab/predictions/256_decaying_Curriculum_8ut_32inject/preds/vorticity.h5"),
    ],
]

ROW_NAMES = ["N_ut=1", "N_ut=2", "N_ut=4", "N_ut=8", "Curriculum (N_ut=8)"]
COL_NAMES = ["H=∞ (free)", "H=1", "H=4", "H=8", "H=16", "H=32"]


# -----------------------------
# Outils de tracé (style identique)
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


# -----------------------------
# Figures
# -----------------------------
def plot_grid_tke(real_h5: str, grid_paths: List[List[Tuple[str,str]]], out_path: str = "grid_tke.png"):
    apply_default_style()  # style identique
    real_omega, real_dom, real_time, real_inj = read_vorticity_domain_time_inj(real_h5, expect_time=True)
    if real_time is None:
        raise RuntimeError("L'attribut 'time' doit exister dans la base réelle.")
    tke_real = compute_tke_series(real_omega, real_dom)

    nrows, ncols = len(grid_paths), len(grid_paths[0])
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.85*ncols, 1.35*nrows), sharex=False, sharey=False)

    # Noms mathématiques (lignes/colonnes)
    row_names = [r"$N_{\mathrm{ut}}=1$", r"$N_{\mathrm{ut}}=2$", r"$N_{\mathrm{ut}}=4$",
                 r"$N_{\mathrm{ut}}=8$", r"Curriculum ($N_{\mathrm{ut}}=8$)"]
    col_names = [r"$H=\infty$ (free)", r"$H=1$", r"$H=4$", r"$H=8$", r"$H=16$", r"$H=32$"]

    # Proxies pour légende globale (handles plus épais sans modifier les courbes)
    inj_proxy = None
    line_real_proxy = None
    line_pred_proxy = None

    for i in range(nrows):
        for j in range(ncols):
            _, pred_h5 = grid_paths[i][j]
            ax = axes[i, j]

            # Réel
            ax.plot(real_time, tke_real, linewidth=0.15, marker='x', markersize=0.3, markevery=1,
                    color=COL_REAL, label=None)
            draw_injections_as_rug(ax, real_time, real_inj)

            # Prédiction alignée sur l'axe de temps réel
            try:
                pred_omega, pred_dom, pred_time, pred_inj = read_vorticity_domain_time_inj(pred_h5)
                tke_pred = compute_tke_series(pred_omega, pred_dom)
                Tm = min(len(real_time), len(tke_pred))
                ax.plot(real_time[:Tm], tke_pred[:Tm], linewidth=0.15, marker='x', markersize=0.3, markevery=1,
                        color=COL_PRED, label=None)
                draw_injections_as_rug(ax, real_time[:Tm], pred_inj)
            except Exception as e:
                ax.text(0.5, 0.5, "ERR\n"+str(e), ha="center", va="center", fontsize=7, transform=ax.transAxes)

            # Titres de colonnes (en haut uniquement)
            if i == 0:
                ax.set_title(col_names[j], fontsize=10)

            # Labels axes (seulement bordure bas/gauche pour éviter surcharge)
            if i == nrows - 1:
                ax.set_xlabel("time(s)")
            if j == 0:
                ax.set_ylabel("TKE")

            ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.5)
            ax.minorticks_on()

            # Proxies pour légende globale (créés une seule fois)
            if line_real_proxy is None:
                line_real_proxy, = ax.plot([], [], linewidth=0.15, marker='x', markersize=0.3,
                                           color=COL_REAL, label="Simulation")
            if line_pred_proxy is None:
                line_pred_proxy, = ax.plot([], [], linewidth=0.15, marker='x', markersize=0.3,
                                           color=COL_PRED, label="Prediction")
            if inj_proxy is None:
                inj_proxy, = ax.plot([], [], linestyle="None", marker="|", markersize=6,
                                     markeredgewidth=0.6, color=COL_INJ, label="injections")

    # Étiquettes de lignes (au bord gauche)
    for i in range(nrows):
        y_center = (nrows - i - 0.5) / nrows
        fig.text(0.012, y_center, row_names[i],
                 va="center", ha="left", rotation=90, fontsize=10)

    # --- Zone supérieure : légende AU-DESSUS du super-label H ---
    handles = [line_real_proxy, line_pred_proxy, inj_proxy]
    labels  = ["Simulation", "Prediction", "injections"]
    # Place la légende très en haut
    leg = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.00),
                     ncol=3, frameon=False, fontsize=10)

    # Compat Matplotlib: legendHandles (camel) vs legend_handles (snake)
    handles_for_tweak = getattr(leg, "legendHandles", None)
    if handles_for_tweak is None:
        handles_for_tweak = getattr(leg, "legend_handles", [])
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

    # Super-labels d’axes (en dessous de la légende)
    fig.text(0.5, 0.972, r"Inference free-forecast horizon before truth injection ($H$)",
             ha="center", va="top", fontsize=11)
    fig.text(0.005, 0.5, r"Training rollout length ($N_{\mathrm{ut}}$)",
             ha="left", va="center", rotation=90, fontsize=11)

    # Réserve de l'espace en haut pour légende + super-label (évite superposition)
    fig.tight_layout(rect=[0.07, 0.06, 1.0, 0.90])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)




def plot_grid_spectrum(real_h5: str, grid_paths: List[List[Tuple[str,str]]], out_path: str = "grid_spectrum.png"):
    apply_default_style()
    real_omega, real_dom, _, _ = read_vorticity_domain_time_inj(real_h5)
    k_real, E_real = compute_spectrum(real_omega, real_dom)

    nrows, ncols = len(grid_paths), len(grid_paths[0])
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.85*ncols, 1.35*nrows), sharex=False, sharey=False)

    line_real_proxy = None
    line_pred_proxy = None

    for i in range(nrows):
        for j in range(ncols):
            _, pred_h5 = grid_paths[i][j]
            ax = axes[i, j]
            try:
                pred_omega, pred_dom, _, _ = read_vorticity_domain_time_inj(pred_h5)
                k_pred, E_pred = compute_spectrum(pred_omega, pred_dom)

                # Interp sur k_real si besoin
                if len(k_pred) != len(k_real) or not np.allclose(k_pred, k_real, rtol=1e-3, atol=1e-8):
                    E_pred_interp = np.interp(k_real, k_pred, E_pred, left=np.nan, right=np.nan)
                    ax.loglog(k_real, E_real, linewidth=0.4, color=COL_REAL, label=None)
                    ax.loglog(k_real, E_pred_interp, linewidth=0.4, color=COL_PRED, label=None)
                else:
                    ax.loglog(k_real, E_real, linewidth=0.4, color=COL_REAL, label=None)
                    ax.loglog(k_pred, E_pred, linewidth=0.4, color=COL_PRED, label=None)

                ax.set_xscale("log")
                ax.set_yscale("log")
            except Exception as e:
                ax.text(0.5, 0.5, "ERR\n"+str(e), ha="center", va="center", fontsize=7, transform=ax.transAxes)

            # Titres colonnes (H)
            if i == 0:
                ax.set_title(COL_NAMES[j], fontsize=10)
            # Labels axes
            if i == nrows - 1:
                ax.set_xlabel(r"Wavenumber $k$")
            if j == 0:
                ax.set_ylabel(r"$E(k)$")

            ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.5)
            ax.minorticks_on()

            if line_real_proxy is None:
                line_real_proxy, = ax.plot([], [], linewidth=0.4, color=COL_REAL, label="Simulation")
            if line_pred_proxy is None:
                line_pred_proxy, = ax.plot([], [], linewidth=0.4, color=COL_PRED, label="Prediction")

    # Étiquettes de lignes (N_ut)
    for i in range(nrows):
        y_center = (nrows - i - 0.5) / nrows
        fig.text(0.005, y_center, ROW_NAMES[i],
                 va="center", ha="left", rotation=90, fontsize=10)

    # Super-labels
    fig.text(0.5, 0.988, "Inference free-forecast horizon before truth injection (H)",
             ha="center", va="top", fontsize=11)
    fig.text(0.0, 0.5, "Training rollout length (N_ut)",
             ha="left", va="center", rotation=90, fontsize=11)

    # Légende unique (handles plus visibles)
    handles = [line_real_proxy, line_pred_proxy]
    labels  = ["Simulation", "Prediction"]
    leg = fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=10)
    for h in leg.legendHandles:
        if hasattr(h, "set_linewidth"):
            h.set_linewidth(1.2)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    plot_grid_tke(REAL_H5, GRID_PATHS, out_path="grid_tke3.png")
    #plot_grid_spectrum(REAL_H5, GRID_PATHS, out_path="grid_spectrum.png")
    print("Saved: grid_tke3.png, grid_spectrum.png")
