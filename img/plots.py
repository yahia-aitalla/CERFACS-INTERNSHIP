# grid_compare_tke_spectrum.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt

# --- Ton module plotting/style (fourni par toi) ---
from inference.plotting import apply_default_style  # assure-toi que ce module est dans PYTHONPATH
# --- Ton module metrics (fourni par toi, signatures respectées) ---
from inference.metrics import (
    tke_from_vorticity_sequence,
    energy_spectrum_from_vorticity_sequence,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Couleurs proches de ton style
COL_REAL = "#264653"
COL_PRED = "#E9C46A"
COL_INJ  = "#BC3754"

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
            # fallback: domaine périodique standard
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
    """Retourne tke_t (T,) en appelant TA fonction avec device & domain."""
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


def draw_injections_as_rug(ax: plt.Axes, times: np.ndarray, inj_idx: np.ndarray) -> None:
    """Graduations rouges fines le long de l'axe du temps (style identique à ta fonction)."""
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
        colors="red",
        linewidths=0.2,
        alpha=0.85,
        zorder=5,
        clip_on=False,
    )

def plot_grid_tke(real_h5: str, grid_paths: List[List[Tuple[str,str]]], out_path: str = "grid_tke.png"):
    apply_default_style()  # ton style
    real_omega, real_dom, real_time, real_inj = read_vorticity_domain_time_inj(real_h5, expect_time=True)
    if real_time is None:
        raise RuntimeError("L'attribut 'time' doit exister dans la base réelle.")
    tke_real = compute_tke_series(real_omega, real_dom)

    nrows, ncols = len(grid_paths), len(grid_paths[0])
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.85*ncols, 1.35*nrows), sharex=False, sharey=False)

    # Légende globale : on posera les handles ensuite
    added_real = False
    added_pred = False
    added_inj  = False

    for i in range(nrows):
        for j in range(ncols):
            label, pred_h5 = grid_paths[i][j]
            ax = axes[i, j]

            # TKE réel
            ax.plot(real_time, tke_real, linewidth=0.15, marker='x', markersize=0.3, markevery=1,
                    color=COL_REAL, label=None)
            # Injections réelles
            draw_injections_as_rug(ax, real_time, real_inj)

            try:
                pred_omega, pred_dom, pred_time, pred_inj = read_vorticity_domain_time_inj(pred_h5)
                # Aligner sur l'axe du temps réel (tronquage à la min longueur)
                tke_pred = compute_tke_series(pred_omega, pred_dom)
                Tm = min(len(real_time), len(tke_pred))
                ax.plot(real_time[:Tm], tke_pred[:Tm], linewidth=0.15, marker='x', markersize=0.3, markevery=1,
                        color=COL_PRED, label=None)
                # Injections prédiction (si présentes)
                draw_injections_as_rug(ax, real_time[:Tm], pred_inj)
            except Exception as e:
                ax.text(0.5, 0.5, "ERR\n"+str(e), ha="center", va="center", fontsize=7, transform=ax.transAxes)

            if i == 0:
                ax.set_title(label, fontsize=10)
            if i == nrows - 1:
                ax.set_xlabel("time(s)")
            if j == 0:
                ax.set_ylabel("TKE")

            ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.5)
            ax.minorticks_on()

            # Construit handles légende globale (une fois)
            if not added_real:
                line_real, = ax.plot([], [], linewidth=0.15, marker='x', markersize=0.3,
                                     color=COL_REAL, label="Simulation")
                added_real = True
            if not added_pred:
                line_pred, = ax.plot([], [], linewidth=0.15, marker='x', markersize=0.3,
                                     color=COL_PRED, label="Prediction")
                added_pred = True
            if not added_inj:
                # proxy vertical comme ta légende "injections"
                inj_proxy, = ax.plot([], [], linestyle="None", marker="|", markersize=6,
                                     markeredgewidth=0.6, color="red", label="injections")
                added_inj = True

        # fin j
    # fin i

    # Légende unique pour toute la grille
    handles = []
    labels = []
    if added_real: handles.append(line_real); labels.append("Simulation")
    if added_pred: handles.append(line_pred); labels.append("Prediction")
    if added_inj:  handles.append(inj_proxy); labels.append("injections")
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def plot_grid_spectrum(real_h5: str, grid_paths: List[List[Tuple[str,str]]], out_path: str = "grid_spectrum.png"):
    apply_default_style()
    real_omega, real_dom, _, _ = read_vorticity_domain_time_inj(real_h5)
    k_real, E_real = compute_spectrum(real_omega, real_dom)

    nrows, ncols = len(grid_paths), len(grid_paths[0])
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.85*ncols, 1.35*nrows), sharex=False, sharey=False)

    added_real = False
    added_pred = False

    for i in range(nrows):
        for j in range(ncols):
            label, pred_h5 = grid_paths[i][j]
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

            if i == 0:
                ax.set_title(label, fontsize=10)
            if i == nrows - 1:
                ax.set_xlabel(r"Wavenumber $k$")
            if j == 0:
                ax.set_ylabel(r"$E(k)$")

            ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.5)
            ax.minorticks_on()

            if not added_real:
                line_real, = ax.plot([], [], linewidth=0.4, color=COL_REAL, label="Simulation")
                added_real = True
            if not added_pred:
                line_pred, = ax.plot([], [], linewidth=0.4, color=COL_PRED, label="Prediction")
                added_pred = True

    handles = []
    labels = []
    if added_real: handles.append(line_real); labels.append("Simulation")
    if added_pred: handles.append(line_pred); labels.append("Prediction")
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    plot_grid_tke(REAL_H5, GRID_PATHS, out_path="grid_tke.png")
    #plot_grid_spectrum(REAL_H5, GRID_PATHS, out_path="grid_spectrum.png")
    print("Saved: grid_tke.png, grid_spectrum.png")
