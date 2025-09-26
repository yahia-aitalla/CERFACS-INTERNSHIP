from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch


__all__ = [
    "apply_default_style",
    "plot_tke_time_series",
    "plot_energy_spectrum",
    "save_prediction_vs_simulation_gif",
]


# Style

def apply_default_style() -> None:
    """
    Apply a compact scientific plotting style (same spirit as your snippet).
    Call once per process (idempotent).
    """
    mpl.rcParams.update({
        "figure.figsize": (5.5, 3.2),
        "figure.dpi": 300,
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


# 
# Helpers

def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


#def _time_axis(n: int, *, dt: Optional[float], t0: float = 0.0) -> np.ndarray:
#    """Build a time vector of length n from dt (if provided) or simple epoch index."""
#    if dt is None or dt <= 0:
#        return np.arange(n, dtype=float)
#    return t0 + dt * np.arange(n, dtype=float)


def _overlay_reference_slope(
    ax: plt.Axes,
    k: np.ndarray,
    exponent: float,
    label: str,
    *,
    anchor_k: Optional[float] = None,
    anchor_E: Optional[float] = None,
    kmin: Optional[float] = None,
    kmax: Optional[float] = None,
    linestyle: str = "--",
    linewidth: float = 0.8,
) -> None:
    """
    Plot a reference power-law line E ~ k^exponent on a log-log plot.

    If (anchor_k, anchor_E) are provided, the line is scaled to pass through that point.
    Otherwise it uses a median k in [kmin,kmax] and E=1.0 (pure slope guide).
    """
    mask = np.isfinite(k) & (k > 0)
    if kmin is not None:
        mask &= (k >= kmin)
    if kmax is not None:
        mask &= (k <= kmax)
    kk = k[mask]
    if kk.size < 2:
        return

    if anchor_k is None:
        anchor_k = np.median(kk)
    if anchor_E is None:
        anchor_E = 1.0

    Eref = anchor_E * (kk / anchor_k) ** exponent
    ax.plot(kk, Eref, linestyle=linestyle, linewidth=linewidth, label=label)


# ---------------------------
# Plots
# ---------------------------





def plot_tke_time_series(
    tke_real: Sequence[float],
    tke_pred: Sequence[float],
    *,
    time: np.ndarray,
    title: str = "TKE over time",
    ylabel: str = "TKE",
    xlabel: Optional[str] = None,
    injection_indices: Optional[Iterable[int]] = None,
    out_path: Optional[str | Path] = None,
) -> Path:
    """
    Plot TKE(t) for real vs predicted sequences.

    Parameters
    ----------
    tke_real, tke_pred : sequences of length T
    dt : optional sampling period for x-axis (otherwise uses frame index)
    injection_indices : optional list of frame indices where teacher-forcing injections occurred
    out_path : where to save the PNG (required)

    Returns
    -------
    Path to the saved figure.
    """
    tke_real = np.asarray(tke_real, dtype=float)
    tke_pred = np.asarray(tke_pred, dtype=float)
    assert tke_real.shape == tke_pred.shape, "Simulation and predicted TKE must have same length."

    T = tke_real.shape[0]
    assert T == len(time), "Time, simulation and predicted  TKE must have same length."

    x = time
    if xlabel is None:
        xlabel = "time(s)" 

    fig, ax = plt.subplots()
    ax.plot(x, tke_real, linewidth=0.15, marker='x', markersize=0.3, markevery=1, label="Simulation")
    ax.plot(x, tke_pred, linewidth=0.15, marker='x', markersize=0.3, markevery=1, label="Prediction")

    #ax.plot(x, tke_real, linewidth=0.3, label="Simulation")
    #ax.plot(x, tke_pred, linewidth=0.3, label="Prediction")

    #ax.plot(x, tke_real, linewidth=0.2, label="Simulation")
    #ax.plot(x, tke_pred, linewidth=0.2, label="Prediction")

    if injection_indices:
        inj = [int(i) for i in injection_indices if 0 <= int(i) < T]
        if inj:
            inj = sorted(set(inj))
            inj_times = np.asarray(x[inj], dtype=float)

            trans = ax.get_xaxis_transform()  
            ax.vlines(
                inj_times, 0.0, 0.015,    
                transform=trans,
                colors="red",
                linewidths=0.2,           
                alpha=0.85,
                zorder=5,
                clip_on=False,
            )

            ax.legend_  
            ax.plot(
                [], [], linestyle="None",
                marker="|",                 
                markersize=4,               
                markeredgewidth=0.2,        
                color="red",
                label="injections",
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.5)
    ax.minorticks_on()
    ax.legend(loc="best")

    if out_path is None:
        out_path = Path("tke_timeseries.png")
    out_path = Path(out_path)
    _ensure_parent_dir(out_path)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path

# def plot_tke_time_series(
#     tke_real: Sequence[float],
#     tke_pred: Sequence[float],
#     *,
#     time: np.ndarray,
#     title: str = "TKE over time",
#     ylabel: str = "TKE",
#     xlabel: Optional[str] = None,
#     injection_indices: Optional[Iterable[int]] = None,
#     out_path: Optional[str | Path] = None,
# ) -> Path:
#     """
#     Plot TKE(t) for real vs predicted sequences.

#     Parameters
#     ----------
#     tke_real, tke_pred : sequences of length T
#     dt : optional sampling period for x-axis (otherwise uses frame index)
#     injection_indices : optional list of frame indices where teacher-forcing injections occurred
#     out_path : where to save the PNG (required)

#     Returns
#     -------
#     Path to the saved figure.
#     """
#     tke_real = np.asarray(tke_real, dtype=float)
#     tke_pred = np.asarray(tke_pred, dtype=float)
#     assert tke_real.shape == tke_pred.shape, "Simulation and predicted TKE must have same length."

#     T = tke_real.shape[0]
#     assert T == len(time), "Time, simulation and predicted  TKE must have same length."

#     x = time
#     if xlabel is None:
#         xlabel = "time(s)" 

#     fig, ax = plt.subplots()
#     ax.plot(x, tke_real, linewidth=0.1, marker='x', markersize=0.3, markevery=2, label="Simulation")
#     ax.plot(x, tke_pred, linewidth=0.1, marker='x', markersize=0.3, markevery=2, label="Prediction")

#     # Optional vertical lines for injection frames
#     if injection_indices:
#         inj = sorted(int(i) for i in injection_indices if 0 <= int(i) < T)
#         if inj:
#             ax.vlines(x[inj], ymin=min(np.nanmin(tke_real), np.nanmin(tke_pred)),
#                       ymax=max(np.nanmax(tke_real), np.nanmax(tke_pred)),
#                       colors="gray", linestyles=":", linewidth=0.6, label="injections")

#     #if injection_indices:
#     #    inj = sorted({int(i) for i in injection_indices if 0 <= int(i) < T})
#     #    if inj:
#     #        ymin, ymax = ax.get_ylim()
#     #        tick_h = 0.03 * (ymax - ymin)  
#     #        ax.vlines(x[inj], ymin, ymin + tick_h, colors="red", linewidth=0.8, zorder=5, clip_on=False)
#     #        from matplotlib.lines import Line2D
#     #        inj_handle = Line2D([0], [0], color="red", linewidth=1.2, label="injections")
#     #        handles, labels = ax.get_legend_handles_labels()
#     #        if "injections" not in labels:
#     #            handles.append(inj_handle)
#     #            labels.append("injections")
#     #            ax.legend(handles, labels, loc="best")
            
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     ax.set_title(title)
#     ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.5)
#     ax.minorticks_on()
#     ax.legend(loc="best")

#     if out_path is None:
#         out_path = Path("tke_timeseries.png")
#     out_path = Path(out_path)
#     _ensure_parent_dir(out_path)
#     fig.tight_layout()
#     fig.savefig(out_path, dpi=300)
#     plt.close(fig)
#     return out_path


def plot_energy_spectrum(
    k_real: np.ndarray,
    E_real: np.ndarray,
    k_pred: np.ndarray,
    E_pred: np.ndarray,
    *,
    title: str = "Energy spectrum",
    xlabel: str = r"Wavenumber $k$",
    ylabel: str = r"$E(k)$",
    show_km53: bool = False,
    show_km3: bool = False,
    anchor_on: str = "real",          # "real", "pred", or None for un-anchored guides
    anchor_quantile: float = 0.6,     # choose k* to anchor the slope near upper-middle k
    out_path: Optional[str | Path] = None,
) -> Path:
    """
    Plot time-averaged isotropic energy spectra for real vs predicted.

    Parameters
    ----------
    k_real, E_real : arrays for the real spectrum
    k_pred, E_pred : arrays for the predicted spectrum
    show_km53 : overlay a k^{-5/3} guide line
    show_km3  : overlay a k^{-3} guide line
    anchor_on : which curve to anchor the guide lines on ("real" | "pred" | None)
    anchor_quantile : choose anchor k* by this quantile of valid k (e.g., 0.6)
    out_path : where to save the PNG (required)

    Returns
    -------
    Path to the saved figure.
    """
    k_real = np.asarray(k_real, dtype=float)
    E_real = np.asarray(E_real, dtype=float)
    k_pred = np.asarray(k_pred, dtype=float)
    E_pred = np.asarray(E_pred, dtype=float)

    fig, ax = plt.subplots()
    ax.loglog(k_real, E_real, linewidth=0.4, label="Simulation")
    ax.loglog(k_pred, E_pred, linewidth=0.4, label="Prediction")

    # Optional reference slopes
    if show_km53 or show_km3:
        if anchor_on == "real":
            k_ref, E_ref = k_real, E_real
        elif anchor_on == "pred":
            k_ref, E_ref = k_pred, E_pred
        else:
            k_ref, E_ref = None, None

        anchor_k = None
        anchor_E = None
        if k_ref is not None:
            mask = np.isfinite(k_ref) & np.isfinite(E_ref) & (k_ref > 0) & (E_ref > 0)
            if np.any(mask):
                # pick a moderately high-k anchor (to see the slope in the inertial-like range)
                kvals = np.sort(k_ref[mask])
                anchor_k = kvals[int(np.clip(anchor_quantile * (kvals.size - 1), 0, kvals.size - 1))]
                # find nearest E for that k
                idx = np.argmin(np.abs(k_ref - anchor_k))
                anchor_E = E_ref[idx]

        #if show_km53:
        #    _overlay_reference_slope(ax, k_real, exponent=-5.0/3.0, label=r"$k^{-5/3}$",
        #                             anchor_k=anchor_k, anchor_E=anchor_E)
        if show_km3:
            _overlay_reference_slope(ax, k_real, exponent=-3.0, label=r"$k^{-3}$",
                                     anchor_k=anchor_k, anchor_E=anchor_E)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.5)
    ax.minorticks_on()
    ax.legend(loc="best")

    if out_path is None:
        out_path = Path("energy_spectrum.png")
    out_path = Path(out_path)
    _ensure_parent_dir(out_path)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path

def save_prediction_vs_simulation_gif(
    pred: np.ndarray | torch.Tensor,
    real: np.ndarray | torch.Tensor,
    figures_dir: Path,
    *,
    filename: str = "pred_vs_real.gif",
    fps: int = 6,
    dpi: int = 300,
    cmap: str = "viridis",
    robust: bool = True,
    equal_clim: bool = True,
    suptitle: Optional[str] = None,
) -> Path:
    """
    Sauvegarde un GIF côte-à-côte (Simulation à gauche, Prediction à droite) avec colorbar partagée.
    - Style calé sur xarray.plot.imshow(..., cmap=sns.cm.icefire, robust=True).
    - Même valeur => même couleur sur les deux animations.
    Retour : Path(figures_dir/filename)
    """
    import numpy as _np
    import matplotlib.pyplot as _plt
    import seaborn as _sns
    import math as _math
    from matplotlib.colors import Normalize as _Normalize
    import matplotlib.animation as _animation

    # Lazy import torch si dispo
    try:
        import torch as _torch  # type: ignore
    except Exception:
        _torch = None  # type: ignore

    def _to_numpy(arr):
        if _torch is not None and isinstance(arr, _torch.Tensor):
            return arr.detach().cpu().numpy()
        return _np.asarray(arr)

    # Limiter à 32 frames pour rester cohérent avec tes usages
    pred_np = _to_numpy(pred[:32])
    real_np = _to_numpy(real[:32])

    # Supprimer dimensions singleton si (T,1,H,W) ou (1,T,H,W)
    pred_np = _np.squeeze(pred_np)
    real_np = _np.squeeze(real_np)

    # Forcer (T,H,W). Si (B,T,H,W) ou (T,C,H,W), on prend le premier batch/canal.
    def _ensure_THW(x):
        if x.ndim == 3:
            return x
        if x.ndim == 4:
            # heuristique : (B,T,H,W) si x.shape[0] < 8 et x.shape[1] >= 8
            if x.shape[0] < 8 and x.shape[1] >= 8:
                return x[0]
            # sinon (T,C,H,W) → on prend le canal 0
            return x[:, 0]
        raise ValueError(f"Expected (T,H,W) or (T,1,H,W)/(B,T,H,W)/(T,C,H,W), got {x.shape}")

    pred_np = _ensure_THW(pred_np)
    real_np = _ensure_THW(real_np)

    T = min(pred_np.shape[0], real_np.shape[0])
    H, W = real_np.shape[1], real_np.shape[2]

    # === Échelle commune (style xarray robust=True) ===
    # On calcule des percentiles globaux sur les deux séries concaténées.
    if robust:
        both_flat = _np.concatenate([pred_np[:T].ravel(), real_np[:T].ravel()])
        vmin_glob = _np.nanpercentile(both_flat, 2.0)
        vmax_glob = _np.nanpercentile(both_flat, 98.0)
    else:
        vmin_glob = _np.nanmin([_np.nanmin(pred_np[:T]), _np.nanmin(real_np[:T])])
        vmax_glob = _np.nanmax([_np.nanmax(pred_np[:T]), _np.nanmax(real_np[:T])])

    # IMPORTANT : pas de symétrisation autour de 0 pour rester fidèle à xarray+robust.
    # (equal_clim est interprété ici comme "mêmes limites entre panneaux" — déjà respecté.)

    # Palette strictement identique au snippet
    cmap_used = _sns.cm.icefire

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / filename

    # === Figure : 2 colonnes (Simulation à gauche, Prediction à droite), colorbar partagée ===
    _plt.ioff()
    # Taille pensée pour lisibilité + petites polices
    fig = _plt.figure(figsize=(7.6, 3.6), dpi=dpi)
    gs = fig.add_gridspec(1, 2, wspace=0.08)
    ax_sim = fig.add_subplot(gs[0, 0])  # gauche
    ax_pred = fig.add_subplot(gs[0, 1])  # droite

    # Coordonnées spatiales comme dans le snippet: x = y = arange(N) * 2π / N
    L = 2 * _np.pi
    extent = (0.0, L, 0.0, L)

    norm = _Normalize(vmin=vmin_glob, vmax=vmax_glob)

    # Simulation (gauche) puis Prediction (droite)
    im_sim = ax_sim.imshow(
        real_np[0], origin="upper", extent=extent, cmap=cmap_used, norm=norm, interpolation="nearest", aspect="equal"
    )
    im_pred = ax_pred.imshow(
        pred_np[0], origin="upper", extent=extent, cmap=cmap_used, norm=norm, interpolation="nearest", aspect="equal"
    )

    # Titres sobres
    ax_sim.set_title("Simulation", fontsize=9, pad=4)
    ax_pred.set_title("Prediction", fontsize=9, pad=4)
    for ax in (ax_sim, ax_pred):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=10, y=0.98)

    # Colorbar unique partagée
    cbar = fig.colorbar(im_sim, ax=[ax_sim, ax_pred], pad=0.02, fraction=0.04, aspect=25)
    cbar.ax.tick_params(labelsize=8)
    # Pas de label pour coller au style minimal d'xarray.plot.imshow
    # cbar.set_label("Vorticité", fontsize=9)  # (désactivé pour rester minimal)

    # Texte de pied discret (frame counter)
    txt = fig.text(0.5, 0.02, "", ha="center", va="bottom", fontsize=8)

    # === Animation ===
    interval_ms = int(1000 / max(fps, 1))

    def _update(i):
        im_sim.set_data(real_np[i])
        im_pred.set_data(pred_np[i])
        txt.set_text(f"frame {i+1}/{T}")
        return im_sim, im_pred, txt

    anim = _animation.FuncAnimation(
        fig, _update, frames=T, interval=interval_ms, blit=False
    )

    # PillowWriter avec le fps demandé
    writer = _animation.PillowWriter(fps=max(fps, 1))
    anim.save(out_path.as_posix(), writer=writer, dpi=dpi)

    _plt.close(fig)
    return out_path

# def save_prediction_vs_simulation_gif(
#     pred: np.ndarray | torch.Tensor,
#     real: np.ndarray | torch.Tensor,
#     figures_dir: Path,
#     *,
#     filename: str = "pred_vs_real.gif",
#     fps: int = 1,
#     dpi: int = 300,
#     cmap: str = "viridis",
#     robust: bool = True,
#     equal_clim: bool = True,
#     suptitle: Optional[str] = None,
# ) -> Path:
#     """
#     Sauvegarde un GIF côte-à-côte (Prediction vs Simulation) de qualité.
#     pred, real : arrays (T, H, W) ou Tensors torch. Sortie : figures_dir/filename.
#     """
#     import numpy as _np
#     import matplotlib.pyplot as _plt

#     # Lazy import torch si dispo
#     try:
#         import torch as _torch  # type: ignore
#     except Exception:
#         _torch = None  # type: ignore

#     def _to_numpy(arr):
#         if _torch is not None and isinstance(arr, _torch.Tensor):
#             return arr.detach().cpu().numpy()
#         return _np.asarray(arr)

#     pred_np = _to_numpy(pred[:32])
#     real_np = _to_numpy(real[:32])

#     # Supprimer dimensions singleton si (T,1,H,W) ou (1,T,H,W)
#     pred_np = _np.squeeze(pred_np)
#     real_np = _np.squeeze(real_np)

#     # Forcer (T,H,W). Si (B,T,H,W) ou (T,C,H,W), on prend le premier batch/canal.
#     def _ensure_THW(x):
#         if x.ndim == 3:
#             return x
#         if x.ndim == 4:
#             # heuristique : (B,T,H,W) si x.shape[0] < 8 et x.shape[1] >= 8
#             if x.shape[0] < 8 and x.shape[1] >= 8:
#                 return x[0]
#             # sinon (T,C,H,W) → on prend le canal 0
#             return x[:, 0]
#         raise ValueError(f"Expected (T,H,W) or (T,1,H,W)/(B,T,H,W)/(T,C,H,W), got {x.shape}")

#     pred_np = _ensure_THW(pred_np)
#     real_np = _ensure_THW(real_np)

#     T = min(pred_np.shape[0], real_np.shape[0])

#     # Limites couleurs robustes (percentiles) pour une comparaison propre
#     if robust:
#         both = _np.concatenate(
#             [pred_np[:T].reshape(T, -1), real_np[:T].reshape(T, -1)],
#             axis=1,
#         )
#         vmin_glob = _np.percentile(both, 2.0)
#         vmax_glob = _np.percentile(both, 98.0)
#     else:
#         vmin_glob = min(pred_np.min(), real_np.min())
#         vmax_glob = max(pred_np.max(), real_np.max())

#     # Limites symétriques
#     if equal_clim:
#         a = max(abs(vmin_glob), abs(vmax_glob))
#         vmin_glob, vmax_glob = -a, a

#     figures_dir = Path(figures_dir)
#     figures_dir.mkdir(parents=True, exist_ok=True)
#     out_path = figures_dir / filename

#     # Figure 2 colonnes, avec colorbars
#     _plt.ioff()
#     fig = _plt.figure(figsize=(10, 4.5), dpi=dpi)
#     gs = fig.add_gridspec(1, 2, wspace=0.15)
#     ax1 = fig.add_subplot(gs[0, 0])
#     ax2 = fig.add_subplot(gs[0, 1])

#     im1 = ax1.imshow(
#         pred_np[0], origin="lower", cmap=cmap,
#         vmin=vmin_glob, vmax=vmax_glob, interpolation="nearest"
#     )
#     im2 = ax2.imshow(
#         real_np[0], origin="lower", cmap=cmap,
#         vmin=vmin_glob, vmax=vmax_glob, interpolation="nearest"
#     )

#     ax1.set_title("Prediction", fontsize=11)
#     ax2.set_title("Simulation", fontsize=11)
#     for ax in (ax1, ax2):
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.set_aspect("equal")

#     if suptitle:
#         fig.suptitle(suptitle, fontsize=12, y=0.98)

#     cbar_kw = dict(shrink=0.85, pad=0.02)
#     #fig.colorbar(im1, ax=ax1, **cbar_kw)
#     #fig.colorbar(im2, ax=ax2, **cbar_kw)
#     cbar = fig.colorbar(im2, ax=[ax1, ax2], **cbar_kw)
    
#     txt = fig.text(0.5, 0.02, "", ha="center", va="bottom", fontsize=10)
#     plt.tight_layout()
#     def _update(i):
#         im1.set_data(pred_np[i])
#         im2.set_data(real_np[i])
#         txt.set_text(f"frame {i+1}/{T}")
#         return im1, im2, txt

#     anim = animation.FuncAnimation(
#         fig, _update, frames=T, interval=1000.0 / max(fps, 1), blit=False
#     )

#     # PillowWriter = portable, qualité propre (dpi contrôlé)
#     writer = animation.PillowWriter(fps=fps)
#     anim.save(out_path.as_posix(), writer=writer, dpi=dpi)

#     _plt.close(fig)
#     return out_path
