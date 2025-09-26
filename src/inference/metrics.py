from __future__ import annotations

from typing import Tuple, Optional
import numpy as np
import torch
from training.losses import Vorticity2Velocity

__all__ = [
    "vorticity_to_velocity_sequence",
    "tke_from_vorticity_sequence",
    "energy_spectrum_from_vorticity_sequence",
]



def _dx_dy_from_domain(
    x_min: float, x_max: float, y_min: float, y_max: float, H: int, W: int
) -> Tuple[float, float]:
    """
    Compute spacings from domain and grid shape.
    Array shape convention: (H, W) = (Ny, Nx) with x along axis=1, y along axis=0.
    """
    Lx = float(x_max - x_min)
    Ly = float(y_max - y_min)
    dx = Lx / float(W)
    dy = Ly / float(H)
    return dx, dy




def vorticity_to_velocity_sequence(
    omega_seq: np.ndarray | torch.Tensor,
    *,
    domain: Tuple[float, float, float, float],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map vorticity sequence to (u, v) using your Vorticity2Velocity.
    Accepts (T,H,W) or (H,W). Returns numpy arrays matching input shape.
    """
    # Ensure np.ndarray (T,H,W)
    if isinstance(omega_seq, torch.Tensor):
        omega_np = omega_seq.detach().cpu().numpy()
    else:
        omega_np = np.asarray(omega_seq)

    if omega_np.ndim == 2:
        omega_np = omega_np[None, ...]  # (1,H,W)

    device = torch.device(device)
    omega_t = torch.from_numpy(omega_np).to(device=device, dtype=torch.float32)

    v2v = Vorticity2Velocity(device=device, domain=domain)
    u_t, v_t = v2v(omega_t)  # shape (T,H,W)

    u = u_t.detach().cpu().numpy().astype(np.float32)
    v = v_t.detach().cpu().numpy().astype(np.float32)

    if u.shape[0] == 1:
        return u[0], v[0]
    return u, v


# ---------------------------
# TKE with temporal mean removed
# ---------------------------

def tke_from_vorticity_sequence(
    omega_seq: np.ndarray | torch.Tensor,
    *,
    domain: Tuple[float, float, float, float],
    device: torch.device,
    area_weighted_total: bool = False,
) -> Tuple[np.ndarray, float]:
    """
    Turbulent Kinetic Energy from a vorticity sequence:
    Returns
    -------
    tke_t : (T,) array of TKE per frame 
    tke_mean : scalar average over time
    """
    u, v = vorticity_to_velocity_sequence(omega_seq, domain=domain, device=device)

    T, H, W = u.shape
    # Temporal means (per pixel)
    #Ubar = u.mean(axis=0, keepdims=True)
    #Vbar = v.mean(axis=0, keepdims=True)

    Ubar = u.mean(axis=(-2, -1), keepdims=True)
    Vbar = v.mean(axis=(-2, -1), keepdims=True)
    up = u - Ubar
    vp = v - Vbar

    tke_t = (0.5 * (up**2 + vp**2)).mean(axis=(-2, -1))  # domain average

    #if area_weighted_total:
    #    x_min, x_max, y_min, y_max = domain
    #    dx, dy = _dx_dy_from_domain(x_min, x_max, y_min, y_max, H, W)
    #    area = dx * dy * H * W
    #    tke_t = tke_t * area  # integral KE over the box

    return tke_t.astype(np.float32), float(tke_t.mean())


# Energy spectrum (time-averaged, from fluctuations)

def _energy_spectrum_snapshot(
    u: np.ndarray,
    v: np.ndarray,
    dx: float,
    dy: float,
    *,
    nbins: int = 64,
    density: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Isotropic KE spectrum from one snapshot (u, v), NumPy FFT conventions.
    Uses Parseval: per-mode energy divided by N^2 (N = H*W), then radial binning.
    density=True -> E(k) density so that sum(E*dk) ≈ KE.
    """
    H, W = u.shape
    N = float(H * W)

    # FFT and per-mode energy (Parseval with numpy.fft)
    U = np.fft.fft2(u)
    V = np.fft.fft2(v)
    e_mode = 0.5 * (np.abs(U)**2 + np.abs(V)**2) / (N**2)

    # Wavenumbers: x along axis=1 (W, dx), y along axis=0 (H, dy)
    kx = 2.0 * np.pi * np.fft.fftfreq(W, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(H, d=dy)
    kxg, kyg = np.meshgrid(kx, ky, indexing="xy")
    kmag = np.sqrt(kxg**2 + kyg**2)

    # Radial bins
    nbins = max(8, int(nbins))
    k_max = float(kmag.max())
    k_edges = np.linspace(0.0, k_max + 1e-12, nbins + 1)
    k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])
    dk = k_edges[1] - k_edges[0]

    # Shell-sum
    k_flat = kmag.ravel()
    e_flat = e_mode.ravel()
    idx = np.digitize(k_flat, k_edges)  # 1..nbins

    E_shell = np.zeros(nbins, dtype=np.float64)
    for b in range(1, nbins + 1):
        m = (idx == b)
        if np.any(m):
            E_shell[b - 1] = e_flat[m].sum()

    E_k = E_shell / max(dk, 1e-12) if density else E_shell
    return k_centers.astype(np.float32), E_k.astype(np.float32)


def energy_spectrum_from_vorticity_sequence(
    omega_seq: np.ndarray | torch.Tensor,
    *,
    domain: Tuple[float, float, float, float],
    device: torch.device,
    nbins: int = 64,
    subtract_time_mean: bool = True,
    density: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Time-averaged isotropic KE spectrum from a vorticity sequence:
      - reconstruct (u, v)
      - remove time mean per pixel (u', v')
      - compute spectrum for each frame and average over time.

    Returns (k_centers, E_k_time_avg).
    """
    u, v = vorticity_to_velocity_sequence(omega_seq, domain=domain, device=device)

    # Ensure time dimension
    single = False
    if u.ndim == 2:
        u = u[None, ...]
        v = v[None, ...]
        single = True

    T, H, W = u.shape
    x_min, x_max, y_min, y_max = domain
    dx, dy = _dx_dy_from_domain(x_min, x_max, y_min, y_max, H, W)

    if subtract_time_mean:
        Ubar = u.mean(axis=0, keepdims=True)
        Vbar = v.mean(axis=0, keepdims=True)
        u = u - Ubar
        v = v - Vbar

    Ek_list = []
    k_ref = None
    for t in range(T):
        k, Ek = _energy_spectrum_snapshot(u[t], v[t], dx, dy, nbins=nbins, density=density)
        if k_ref is None:
            k_ref = k
        Ek_list.append(Ek)

    E_k_timeavg = np.mean(np.stack(Ek_list, axis=0), axis=0) if not single else Ek_list[0]
    return k_ref, E_k_timeavg
