# src/turbounet/losses/turbulence.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Tuple, Dict

import torch
import torch.nn as nn


class Vorticity2Velocity:
    """
    Map 2D vorticity (..., H, W) to velocity components (u, v).
    """

    def __init__(self, device: torch.device | str, domain: Tuple[float, float, float, float]) -> None:
        self.device = torch.device(device)
        self.domain = tuple(float(v) for v in domain)  # (x_min, x_max, y_min, y_max)
        self._cache: Dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor, complex]] = {}

    def _get_freqs(self, H: int, W: int, device: torch.device):
        """Build (and cache) frequency grids and Laplacian for given H, W, device, domain."""
        key = (H, W, device, self.domain)
        if key in self._cache:
            return self._cache[key]

        x_min, x_max, y_min, y_max = self.domain
        # Spatial steps from domain and grid size
        dx = (x_max - x_min) / H
        dy = (y_max - y_min) / W

        # Spatial frequencies (FFTFREQ for H, RFFTFREQ for W)
        kx_1d = torch.fft.fftfreq(H, d=dx, device=device)
        ky_1d = torch.fft.rfftfreq(W, d=dy, device=device)

        # Frequency mesh (H, W//2+1)
        kx, ky = torch.meshgrid(kx_1d, ky_1d, indexing="ij")

        two_pi_i = 2 * math.pi * 1j

        # Laplacian in Fourier space: (2πi)^2 * (|kx|^2 + |ky|^2)
        laplace = (two_pi_i ** 2) * (kx.abs() ** 2 + ky.abs() ** 2)
        laplace[0, 0] = 1  # avoid division by zero (psi_hat(0,0) = 0 anyway)

        self._cache[key] = (kx, ky, laplace, two_pi_i)
        return self._cache[key]

    def __call__(self, vort: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute (u, v) from vorticity with automatic (H, W, device) detection."""
        # Detect grid and device from input
        H, W = vort.shape[-2], vort.shape[-1]
        device = vort.device

        kx, ky, laplace, two_pi_i = self._get_freqs(H, W, device)

        # Real FFT on spatial dims
        vorticity_hat = torch.fft.rfftn(vort, dim=(-2, -1))

        # Streamfunction in Fourier space
        psi_hat = -vorticity_hat / laplace

        # Velocity in Fourier space
        vx_hat = two_pi_i * ky * psi_hat
        vy_hat = -two_pi_i * kx * psi_hat

        # Inverse real FFT back to physical space
        vx = torch.fft.irfftn(vx_hat, s=(H, W), dim=(-2, -1))
        vy = torch.fft.irfftn(vy_hat, s=(H, W), dim=(-2, -1))
        return vx, vy


def TKE(u: torch.Tensor,
        v: torch.Tensor,
        space_axes = (-2, -1),
        mean_axes = None) -> torch.Tensor:
    """
    Turbulent kinetic energy (spatial average):

      TKE = 0.5 * < (u - <u>)^2 + (v - <v>)^2 >

    where <.> is the mean over spatial axes.
    """
    if mean_axes is None:
        mean_axes = space_axes

    u_bar = u.mean(dim=mean_axes, keepdim=True)
    v_bar = v.mean(dim=mean_axes, keepdim=True)

    u_prime = u - u_bar
    v_prime = v - v_bar

    tke_density = 0.5 * (u_prime**2 + v_prime**2)
    tke = tke_density.mean(dim=space_axes)
    return tke


class TKEMSELoss(nn.Module):
    """
    Loss = MSE(TKE(pred), TKE(targ)) + MSE(pred, targ)
    """
    def __init__(self, device: torch.device | str, domain: Tuple[float, float, float, float]) -> None:
        super().__init__()
        self.v2v = Vorticity2Velocity(device=device, domain=domain)
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        up, vp = self.v2v(pred)
        ut, vt = self.v2v(targ)
        ke_p, ke_t = TKE(up, vp), TKE(ut, vt)
        return self.mse(ke_p, ke_t) + self.mse(pred, targ)


class TKELoss(nn.Module):
    """
    Loss = MSE(TKE(pred), TKE(targ))
    """
    def __init__(self, device: torch.device | str, domain: Tuple[float, float, float, float]) -> None:
        super().__init__()
        self.v2v = Vorticity2Velocity(device=device, domain=domain)
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        up, vp = self.v2v(pred)
        ut, vt = self.v2v(targ)
        ke_p, ke_t = TKE(up, vp), TKE(ut, vt)
        return self.mse(ke_p, ke_t)



LOSS_REGISTRY = {
    "mse":    lambda device, domain: nn.MSELoss(),
    "tke":    lambda device, domain: TKELoss(device=device, domain=domain),
    "tkemse": lambda device, domain: TKEMSELoss(device=device, domain=domain),
}

def get_loss(
    name: str | None,
    *,
    device: torch.device,
    domain: tuple[float, float, float, float] | None = None,
) -> nn.Module:
    """
    Return a loss module by name.

    Args:
        name: "mse" (default), "tke", or "tkemse".
        device: torch device for internal tensors.
        domain: (x_min, x_max, y_min, y_max). Required for "tke"/"tkemse".

    Notes:
        - If name is None or "", defaults to MSELoss().
        - For "tke"/"tkemse", 'domain' must be provided.
    """
    key = (name or "mse").strip().lower()
    if key in ("mse", "l2"):
        return LOSS_REGISTRY["mse"](device, domain)

    if key in ("tke", "tkemse") and domain is None:
        raise ValueError(
            f"'domain' is required for '{key}' loss. "
            "Provide (x_min, x_max, y_min, y_max) from your dataset attrs."
        )

    try:
        return LOSS_REGISTRY[key](device, domain)
    except KeyError:
        raise ValueError(f"Unknown loss '{name}'. Available: mse | tke | tkemse.")
