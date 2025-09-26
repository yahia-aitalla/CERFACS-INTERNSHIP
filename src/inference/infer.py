from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import seaborn as sns

import torch
import yaml
import numpy as np

from inference.core import infer_one_experiment
from inference.metrics import (
    tke_from_vorticity_sequence,
    energy_spectrum_from_vorticity_sequence,
)
from inference.plotting import (
    apply_default_style,
    plot_tke_time_series,
    plot_energy_spectrum,
    save_prediction_vs_simulation_gif
)

from models.unet import UNet



PROJECT_ROOT = Path("/scratch/algo/aitalla/StageGitlab")
DEFAULT_CFG  = PROJECT_ROOT / "configs" / "infer" / "infer.yaml"
RUNS_ROOT    = PROJECT_ROOT / "predictions" 


# Helpers

def _read_yaml(path: str | os.PathLike) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _sanitize_name(s: str) -> str:
    s = s.strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9._\-]", "-", s)


def _prepare_run_dir(expe_name: Optional[str], cfg_path: Path, mode: str, ckpt_name: str) -> Path:
    """
    Make a unique run directory under RUNS_ROOT.
    Layout:
      run_dir/
        config.yaml
        preds/
        figures/
        logs/
    """
    ts = time.strftime("%Y%m%d-%H%M%S")
    if expe_name and expe_name.strip():
        name = _sanitize_name(expe_name)
    else:
        base_ckpt = _sanitize_name(Path(ckpt_name).stem or "ckpt")
        name = f"infer_{ts}_{mode}_{base_ckpt}"
    run_dir = RUNS_ROOT / name
    (run_dir / "preds").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    # Copy config for provenance
    dst_cfg = run_dir / "config.yaml"
    if Path(cfg_path).resolve() != dst_cfg.resolve():
        try:
            dst_cfg.write_text(Path(cfg_path).read_text())
        except Exception:
            dst_cfg.write_text(f"# config copied from: {cfg_path}\n")
    return run_dir


def _build_model_from_cfg(model_cfg: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    """
    Build your UNet with defaults matching training.
    """
    in_ch  = int(model_cfg.get("in_channels", 1))
    out_ch = int(model_cfg.get("num_classes", 1))
    padding_mode = str(model_cfg.get("padding_mode", "zeros"))
    padding = int(model_cfg.get("padding", 1))
    model = UNet(
        in_channels=in_ch,
        num_classes=out_ch,
        padding_mode=padding_mode,
        padding=padding,
    ).to(device)
    return model


def _load_checkpoint_into_model(model: torch.nn.Module, ckpt_path: str | os.PathLike, device: torch.device) -> None:
    """
    Load either a pure state_dict or a dict with a 'model' key.
    """
    state = torch.load(os.fspath(ckpt_path), map_location=torch.device(device), weights_only=True)
    model.load_state_dict(state)



# Main
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Inference runner (free or chunked) with YAML config. "
                    "Saves predictions (HDF5) + plots (PNG) into a unique run dir."
    )
    ap.add_argument("--config", type=str, default=None,
                    help=f"Path to infer YAML. Default: {DEFAULT_CFG}")
    ap.add_argument("--expe_name", type=str, default=None,
                    help="Custom name for the run directory under runs/infer/.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    cfg_path = Path(args.config or DEFAULT_CFG).expanduser().resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config YAML not found: {cfg_path}")
    cfg = _read_yaml(cfg_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = cfg.get("model", {}) or {}
    model = _build_model_from_cfg(model_cfg, device=device)

    ckpt_path = cfg.get("checkpoint", None)
    if not ckpt_path:
        raise ValueError("Config must provide 'checkpoint' path to a trained model.")
    ckpt_path = Path(ckpt_path).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    _load_checkpoint_into_model(model, ckpt_path,device)

    rollout = cfg.get("rollout", {}) or {}
    mode = str(rollout.get("mode", "free")).lower()  # "free" | "chunked"
    seed_index = int(rollout.get("seed_index", 0))
    block_n = int(rollout.get("block_n", 8)) if mode == "chunked" else None

    data_cfg = cfg.get("data", {}) or {}
    exp_dir = data_cfg.get("experiment_dir") 
    
    h5_name = str(data_cfg.get("h5_name", "vorticity.h5"))
    key     = str(data_cfg.get("key", "vorticity"))

    metrics_cfg = cfg.get("metrics", {}) or {}
    spectrum_nbins    = int(metrics_cfg.get("spectrum_nbins", 64))
    spectrum_density  = bool(metrics_cfg.get("spectrum_density", True))
    tke_total         = bool(metrics_cfg.get("tke_area_weighted_total", False))

    plots_cfg = cfg.get("plots", {}) or {}
    show_km53 = bool(plots_cfg.get("show_km53", False))
    show_km3  = bool(plots_cfg.get("show_km3", False))

    run_dir = _prepare_run_dir(args.expe_name, cfg_path, mode=mode, ckpt_name=str(ckpt_path.name))
    preds_dir   = run_dir / "preds"
    figures_dir = run_dir / "figures"

    apply_default_style()

    exp_dir = Path(exp_dir).expanduser().resolve()
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    out_h5 = preds_dir / "vorticity.h5"

    result = infer_one_experiment(
        model, device,
        experiment_dir=exp_dir,
        h5_name=h5_name,
        key=key,
        mode=mode,
        block_n=block_n,
        seed_index=seed_index,
        out_path=out_h5,
        source_checkpoint=str(ckpt_path),
    )

    # result: pred (T,H,W), real (T,H,W), real_attrs, injection_indices, pred_path
    pred = result["pred"]
    real = result["real"]
    attrs = result["real_attrs"]
    inj_idx = result.get("injection_indices", [])
    print(inj_idx)

    # metrics
    domain = (attrs["x_min"], attrs["x_max"], attrs["y_min"], attrs["y_max"])
    time = attrs["time"]
    # TKE time-series
    tke_real_t, _ = tke_from_vorticity_sequence(real, domain=domain, device=device, area_weighted_total=tke_total)
    tke_pred_t, _ = tke_from_vorticity_sequence(pred, domain=domain, device=device, area_weighted_total=tke_total)

    # Energy spectra (time-averaged)
    k_real, Ek_real = energy_spectrum_from_vorticity_sequence(
        real, domain=domain, device=device, nbins=spectrum_nbins, subtract_time_mean=True, density=spectrum_density
    )
    k_pred, Ek_pred = energy_spectrum_from_vorticity_sequence(
        pred, domain=domain, device=device, nbins=spectrum_nbins, subtract_time_mean=True, density=spectrum_density
    )

    # plotting 
    # TKE(t)
    tke_png = figures_dir / f"tke_timeseries.png"
    plot_tke_time_series(
        tke_real=tke_real_t,
        tke_pred=tke_pred_t,
        time=time,
        injection_indices=inj_idx,
        out_path=tke_png,
    )

    # Spectrum
    sp_png = figures_dir / f"energy_spectrum.png"
    plot_energy_spectrum(
        k_real=k_real, E_real=Ek_real,
        k_pred=k_pred, E_pred=Ek_pred,
        show_km53=show_km53, show_km3=show_km3,
        anchor_on="real",
        out_path=sp_png,
    )

    # Animation GIF côte-à-côte (Prediction vs Simulation)
    gif_path = save_prediction_vs_simulation_gif(
        pred=pred,
        real=real,
        figures_dir=figures_dir,
        filename="prediction_vs_simulation.gif",
        fps=1,
        dpi=150,
        cmap=sns.cm.icefire,   
        robust=True,           
        equal_clim=False,      
        suptitle="Vorticity snapshots (pred vs real)",
    )


    print(f"[OK] Prediction: {result['pred_path']}")
    print(f"[OK] Figures   : {tke_png}, {sp_png}, {gif_path} ")
    #print(f"[OK] Figures   : {tke_png}")

    print(f"\n[Done] Outputs in: {run_dir}")


if __name__ == "__main__":
    main()
