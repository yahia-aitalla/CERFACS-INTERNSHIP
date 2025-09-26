from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

import h5py
import numpy as np
import torch


def read_real_h5(h5_path: str | Path, key: str = "vorticity") -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load the 'real' solver trajectory from HDF5 and return (array, attrs).
    Returns (T,H,W) array and a dict of attributes (mean, std, dt, domain, kind, ...).
    """
    h5_path = Path(h5_path).expanduser().resolve()
    if not h5_path.is_file():
        raise FileNotFoundError(f"HDF5 not found: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        if key not in f:
            raise KeyError(f"Dataset '{key}' not in {h5_path}. Available: {list(f.keys())}")
        d = f[key]
        arr = d[()]  # (T,H,W)
        a = d.attrs

        required = [
            "mean", "var", "std", "min", "max",
            "dt", "inner_steps", "outer_steps", "final_time", "time",
            "viscosity", "x_min", "x_max", "y_min", "y_max",
            "kind", "solver_iteration_time",
        ]
        missing = [k for k in required if k not in a]
        if missing:
            raise KeyError(f"Missing required attrs in {h5_path}/{key}: {missing}")
        

        attrs = {
            "mean": float(a.get("mean")),
            "std": float(a.get("std")),
            "var": float(a.get("var")),
            "min": float(a.get("min")),
            "max": float(a.get("max")),
            "dt": float(a.get("dt")),
            "viscosity": float(a.get("viscosity")),
            "x_min": float(a.get("x_min")),
            "x_max": float(a.get("x_max")),
            "y_min": float(a.get("y_min")),
            "y_max": float(a.get("y_max")),
            "inner_steps": int(a.get("inner_steps")),
            "outer_steps": int(a.get("outer_steps")),
            "final_time": float(a.get("final_time")),
            "time": np.asarray(a.get("time"), dtype=np.float64),

            "kind": str(a.get("kind")),
            "solver_iteration_time": float(a.get("solver_iteration_time", 0.0)),
        }
    if arr.ndim != 3:
        raise ValueError(f"Expected (T,H,W), got {arr.shape}")
    return arr, attrs


@torch.no_grad()
def rollout_free_run(
    model: torch.nn.Module,
    device: torch.device,
    real: np.ndarray,
    *,
    mean: float,
    std: float,
    seed_index: int = 0,
) -> np.ndarray:
    """
    Fully autoregressive rollout: start from real[t0], then only use model outputs.
    Returns a full-length prediction with shape (T,H,W) in physical units.
    pred[0] = real[seed_index].
    """
    T, H, W = real.shape
    assert 0 <= seed_index < T
    eps = 1e-12
    s = float(std if abs(std) > eps else 1.0)

    pred = np.empty_like(real, dtype=np.float32)
    pred[seed_index] = real[seed_index].astype(np.float32)

    # normalized seed (1,1,H,W)
    x = torch.from_numpy(((real[seed_index].astype(np.float32) - mean) / s)[None, None, ...]).to(device)
    model.eval()

    for t in range(seed_index + 1, T):
        y = model(x)  
        y_denorm = (y.squeeze(0).squeeze(0).detach().cpu().float().numpy() * s + mean)
        pred[t] = y_denorm.astype(np.float32)
        x = y  


    return pred


@torch.no_grad()
def rollout_chunked_teacher_forcing(
    model: torch.nn.Module,
    device: torch.device,
    real: np.ndarray,
    *,
    mean: float,
    std: float,
    seed_index: int = 0,
    block_n: int = 8,
    copy_injected_into_pred: bool = True,
) -> Tuple[np.ndarray, List[int]]:
    """
    Chunked teacher-forcing rollout:
      - seed with real[t0]
      - generate 'block_n' predictions autoregressively
      - inject real at t = t0 + (block_n+1), then repeat...
    At injection times, we set x = normalize(real[t_inj]) before predicting the next steps.
    Optionally, we also copy the injected real frame into pred[t_inj] for continuity.

    Returns
    -------
    pred : np.ndarray
        Full-length prediction (T,H,W) in physical units, with pred[0] = real[seed_index].
        At injection indices, pred[t_inj] = real[t_inj] if copy_injected_into_pred=True.
    injection_indices : list[int]
        Sorted list of injection time indices (excluding seed_index).
    """
    T, H, W = real.shape
    assert 0 <= seed_index < T
    block_n = int(max(1, block_n))
    eps = 1e-12
    s = float(std if abs(std) > eps else 1.0)

    pred = np.empty_like(real, dtype=np.float32)
    pred[seed_index] = real[seed_index].astype(np.float32)

    # Build injection indices: t = seed_index + (block_n + 1), 2*(block_n+1), ...
    step = block_n + 1
    injection_indices: List[int] = list(range(seed_index + step, T, step))

    # normalized seed
    x = torch.from_numpy(((real[seed_index].astype(np.float32) - mean) / s)[None, None, ...]).to(device)
    model.eval()

    for t in range(seed_index + 1, T):
        if t in injection_indices:
            # Inject real frame: reset state with normalized ground truth at t
            if copy_injected_into_pred:
                pred[t] = real[t].astype(np.float32)  # zero error at injection for clarity
            # Prepare x for the NEXT step using the injected real frame
            x = torch.from_numpy(((real[t].astype(np.float32) - mean) / s)[None, None, ...]).to(device)
            continue

        # Normal autoregressive step
        y = model(x)  # normalized
        y_denorm = (y.squeeze(0).squeeze(0).detach().cpu().float().numpy() * s + mean)
        pred[t] = y_denorm.astype(np.float32)
        x = y

    return pred, injection_indices


def write_predictions_h5(
    out_path: str | Path,
    pred: np.ndarray,
    *,
    attrs_real: Dict[str, Any],
    key: str = "vorticity",
    source_checkpoint: Optional[str] = None,
    mode: str = "free",
    block_n: Optional[int] = None,
    injection_indices: Optional[List[int]] = None,
) -> Path:
    """
    Write predicted trajectory to HDF5 with consistent attributes.
    - Keep dt identical to real (full-length prediction).
    - Add provenance fields: mode, block_n, injection_indices, source_checkpoint, source_kind.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    H5OPTS = dict(chunks=(1, pred.shape[1], pred.shape[2]), compression="gzip",
                  compression_opts=4, shuffle=True)

    with h5py.File(out_path, "w") as f:
        dset = f.create_dataset(key, data=pred.astype(np.float32), dtype="float32", **H5OPTS)
        dset.attrs["description"] = "Predicted vorticity (time, x, y)"
        dset.attrs["kind"] = str(attrs_real.get("kind"))

        # Physical meta copied from real
        dset.attrs["dt"] = float(attrs_real.get("dt"))
        dset.attrs["time"] = np.asarray(attrs_real.get("time"))
        dset.attrs["viscosity"] = float(attrs_real.get("viscosity"))
        dset.attrs["x_min"] = float(attrs_real.get("x_min"))
        dset.attrs["x_max"] = float(attrs_real.get("x_max"))
        dset.attrs["y_min"] = float(attrs_real.get("y_min"))
        dset.attrs["y_max"] = float(attrs_real.get("y_max"))
        dset.attrs["inner_steps"] = int(attrs_real.get("inner_steps"))
        dset.attrs["outer_steps"] = int(attrs_real.get("outer_steps"))
        dset.attrs["final_time"] = float(attrs_real.get("final_time"))
        dset.attrs["solver_iteration_time"] = float(attrs_real.get("solver_iteration_time"))

        #stats
        stats_mean = float(pred.mean())
        stats_var  = float(pred.var())
        stats_std  = float(pred.std())
        stats_min  = float(pred.min())
        stats_max  = float(pred.max())
        dset.attrs['mean'] = stats_mean
        dset.attrs['var']  = stats_var
        dset.attrs['std']  = stats_std
        dset.attrs['min']  = stats_min
        dset.attrs['max']  = stats_max

        # Provenance
        if source_checkpoint is not None:
            dset.attrs["source_checkpoint"] = str(source_checkpoint)
        dset.attrs["rollout_mode"] = str(mode)
        if block_n is not None:
            dset.attrs["block_n"] = int(block_n)
        if injection_indices is not None:
            dset.attrs["injection_indices"] = np.asarray(injection_indices, dtype=np.int64)

    return out_path


def infer_one_experiment(
    model: torch.nn.Module,
    device: torch.device,
    *,
    experiment_dir: str | Path,
    h5_name: str = "vorticity.h5",
    key: str = "vorticity",
    mode: str = "free",          # "free" | "chunked"
    block_n: Optional[int] = None,
    seed_index: int = 0,
    out_path: Optional[str | Path] = None,
    source_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    High-level helper: load real HDF5, run rollout (free or chunked), write HDF5 predictions.

    Returns a dict with:
      - "pred": (T,H,W) predicted (denormalized)
      - "real": (T,H,W) real (full)
      - "injection_indices": list[int] (empty in free mode)
      - "real_attrs": dict
      - "pred_path": Path
    """
    experiment_dir = Path(experiment_dir).expanduser().resolve()
    simulation_h5 = experiment_dir / h5_name
    real, attrs = read_real_h5(simulation_h5, key=key)

    mean = float(attrs["mean"])
    std = float(attrs["std"])

    if mode == "free":
        pred = rollout_free_run(model, device, real, mean=mean, std=std, seed_index=seed_index)
        injection_indices: List[int] = []
    elif mode == "chunked":
        if block_n is None:
            raise ValueError("mode='chunked' requires block_n (number of autoregressive steps between injections).")
        pred, injection_indices = rollout_chunked_teacher_forcing(
            model, device, real, mean=mean, std=std, seed_index=seed_index, block_n=block_n
        )
    else:
        raise ValueError("mode must be 'free' or 'chunked'.")

    T_r, H_r, W_r = real.shape
    T_p, H_p, W_p = pred.shape
    if (T_p, H_p, W_p) != (T_r, H_r, W_r):
        raise ValueError(
            f"Prediction shape {pred.shape} must match real shape {real.shape} "
            f"(mode='{mode}', seed_index={seed_index}, block_n={block_n})."
        )

    #if out_path is None:
    #    suffix = f"{mode}" if mode == "free" else f"{mode}_n{int(block_n)}"
    #    out_path = experiment_dir / f"predictions_{suffix}.h5"

    pred_path = write_predictions_h5(
        out_path, pred, attrs_real=attrs, key=key,
        source_checkpoint=source_checkpoint, mode=mode, block_n=block_n,
        injection_indices=injection_indices if injection_indices else None,
    )

    return {
        "pred": pred,
        "real": real.astype(np.float32),
        "injection_indices": injection_indices,
        "real_attrs": attrs,
        "pred_path": pred_path,
    }

if __name__ == "__main__":
    arr, attrs = read_real_h5('/scratch/algo/aitalla/StageGitlab/data/ForcedTurbulence/force1024x64x64/vorticity.h5')
    print(arr.shape)

