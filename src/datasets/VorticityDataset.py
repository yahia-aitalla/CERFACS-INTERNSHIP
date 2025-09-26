from __future__ import annotations

from typing import Optional, Sequence
from pathlib import Path
import os
import h5py
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, DataLoader
import yaml 

def _read_stats_from_h5(h5_path: Path, key: str) -> dict:
    """Read mean/var/std/min/max/domain from HDF5 dataset attributes."""
    with h5py.File(os.fspath(h5_path), "r") as f:
        if key not in f:
            available = ", ".join(list(f.keys()))
            raise KeyError(
                f"HDF5 key '{key}' not found in {h5_path}. "
                f"Available: {available}"
            )
        attrs = f[key].attrs
        missing = [k for k in ("mean", "var", "std", "min", "max", "solver_iteration_time", "x_min", "x_max", "y_min", "y_max", "kind") if k not in attrs]
        if missing:
            raise KeyError(
                f"Missing attributes in {h5_path} / {key}: {missing}. "
                f"Regenerate the data with stats writing."
            )
        
        domain = (
                float(attrs["x_min"]),
                float(attrs["x_max"]),
                float(attrs["y_min"]),
                float(attrs["y_max"]),
            )
        
        return {
            "mean": float(attrs["mean"]),
            "var":  float(attrs["var"]),
            "std":  float(attrs["std"]),
            "min":  float(attrs["min"]),
            "max":  float(attrs["max"]),
            "kind": str(attrs["kind"]),
            "domain": domain
        }


def _h5_path_from_experiment(experiment_dir: str | Path, h5_name: str = "vorticity.h5") -> Path:
    """Return absolute HDF5 path '<experiment_dir>/<h5_name>' with checks."""
    exp = Path(experiment_dir).expanduser().resolve()
    if not exp.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {exp}")
    h5_path = exp / h5_name
    if not h5_path.is_file():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
    return h5_path


class VorDataset(Dataset):
    """
    HDF5 vorticity dataset (T, H, W) -> (x, y) where:
      - x = frame at time t, normalized, shape (1, H, W)
      - y = sequence [t+1, ..., t+n], normalized, shape (n, H, W)
    Stats are read exclusively from HDF5 attributes.
    """

    def __init__(
        self,
        experiment_dir: str | Path,
        *,
        h5_name: str = "vorticity.h5",
        key: str = "vorticity",
        nstep: int = 16,
        db_size: Optional[int] = None,
        dtype: torch.dtype | str = "float32",
    ) -> None:
        super().__init__()
        self.key = str(key)
        self.nstep = int(nstep)
        self.db_size = int(db_size) if db_size is not None else None
        self.torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

        # Resolve the HDF5 path from the absolute experiment directory
        self.h5_path = _h5_path_from_experiment(experiment_dir, h5_name=h5_name)

        # Open file and dataset
        self._file = h5py.File(os.fspath(self.h5_path), "r")
        if self.key not in self._file:
            available = ", ".join(list(self._file.keys()))
            self._file.close()
            raise KeyError(
                f"HDF5 key '{self.key}' not found in {self.h5_path}. "
                f"Available: {available}"
            )
        self._ds = self._file[self.key]  # expected shape: (T, H, W)

        stats = _read_stats_from_h5(self.h5_path, self.key)
        self.domain = tuple(stats["domain"])
        self.mean = float(stats["mean"])
        self.var  = float(stats["var"])
        self.std  = float(max(stats["std"], 1e-12))  
        self._stats = stats
        self.kind = str(stats["kind"])
        # Effective temporal length
        T = int(self._ds.shape[0])
        self._T_eff = min(T, self.db_size) if self.db_size is not None else T
        if self._T_eff <= self.nstep:
            self._file.close()
            raise ValueError(
                f"T_effective={self._T_eff} <= nstep={self.nstep}. "
                f"Increase db_size or reduce nstep."
            )
        self._len = self._T_eff - self.nstep

        # Cache (H, W) for info
        _, self._H, self._W = self._ds.shape

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # x = frame t ; y = [t+1, ..., t+n]
        img = self._ds[index]                               # (H, W)
        mask = self._ds[index + 1 : index + 1 + self.nstep] # (n, H, W)

        # Normalize with HDF5 stats
        img = (img - self.mean) / self.std
        mask = (mask - self.mean) / self.std

        # To torch tensors
        x = torch.from_numpy(np.asarray(img)).unsqueeze(0).to(self.torch_dtype)  # (1, H, W)
        y = torch.from_numpy(np.asarray(mask)).to(self.torch_dtype)              # (n, H, W)
        return x, y

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return (T_effective, H, W)."""
        return self._T_eff, self._H, self._W

    @property
    def stats(self) -> dict:
        """Return normalization stats (HDF5 attributes)."""
        return dict(self._stats)

    def close(self) -> None:
        try:
            if getattr(self, "_file", None) is not None:
                self._file.close()
        finally:
            self._file = None
            self._ds = None

    def __del__(self):
        self.close()


def build_dataset(
    experiments_dirs: Sequence[str | Path],
    *,
    h5_name: str = "vorticity.h5",
    key: str = "vorticity",
    nstep: int = 16,
    db_size: Optional[int] = None,
    dtype: torch.dtype | str = "float32",
) -> Dataset:
    """Build a single VorDataset if 1 dir, otherwise a ConcatDataset of many."""
    datasets: list[VorDataset] = []
    for d in experiments_dirs:
        datasets.append(
            VorDataset(
                experiment_dir=d,      
                h5_name=h5_name,
                key=key,
                nstep=nstep,
                db_size=db_size,
                dtype=dtype,
            )
        )
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def make_dataloader(
    experiments_dirs: Sequence[str | Path],
    *,
    h5_name: str = "vorticity.h5",
    key: str = "vorticity",
    nstep: int = 16,
    db_size: Optional[int] = None,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    dtype: torch.dtype | str = "float32",
) -> DataLoader:
    """Same as above, then wrap with a DataLoader."""
    ds = build_dataset(
        experiments_dirs,
        h5_name=h5_name,
        key=key,
        nstep=nstep,
        db_size=db_size,
        dtype=dtype,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

