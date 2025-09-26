
from __future__ import annotations


import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import xarray

import jax_cfd.base as cfd
import jax_cfd.base.grids as grids
import jax_cfd.spectral as spectral

import dataclasses  


import os
import shutil
import time
import re
from typing import Dict, Any, Tuple, Optional
import h5py
import yaml
import numpy as np


class BaseTurbulenceGenerator:
    """
    Base class for turbulence data generation. Loads YAML config, prepares the
    run directory, runs the simulation, and writes an HDF5 dataset.
    """

    DEFAULT_DATASET_NAME = "vorticity"

    def __init__(self, config_path: str, expe_name: Optional[str] = None):
        self.config_path = config_path
        self.expe_name = expe_name  
        self.cfg: Dict[str, Any] = self._load_cfg(config_path)
        self._extract_values(self.cfg)  

       
        self.run_dir: Optional[str] = None
        self.cfg_copy_path: Optional[str] = None
        self.out_path: Optional[str] = None

    def _load_cfg(self, path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f) or {}
            return cfg

    def _sanitize_name(self, name: str) -> str:
        name = name.strip().replace(" ", "_")
        return re.sub(r"[^A-Za-z0-9._\\-]", "-", name)

    def _prepare_run_dir(self) -> None:
        assert isinstance(self.kind, str)
        ts = time.strftime("%Y%m%d-%H%M%S")

        if self.expe_name and self.expe_name.strip():
            run_name = self._sanitize_name(self.expe_name)
        else:
            run_name = f"gen_{ts}_{self.kind}_seed{self.seed}"

        kind_to_subdir = {
        "decaying": "DecayingTurbulence",
        "forced":   "ForcedTurbulence",  
        }
        if self.kind not in kind_to_subdir:
            raise ValueError(f"Unknown kind {self.kind!r}. Expected 'decaying' or 'forced'.")

        subdir = kind_to_subdir[self.kind]

        run_dir = os.path.join("/scratch/algo/aitalla/StageGitlab/data", subdir, run_name)

        os.makedirs(run_dir, exist_ok=True)

        cfg_copy_path = os.path.join(run_dir, os.path.basename(self.config_path))
        shutil.copyfile(self.config_path, cfg_copy_path)

   
        out_path = os.path.join(run_dir, f"{self.dataset_name}.h5")

        self.run_dir = run_dir
        self.cfg_copy_path = cfg_copy_path
        self.out_path = out_path

    def _extract_values(self, cfg: Dict[str, Any]) -> None:
        """
        Extracts scalar and structured values from the YAML config and assigns
        them to instance attributes with appropriate types/defaults.
        """
        self.viscosity    = float(cfg.get("viscosity", 1e-3))
        self.max_velocity = float(cfg.get("max_velocity", 7))
        self.cfl          = float(cfg.get("cfl", 0.5))
        self.smooth       = bool(cfg.get("smooth", True))

        
        self.final_time   = float(cfg.get("final_time", 2500.0))
        self.outer_steps  = int(cfg.get("frames", 1024))
        
        
        self.seed         = int(cfg.get("seed", 42))
        self.init_kmax    = int(cfg.get("init_kmax", 4))

        
        g = cfg.get("grid", {}) or {}
        d = cfg.get("domain", {}) or {}
        self.nx = int(g.get("nx", 256))
        self.ny = int(g.get("ny", 256))
        self.x_min = float(d.get("x_min", 0.0))
        self.x_max = float(d.get("x_max", 2 * jnp.pi))
        self.y_min = float(d.get("y_min", 0.0))
        self.y_max = float(d.get("y_max", 2 * jnp.pi))

        
        self.dataset_name = str(cfg.get("dataset_name", self.DEFAULT_DATASET_NAME))

    
    def run(self) -> str:
        """Run the full simulation and write the HDF5 file under the run directory."""
        self._prepare_run_dir()
        arr, dt = self._simulate()  

        
        arr = np.asarray(arr, dtype=np.float32)
        T, H, W = int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
        
        stats_mean = float(arr.mean())
        stats_var  = float(arr.var())
        stats_std  = float(arr.std())
        stats_min  = float(arr.min())
        stats_max  = float(arr.max())

        with h5py.File(self.out_path, 'w') as f:
            dset = f.create_dataset(
                self.dataset_name,
                data=arr,
                dtype='float32',
                chunks=(1, H, W),
                compression='gzip',
                compression_opts=4,
                shuffle=True
            )
            dset.attrs['description'] = 'Simulated vorticity (time, x, y)'
            dset.attrs['dt'] = float(dt)
            dset.attrs['viscosity'] = float(self.viscosity)
            dset.attrs['kind'] = self.kind
            dset.attrs['outer_steps'] = float(self.outer_steps)
            dset.attrs['inner_steps'] = float((self.final_time // dt) // self.outer_steps)
            dset.attrs['final_time'] = float(self.final_time)
            dset.attrs['time']= dt * jnp.arange(self.outer_steps) * ((self.final_time // dt) // self.outer_steps)
            dset.attrs['mean'] = stats_mean
            dset.attrs['var']  = stats_var
            dset.attrs['std']  = stats_std
            dset.attrs['min']  = stats_min
            dset.attrs['max']  = stats_max
            dset.attrs['solver_iteration_time'] = float(0)
            dset.attrs['x_min'] = float(self.x_min)
            dset.attrs['x_max'] = float(self.x_max)
            dset.attrs['y_min'] = float(self.y_min)
            dset.attrs['y_max'] = float(self.y_max)

        print(f"[OK] HDF5 written : {self.out_path}")
        print(f"[OK] YAML copied  : {self.cfg_copy_path}")
        return self.out_path

    def _simulate(self) -> Tuple[np.ndarray, float]:
        """Return a tuple (vorticity_array[T, H, W], dt)."""
        raise NotImplementedError


class DecayingGenerator(BaseTurbulenceGenerator):
    """Decaying turbulence simulation """

    def __init__(self, config_path: str, expe_name: Optional[str] = None):
        super().__init__(config_path, expe_name)
        self.kind = "decaying"

    def _simulate(self) -> Tuple[np.ndarray, float]:
        print(self.kind)
        grid = grids.Grid((self.nx, self.ny),
                          domain=((self.x_min, self.x_max), (self.y_min, self.y_max)))
        dt = cfd.equations.stable_time_step(self.max_velocity, self.cfl, self.viscosity, grid)

        smooth = True if self.smooth else False
        step_fn = spectral.time_stepping.crank_nicolson_rk4(
            spectral.equations.NavierStokes2D(self.viscosity, grid, smooth=smooth), dt)

        inner_steps = (self.final_time // dt) // self.outer_steps

        trajectory_fn = cfd.funcutils.trajectory(
            cfd.funcutils.repeated(step_fn, inner_steps), self.outer_steps)

        v0 = cfd.initial_conditions.filtered_velocity_field(
            jax.random.PRNGKey(self.seed), grid, self.max_velocity, self.init_kmax)
        vorticity0 = cfd.finite_differences.curl_2d(v0).data
        vorticity_hat0 = jnp.fft.rfftn(vorticity0)

        _, trajectory = trajectory_fn(vorticity_hat0)
        vorticity_space = jnp.fft.irfftn(trajectory, axes=(1, 2))
        return vorticity_space, float(dt)


class ForcedGenerator(BaseTurbulenceGenerator):
    """Decaying turbulence simulation """

    def __init__(self, config_path: str, expe_name: Optional[str] = None):
        super().__init__(config_path, expe_name)
        self.kind = "forced"

    def _simulate(self) -> Tuple[np.ndarray, float]:
        print(self.kind)
        grid = grids.Grid((self.nx, self.ny),
                          domain=((self.x_min, self.x_max), (self.y_min, self.y_max)))
        dt = cfd.equations.stable_time_step(self.max_velocity, self.cfl, self.viscosity, grid)

        smooth = True if self.smooth else False
        step_fn = spectral.time_stepping.crank_nicolson_rk4(
            spectral.equations.ForcedNavierStokes2D(self.viscosity, grid, smooth=smooth), dt)

        inner_steps = (self.final_time // dt) // self.outer_steps

        trajectory_fn = cfd.funcutils.trajectory(
            cfd.funcutils.repeated(step_fn, inner_steps), self.outer_steps)

        v0 = cfd.initial_conditions.filtered_velocity_field(
            jax.random.PRNGKey(self.seed), grid, self.max_velocity, self.init_kmax)
        vorticity0 = cfd.finite_differences.curl_2d(v0).data
        vorticity_hat0 = jnp.fft.rfftn(vorticity0)

        _, trajectory = trajectory_fn(vorticity_hat0)
        vorticity_space = jnp.fft.irfftn(trajectory, axes=(1, 2))
        return vorticity_space, float(dt)
