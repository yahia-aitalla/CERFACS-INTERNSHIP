# Modular Pipeline for 2D CFD Surrogate Modeling (Offline & Online)

A modular, reproducible software stack for surrogate modeling of two-dimensional Navier–Stokes flows with convolutional neural networks. The stack provides **data generation**, **dataset construction**, **model definitions**, **offline training**, an **online-learning emulation pipeline**, and **inference and diagnostics**. The design emphasizes **general, machine-agnostic usage** and **reproducible configuration**.

> This project **does not** report results in this document. The focus is solely on **goals**, **motivations**, **capabilities**, and a **clear user guide**.

---

## 1. Scope and Motivation

Modern CFD solvers generate high-rate streams of high-dimensional states. Persisting large offline datasets can become infeasible due to storage and I/O constraints. This repository targets two complementary needs:

- **Surrogate modeling** for 2D turbulence (forced and decaying) using CNN-based predictors with multi-step rollout.
- **Streaming/online training emulation**, enabling experimentation without requiring the full dataset to be stored on disk, and allowing study of continual learning behaviors under data-stream constraints.

**Why an online-style pipeline?**
- In many CFD scenarios, the unique realizations and/or long trajectories produce terabytes of data. Online-style training allows consuming the stream *as it is produced* (or loaded incrementally), reducing disk pressure and improving I/O locality.
- Online-style training does **not** require epochs over a fixed corpus. Instead, it uses a flow of examples, buffering, and stopping criteria.

---

## 2. References to the Underlying CFD Solver

Data generation relies on the **spectral solver** implemented in **JAX-CFD**:

- **JAX-CFD** (software): Google Research, “JAX-CFD: Computational Fluid Dynamics in JAX,” GitHub repository, available at: https://github.com/google/jax-cfd (accessed 2025-10-10).

When publishing or sharing this work, please acknowledge JAX-CFD accordingly. If a formal citation entry is required, reference the software repository as above (or any official citation that the repository provides).

---

## 3. Repository Structure

```
repo-root/
├─ src/
│  ├─ datagen/          # Data generation via JAX-CFD + HDF5 writing
│  ├─ datasets/         # VorticityDataset, concatenation helpers, DataLoader helpers
│  ├─ models/           # UNet, AutoRegUNet
│  ├─ training/         # OfflineTrainer, OnlineTrainer (streaming emulation)
│  └─ inference/        # Free/chunked rollout, metrics (TKE, spectrum), plotting, GIF
├─ configs/
│  ├─ data/             # YAML configs: decaying/, forced/
│  ├─ train/            # YAML configs: offline/, online/
│  └─ infer/            # YAML config: inference settings
├─ data/
│  ├─ DecayingTurbulence/
│  └─ ForcedTurbulence/     # one subfolder per generated experiment
├─ runs/                # offline/ and online/ runs: checkpoints, logs, TensorBoard
└─ predictions/         # per-run predictions (.h5) and diagnostic figures
```

All `src/*` subpackages contain `__init__.py` to support `python -m` execution from the repository root.

---

## 4. System Requirements

- **Python** ≥ 3.10  
- **GPU** recommended (CUDA-enabled PyTorch and, optionally, JAX GPU builds)  
- Core libraries: **PyTorch**, **NumPy**, **h5py**, **PyYAML**, **Matplotlib**, **TensorBoard**  
- For data generation: **JAX** and **JAX-CFD** (CPU or GPU build as available)

> **Note on I/O:** When reading HDF5 from network filesystems, set `num_workers=0` in the DataLoader to avoid multiprocessing I/O contention.

---

## 5. Installation

```bash
# From repository root
python -m venv .venv

# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

JAX/JAX-CFD installations vary by platform; consult the JAX and JAX-CFD documentation for the appropriate wheels and versions.

---

## 6. Data Generation (Spectral Solver via JAX-CFD)

**Execution principle:** data are generated with a spectral Navier–Stokes integrator (Crank–Nicolson / Runge–Kutta in JAX-CFD), then stored as HDF5 with physical metadata and basic statistics.

**Entrypoint:** `src/datagen/generate.py` (always run with `python -m` from the repository root).

### 6.1 Forced (stationary) example
```bash
python -m src.datagen.generate   --config configs/data/forced/forced.yaml   --expe_name forced_seed42
```

### 6.2 Decaying (non-stationary) example
```bash
python -m src.datagen.generate   --config configs/data/decaying/decaying.yaml   --expe_name decaying_seed42
```

**Outputs:** each generation produces a new experiment directory such as:
```
data/ForcedTurbulence/gen_YYYYMMDD-HHMMSS_forced_seed42/{vorticity.h5, config.yaml}
data/DecayingTurbulence/gen_YYYYMMDD-HHMMSS_decaying_seed42/{vorticity.h5, config.yaml}
```

**HDF5 content (schema):**
- Dataset: `vorticity` with shape `(T, H, W)` and dtype `float32` (by default).
- Attributes (required):  
  - Statistical: `mean`, `std`, `var`, `min`, `max`  
  - Temporal: `dt`, `time` (vector length `T`)  
  - Simulation: `viscosity`, `inner_steps`, `outer_steps`, `final_time`, `solver_iteration_time`  
  - Domain: `x_min`, `x_max`, `y_min`, `y_max`  
  - Tag: `kind` in {{`forced`, `decaying`}}

---

## 7. Datasets and Normalization

**Module:** `src/datasets/VorticityDataset.py`

- **Normalization**: per-file `mean/std` are read from the HDF5 attributes, and applied on-the-fly.  
- **Samples**: for time index `t`, return input `x_t ∈ ℝ^{1×H×W}` and target sequence `y_{{t+1: t+n}} ∈ ℝ^{{n×H×W}}` with `n = nstep`.  
- **Effective length**: `len = T_effective − nstep` where `T_effective` equals `T` or a user-specified `db_size`.  
- **Concatenation**: `build_dataset([exp_dir1, exp_dir2, ...])` returns either a single dataset or a `ConcatDataset` that **does not** create cross-simulation windows.  
- **DataLoader**: `make_dataloader([...], nstep, batch_size, shuffle, num_workers, pin_memory)` wraps the dataset into a standard PyTorch DataLoader.

Example (Python API):
```python
from src.datasets.VorticityDataset import make_dataloader

loader = make_dataloader(
    ["data/ForcedTurbulence/gen_..."],
    nstep=4,
    batch_size=8,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
)
```

---

## 8. Models

**UNet**: encoder–decoder with skip connections (four scales). Each block applies `(Conv3×3 → GroupNorm(1) → ReLU) × 2`. `padding_mode` supports `"circular"` to reflect periodic CFD domains.

**AutoRegUNet**: extends UNet to output `n` consecutive steps in a single forward pass, concatenated on the channel dimension (`n × num_classes`). This aligns naturally with multi-step supervision from the dataset (`nstep`).

---

## 9. Training

**Entrypoint:** `src/training/train.py` (execute from repository root with `python -m`). The trainer is selected by the **location** of the YAML configuration.

### 9.1 Offline (Curriculum on Rollout Horizon)

```bash
python -m src.training.train   --config configs/train/offline/offline.yaml   --expe_name unet_offline_baseline
```
Key configuration concepts:
- `curr_lr_steps: [1, 2, 4, 8]` to progressively increase the supervised rollout horizon.
- `optim` (e.g., learning rate), `train.epochs`, batch size, workers, etc.

**Outputs:** `runs/offline/<run_name>/` with `checkpoints/`, `events/` (TensorBoard), `logs/` (CSV), and `loss_epochs.png`.

### 9.2 Online (Streaming Emulation)

```bash
python -m src.training.train   --config configs/train/online/online.yaml   --expe_name unet_online_emulated
```

**Principle of the online pipeline (emulation):**
- A **producer thread** generates (or selects) sample **indices** and pushes them into a **bounded ring buffer** at a configurable cadence (`producer_dt`, `buffer_capacity`).  
- The **consumer (training loop)** pops indices and constructs batches on-the-fly, drawing `(x_t, y_{{t+1: t+n}})` directly from the dataset.  
- Training proceeds without epochs; termination is controlled by **stopping criteria** (e.g., `target_loss`, `max_rounds`).  
- Optional hooks can implement **sample filtering**, simple **replay**, or curriculum over the rollout horizon (a single horizon is typical in online mode).

**Outputs:** `runs/online/<run_name>/` with `checkpoints/`, CSV logs, TensorBoard events, and `loss_rounds.png`.

---

## 10. Inference and Diagnostics

**Entrypoint:** `src/inference/infer.py` (execute with `python -m`).

```bash
python -m src.inference.infer   --config configs/infer/infer.yaml
```

**Configuration fields:**
- `model`: UNet hyperparameters and checkpoint path under `runs/.../checkpoints/...`  
- `rollout`: `mode` in {{`free`, `chunked`}}; `seed_index` for the initial frame; `block_n` if `chunked`  
- `data`: `experiment_dir` (e.g., `data/DecayingTurbulence/gen_...`), `h5_name`, `key`

**Outputs:**
```
predictions/<run_name>/
├─ config.yaml
├─ preds/vorticity.h5            # predicted vorticity sequence with metadata and stats
└─ figures/
   ├─ tke_timeseries.png         # kinetic energy over time (optional injection markers)
   ├─ energy_spectrum.png        # isotropic spectrum with optional reference slope overlays
   └─ prediction_vs_simulation.gif  # qualitative side-by-side animation
```

`src/inference/metrics.py` computes vorticity→velocity (FFT), TKE time series, and isotropic spectra; `src/inference/plotting.py` renders standard figures with consistent styles.

---

## 11. Configuration Files

All components are driven by YAML files under `configs/`:
- **Data generation**: `configs/data/decaying/*.yaml`, `configs/data/forced/*.yaml`  
  - viscosity, max velocity, CFL, smoothing flag, `final_time`, `frames`, domain bounds, grid resolution, RNG seed, dataset key, etc.
- **Training (offline/online)**: `configs/train/offline/*.yaml`, `configs/train/online/*.yaml`  
  - data paths (experiment directories), DataLoader options, model hyperparameters, optimizer settings, curriculum or online-specific parameters
- **Inference**: `configs/infer/infer.yaml`  
  - model hyperparameters, checkpoint path, rollout mode and parameters, data source

Each run persists a copy of the used configuration for traceability.

---

## 12. Reproducibility

- **Environment**: use a dedicated virtual environment to fix dependency versions.  
- **Determinism**: seeds are configurable for data generation and training.  
- **Logging**: CSV logs (batch/epoch for offline; round-based for online) and TensorBoard summaries are produced for each run.  
- **Artifacts**: checkpoints, configuration snapshots, prediction files, and figures are stored under `runs/` and `predictions/`.

---
<img width="3465" height="2058" alt="losses_grid_forced2" src="https://github.com/user-attachments/assets/ab98b3b3-54db-4ecf-87bd-63d4c27aa7fc" />
<img width="3529" height="2058" alt="losses_grid_decaying" src="https://github.com/user-attachments/assets/23a5e93b-e96b-45f9-99e5-577895a0770b" />
<img width="5635" height="2274" alt="final_grid_tke_64x64vorticity" src="https://github.com/user-attachments/assets/36c0f107-657f-4c3f-860c-22334bc32fc4" />
<img width="4518" height="2274" alt="final_grid_tke_256x256vorticity" src="https://github.com/user-attachments/assets/7418b143-3d7e-45a9-aae2-be2db72efc2f" />
<img width="3937" height="2337" alt="tke_rows_forced_free" src="https://github.com/user-attachments/assets/f874f8d3-458a-4c39-8a4c-82ae5dc0cc2c" />
