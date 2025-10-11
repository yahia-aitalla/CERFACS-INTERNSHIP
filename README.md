# A modular pipeline for surrogate modeling of 2D CFD (forced & decaying turbulence) with offline training and online-learning emulation.

This repository provides an end-to-end, reproducible stack to (i) generate controlled 2D turbulence datasets using a **spectral solver**, (ii) train a **UNet** surrogate with **multi-step rollout supervision** (with **curriculum** on horizon), (iii) emulate **online/streaming training** (no epochs), and (iv) run **inference & diagnostics** (TKE, isotropic spectra, GIFs). The document is neutral and focuses on **objectives, capabilities, and usage**.

> Data generation relies on the spectral Navier–Stokes solver implemented in **JAX-CFD**. Please cite the software appropriately: Google Research, “JAX-CFD: Computational Fluid Dynamics in JAX,” GitHub, https://github.com/google/jax-cfd (accessed YYYY-MM-DD).

---

## Table of Contents
1. [Scope & Objectives](#scope--objectives)  
2. [Features](#features)  
3. [Repository Structure](#repository-structure)  
4. [System Requirements](#system-requirements)  
5. [Two Environments (Recommended)](#two-environments-recommended)  
6. [Local Paths & Placeholders](#local-paths--placeholders)  
7. [Quick Start](#quick-start)  
8. [Data Generation (Spectral / JAX-CFD)](#data-generation-spectral--jax-cfd)  
9. [Datasets & Normalization](#datasets--normalization)  
10. [Model (UNet)](#model-unet)  
11. [Training from `src/`](#training-from-src)  
    - [Offline (Curriculum)](#offline-curriculum)  
    - [Online (Streaming Emulation)](#online-streaming-emulation)  
    - [Rollout Loss — Mathematical Formulation](#rollout-loss--mathematical-formulation)  
12. [Inference from `src/`](#inference-from-src)  
13. [Artifacts & Reproducibility](#artifacts--reproducibility)  
14. [Troubleshooting](#troubleshooting)  
15. [Acknowledgements](#acknowledgements)  
16. [License](#license)

---

## Scope & Objectives
- **Motivation.** CFD solvers produce long trajectories and high data rates. Persisting complete offline corpora is often infeasible. This repository targets:  
  (i) **surrogate modeling** of 2D turbulence (forced/decaying) with **UNet**;  
  (ii) **online-learning emulation** for streaming/continual settings without epochs.
- **Goal.** Provide a clean, configurable baseline for offline and online training, plus inference and physical diagnostics.

---

## Features
- **Spectral data generation** (JAX-CFD) to HDF5 with physical metadata.
- **Dataset** with on-the-fly **normalization** (from HDF5 attributes) and **multi-step targets**.
- **UNet** surrogate; multi-step **rollout** implemented in the **training loop**.
- **Offline curriculum** on the rollout horizon (e.g., 1→2→4→8).
- **Online emulation** (producer/consumer ring buffer) with target-loss or round-limit stopping.
- **Inference** (free or chunked) + **TKE** and **isotropic spectra** + GIFs.
- **YAML-driven** configuration; per-run config snapshot; CSV logs, TensorBoard events, checkpoints, figures.

---

## Repository Structure
```
configs/
  data/            # data generation (decaying/, forced/)
  train/           # training (offline/, online/)
  infer/           # inference
data/              # generated experiments (DecayingTurbulence/, ForcedTurbulence/)
predictions/       # inference outputs (.h5 + figures) per run
runs/              # training runs (checkpoints, logs, events, config copies)
src/
  datagen/         # JAX-CFD generation + HDF5 writer
  datasets/        # VorticityDataset, dataloader helpers
  inference/       # rollout core, metrics, plotting, CLI
  models/          # UNet
  training/        # offline.py, online.py, trainer.py, losses.py, train.py
README.md
requirements.txt                     # PyTorch env (training/inference)
src/datagen/requirements_jaxcfd.txt  # JAX/JAX-CFD env (generation)
```

> **Always execute from the `src/` directory** using `python -m <package.module>` so that imports resolve exactly as intended.

---

## System Requirements
- **Python** ≥ 3.10  
- **GPU** recommended for training (PyTorch). JAX-CFD supports CPU/GPU.  
- Core libs: PyTorch, NumPy, h5py, PyYAML, Matplotlib, TensorBoard.  
- Generation requires JAX + JAX-CFD (CPU or GPU build depending on your platform).

> HDF5 on network filesystems: prefer `num_workers=0` in DataLoaders to avoid I/O issues.

---

## Two Environments (Recommended)
Use **two separate virtualenvs** to avoid CUDA/cuDNN conflicts.

### A) Training/Inference (PyTorch)
```bash
# from repo root
python -m venv .venv_torch
# Linux/macOS
source .venv_torch/bin/activate
# Windows (PowerShell)
# .venv_torch\Scripts\Activate.ps1

pip install -r requirements.txt
```

### B) Data Generation (JAX-CFD)
```bash
# from repo root
python -m venv .venv_jax
# Linux/macOS
source .venv_jax/bin/activate
# Windows (PowerShell)
# .venv_jax\Scripts\Activate.ps1

pip install -r src/datagen/requirements_jaxcfd.txt
```

> Keep them **separate**: do not install JAX in `.venv_torch` or PyTorch in `.venv_jax`.

---

## Local Paths & Placeholders
Some scripts accept `--config` but it is **optional**. If `--config` is **omitted**, a **default config path** hard-coded in the script is used. Those defaults are **absolute** and machine-specific.  
Two options:
- **(Recommended)** Pass `--config` explicitly with **your own path**.  
- **(Alternative)** Edit the hard-coded defaults in the script so that the default path points to your local repo.

**Placeholder convention used below**  
- `<ABS_REPO>` = **absolute path to your cloned repo**.  
  - Linux/macOS: run `pwd` at the repo root.  
  - PowerShell: run `Get-Location` at the repo root.  
- Replace `<ABS_REPO>` with your real absolute path (keep quotes on Windows if spaces).

> All commands below are launched **from the `src/` directory**.

---

## Quick Start
```bash
# 1) (JAX env) generate data - forced example
cd <ABS_REPO>/src
python -m datagen.generate \
  --config <ABS_REPO>/configs/data/forced/forced.yaml \
  --expe_name forced_seed42

# 2) (PyTorch env) train offline
cd <ABS_REPO>/src
python -m training.train \
  --config <ABS_REPO>/configs/train/offline/offline.yaml \
  --expe_name unet_offline_baseline

# 3) (PyTorch env) inference
cd <ABS_REPO>/src
python -m inference.infer \
  --config <ABS_REPO>/configs/infer/infer.yaml
```

---

## Data Generation (Spectral / JAX-CFD)
Activate **`.venv_jax`**. The generator selects **forced** vs **decaying** from the **parent folder** of the YAML path.

### CLI (from `src/`)
Forced:
```bash
python -m datagen.generate \
  --config <ABS_REPO>/configs/data/forced/forced.yaml \
  [--expe_name <NAME>]
```
Decaying:
```bash
python -m datagen.generate \
  --config <ABS_REPO>/configs/data/decaying/decaying.yaml \
  [--expe_name <NAME>]
```

**Arguments**
- `--config` (optional but recommended): YAML under `configs/data/{forced|decaying}/`.  
  If omitted, the script uses its **internal default** path; ensure it points to your machine (edit the constant if needed).
- `--expe_name` (optional): explicit output directory name. If omitted, a timestamped name is auto-generated as `gen_YYYYMMDD-HHMMSS_{forced|decaying}_seedXX` (seed from YAML).

**Outputs**
```
<ABS_REPO>/data/ForcedTurbulence/<EXP_NAME>/{vorticity.h5, config.yaml}
<ABS_REPO>/data/DecayingTurbulence/<EXP_NAME>/{vorticity.h5, config.yaml}
```

---

## Datasets & Normalization
`src/datasets/VorticityDataset.py` reads `vorticity.h5`, applies **normalization** from file attributes, and returns samples `(x_t, Y_t)` where `Y_t = (x_{t+1},…,x_{t+n})` with `n = nstep`. Concatenation across experiments is supported. For training, the loader is built with `nstep = max(curr_lr_steps)`; the loop truncates to the current curriculum horizon.

---

## Model (UNet)
UNet encoder–decoder (4 scales) with skip connections. Building block: `(Conv3×3 → GroupNorm(1) → ReLU) × 2`. `padding_mode="circular"` is available for periodic domains. **Rollout** is implemented in the **training loop** (UNet predicts 1 step).

---

## Training from `src/`
Activate **`.venv_torch`**.

### Offline (Curriculum)
```bash
python -m training.train \
  [--config <ABS_REPO>/configs/train/offline/offline.yaml] \
  [--expe_name <RUN_NAME>]
```
**Scenarios**
- **With `--config`**: strategy = **offline** (path must be under `configs/train/offline/`); YAML controls data, model, optim, curriculum and epochs.  
- **Without `--config`**: the script uses its internal **offline default** path; **edit it** in the source if it does not exist on your machine.

**Typical YAML fields**
- `data`: `experiments_dirs` (list of `<ABS_REPO>/data/.../gen_...`), `h5_name`, `key`, `batch_size`, `shuffle`, `num_workers`, `pin_memory`, optional `db_size`.  
- `model`: `in_channels`, `num_classes`, `padding_mode`, `padding`.  
- `optim`: learning rate, etc.  
- `curr_lr_steps`: rollout horizons, e.g., `[1,2,4,8]`.  
- `train`: `epochs` per stage.

**Outputs**
```
<ABS_REPO>/runs/offline/<RUN_NAME>/
  checkpoints/  events/  logs/  config.yaml
```

### Online (Streaming Emulation)
```bash
python -m training.train \
  [--config <ABS_REPO>/configs/train/online/online.yaml] \
  [--expe_name <RUN_NAME>]
```
**Scenarios**
- **With `--config`**: strategy = **online** (path must be under `configs/train/online/`); YAML controls data + `buffer_capacity`, `producer_dt`, `curr_lr_steps: [h]`, `target_loss`, `max_rounds`.  
- **Without `--config`**: the script uses its internal **online default** path; **edit it** in the source if it does not exist on your machine.

**Outputs**
```
<ABS_REPO>/runs/online/<RUN_NAME>/
  checkpoints/  events/  logs/  config.yaml
```

### Rollout Loss — Mathematical Formulation
Let normalized input at time \(t\) be \(x_t\). The UNet \(f_\theta\) predicts one step ahead; multi-step rollout is built iteratively:
\[
\hat{x}_{t+1} = f_\theta(x_t),\quad
\hat{x}_{t+2} = f_\theta(\hat{x}_{t+1}),\ \dots,\
\hat{x}_{t+k} = f_\theta(\hat{x}_{t+k-1}).
\]
For horizon \(h\), the per-sample rollout loss is:
\[
\mathcal{L}_\text{rollout}^{(h)}(t;\theta) =
\sum_{k=1}^{h} w_k\,\ell\!\big(\hat{x}_{t+k}, x_{t+k}\big),
\]
where \(\ell\) is a pointwise discrepancy (e.g., MSE) and \(w_k\ge 0\) optional weights (uniform by default). With a windowed physical penalty (e.g., TKE), one may use:
\[
\mathcal{L}^{(h)}(t;\theta)
= \sum_{k=1}^{h} w_k\,\ell_\text{MSE}(\hat{x}_{t+k}, x_{t+k})
\;+\; \lambda\,\Phi\!\big(\hat{x}_{t+1:t+h}, x_{t+1:t+h}\big),
\]
with \(\Phi\) aggregating a physical discrepancy over the window and \(\lambda\ge 0\).  
**Implementation note:** the active loss is **MSE** by default; TKE/TKEMSE variants exist but are disabled in the current CLI.

---

## Inference from `src/`
```bash
python -m inference.infer \
  [--config <ABS_REPO>/configs/infer/infer.yaml]
```
**Scenarios**
- **With `--config`**: the YAML specifies UNet hyperparameters and `checkpoint` path under `<ABS_REPO>/runs/.../checkpoints/...`, rollout mode (`free`/`chunked`), `seed_index`, `block_n` (if chunked), and a data source (`experiment_dir`, `h5_name`, `key`).  
- **Without `--config`**: the script uses its internal default. Ensure that default exists on your machine (edit constant if needed).

**Outputs**
```
<ABS_REPO>/predictions/<RUN_NAME>/
  config.yaml
  preds/vorticity.h5
  figures/ {tke_timeseries.png, energy_spectrum.png, prediction_vs_simulation.gif}
```

---

## Artifacts & Reproducibility
- Per-run **config snapshot**.
- **CSV logs** (batch/epoch offline; round-based online).
- **TensorBoard** under `events//`.
- **Checkpoints** (periodic + final).
- **Figures** (training loss curves; inference diagnostics).
- Seeds configurable in YAML for generation and training.

---

## Troubleshooting
- **ModuleNotFoundError** → Always run **from `src/`** with `python -m ...`.  
- **HDF5 I/O issues** → set `num_workers: 0` (especially on NFS).  
- **PyTorch CUDA not visible** → check `nvidia-smi` and install the correct wheel for your CUDA.  
- **JAX/JAX-CFD install** → start with CPU wheels; switch to GPU following official JAX instructions for your CUDA.  
- **Defaults without `--config`** → edit hard-coded default paths in the source to point to your `<ABS_REPO>`.

---

## Acknowledgements
Data generation uses **JAX-CFD** (spectral Navier–Stokes solver). Please cite:  
- Google Research, “**JAX-CFD: Computational Fluid Dynamics in JAX**,” GitHub. https://github.com/google/jax-cfd (accessed YYYY-MM-DD).

---

## License
Specify the license applicable to this repository (e.g., MIT).


---
<img width="3465" height="2058" alt="losses_grid_forced2" src="https://github.com/user-attachments/assets/ab98b3b3-54db-4ecf-87bd-63d4c27aa7fc" />
<img width="3529" height="2058" alt="losses_grid_decaying" src="https://github.com/user-attachments/assets/23a5e93b-e96b-45f9-99e5-577895a0770b" />
<img width="5635" height="2274" alt="final_grid_tke_64x64vorticity" src="https://github.com/user-attachments/assets/36c0f107-657f-4c3f-860c-22334bc32fc4" />
<img width="4518" height="2274" alt="final_grid_tke_256x256vorticity" src="https://github.com/user-attachments/assets/7418b143-3d7e-45a9-aae2-be2db72efc2f" />
<img width="3937" height="2337" alt="tke_rows_forced_free" src="https://github.com/user-attachments/assets/f874f8d3-458a-4c39-8a4c-82ae5dc0cc2c" />
