# CFDStreamSurrogate
**A modular pipeline for surrogate modeling of 2D CFD (forced & decaying turbulence) with offline training and online-learning emulation.**

This repository provides an **end-to-end, reproducible** stack to (i) generate controlled 2D turbulence datasets using a **spectral solver**, (ii) train a **UNet** surrogate with **multi‑step rollout supervision** (including **curriculum** on horizon), (iii) emulate **online/streaming training** without epochs, and (iv) run **inference & diagnostics** (TKE, isotropic energy spectra, GIFs). The document is neutral and focuses on **objectives, capabilities, and usage**. No experiment claims are reported here.

> Data generation relies on the spectral Navier–Stokes solver implemented in **JAX‑CFD**. Please acknowledge the software accordingly: Google Research, “JAX‑CFD: Computational Fluid Dynamics in JAX,” GitHub, https://github.com/google/jax-cfd (accessed YYYY‑MM‑DD).

---

## Table of Contents
1. [Scope & Objectives](#scope--objectives)  
2. [Key Features](#key-features)  
3. [Repository Structure](#repository-structure)  
4. [System Requirements](#system-requirements)  
5. [Two Environments (Recommended Setup)](#two-environments-recommended-setup)  
6. [Configure Local Paths (One-Time)](#configure-local-paths-one-time)  
7. [Quick Start (TL;DR)](#quick-start-tldr)  
8. [Data Generation (Spectral / JAX‑CFD)](#data-generation-spectral--jaxcfd)  
9. [Datasets & Normalization](#datasets--normalization)  
10. [Model (UNet)](#model-unet)  
11. [Training](#training)  
    - [Offline (Curriculum on Rollout Horizon)](#offline-curriculum-on-rollout-horizon)  
    - [Online (Streaming Emulation)](#online-streaming-emulation)  
    - [Rollout Loss — Mathematical Formulation](#rollout-loss--mathematical-formulation)  
12. [Inference & Diagnostics](#inference--diagnostics)  
13. [Artifacts, Logs & Reproducibility](#artifacts-logs--reproducibility)  
14. [Troubleshooting](#troubleshooting)  
15. [Acknowledgements & Citation](#acknowledgements--citation)  
16. [License](#license)

---

## Scope & Objectives
- **Motivation.** CFD solvers produce large, continuous streams of high‑dimensional states. Persisting full offline corpora is often infeasible due to storage/I/O constraints. This repository targets:  
  (i) **surrogate modeling** of 2D turbulence (forced ≈ stationary; decaying = non‑stationary) with **UNet**;  
  (ii) an **online‑learning emulation** to study streaming/continual training without epochs and to mimic practical CFD pipelines.
- **Goal.** Provide a clean, configurable baseline for **offline** training (with curriculum on rollout length) and an **online** emulation, plus tools for **inference** and **physical diagnostics**.

---

## Key Features
- **Data generation** (forced & decaying turbulence) using a **spectral Navier–Stokes** solver (JAX‑CFD) with HDF5 export + physical metadata.
- **Datasets & loaders** with **on‑the‑fly normalization** from HDF5 stats and **multi‑step targets** for rollout supervision.
- **UNet** surrogate; **rollout** is implemented **in the training loop** (no autoregressive wrapper).
- **Offline curriculum** on rollout horizon (e.g., 1 → 2 → 4 → 8) to stabilize optimization and expose the model to longer horizons progressively.
- **Online emulation** (producer/consumer index buffer): streaming‑style training **without epochs**, stopping on target loss or max rounds.
- **Inference** in free or chunked mode; **TKE** and **isotropic spectrum** metrics; consistent plotting and GIFs.
- **YAML‑driven configuration**, per‑run config snapshot, CSV logs, TensorBoard events, checkpoints, and figures.

---

## Repository Structure
```
configs/                      # YAML configs
  data/                       #   data generation (decaying/, forced/)
  train/                      #   training (offline/, online/)
  infer/                      #   inference
data/                         # generated experiments (DecayingTurbulence/, ForcedTurbulence/)
predictions/                  # inference outputs (.h5 + figures) per run
runs/                         # training runs: checkpoints, logs, events, config copies
src/                          # source code (all subpackages have __init__.py)
  datagen/                    #   JAX-CFD generation + HDF5 writer
  datasets/                   #   VorticityDataset, dataloader helpers
  inference/                  #   rollout core, metrics, plotting, CLI
  models/                     #   UNet
  training/                   #   offline.py, online.py, trainer.py, losses.py, train.py
README.md
requirements.txt              # core training/inference deps (PyTorch stack)
src/datagen/requirements_jaxcfd.txt   # JAX/JAX-CFD deps for data generation
```

Run **all** modules from the repository root using **`python -m`** (ensures imports resolve).

---

## System Requirements
- **Python** ≥ 3.10  
- **GPU** recommended for training (PyTorch). JAX‑CFD supports CPU or GPU.  
- Core libs: PyTorch, NumPy, h5py, PyYAML, Matplotlib, TensorBoard.  
- For generation: JAX, JAX‑CFD (CPU or GPU build depending on your platform).

> HDF5 on networked filesystems: prefer `num_workers=0` in DataLoaders to avoid I/O contention.

---

## Two Environments (Recommended Setup)
Two separate virtual environments are recommended to avoid CUDA/cuDNN conflicts between **PyTorch** and **JAX‑CFD**.

### A) Training/Inference environment (PyTorch)
```bash
# from repo root
python -m venv .venv_torch
# Linux/macOS
source .venv_torch/bin/activate
# Windows (PowerShell)
# .venv_torch\Scripts\Activate.ps1

pip install -r requirements.txt

# quick test
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"
```

### B) Data‑Generation environment (JAX‑CFD)
```bash
# from repo root
python -m venv .venv_jax
# Linux/macOS
source .venv_jax/bin/activate
# Windows (PowerShell)
# .venv_jax\Scripts\Activate.ps1

pip install -r src/datagen/requirements_jaxcfd.txt

# quick test
python -c "import jax; print('jax', jax.__version__)"
```

> Keep the environments **separate**. Do not install JAX into `.venv_torch`, nor PyTorch into `.venv_jax`.

---

## Configure Local Paths (One-Time)
Some training scripts expect **absolute paths** for configs/data/runs. As paths are machine‑specific, use **placeholders** in commands and/or **edit the path constants** in the source files to point to **your** local clone.

- Determine your absolute repo path `<ABS_REPO>`:  
  Linux/macOS → `pwd`; Windows PowerShell → `Get-Location` at repo root.  
  Examples: `/home/alice/CFDStreamSurrogate` or `C:\Users\Alice\CFDStreamSurrogate`.

- If required by your version of the code, update absolute root constants (e.g., in `src/training/train.py`, `src/training/trainer.py`) so that `/scratch/.../StageGitlab/...` becomes `<ABS_REPO>/...` on your machine.

> The commands below use **`<ABS_REPO>`** as a **placeholder**. Replace it with your own absolute path. On Windows, keep the quotes around paths with spaces.

---

## Quick Start (TL;DR)
```bash
# 1) generate data (JAX env)
#    (example: forced turbulence)
python -m src.datagen.generate \
  --config <ABS_REPO>/configs/data/forced/forced.yaml \
  --expe_name forced_seed42

# 2) train offline (PyTorch env)
python -m src.training.train \
  --config <ABS_REPO>/configs/train/offline/offline.yaml \
  --expe_name unet_offline_baseline

# 3) run inference (PyTorch env)
python -m src.inference.infer \
  --config <ABS_REPO>/configs/infer/infer.yaml
```

---

## Data Generation (Spectral / JAX‑CFD)
Activate **`.venv_jax`**. The generator uses the **parent folder** of the config to select **forced** vs **decaying**.

### CLI
```bash
python -m src.datagen.generate \
  --config <ABS_REPO>/configs/data/forced/forced.yaml \
  [--expe_name <NAME>]
```
```bash
python -m src.datagen.generate \
  --config <ABS_REPO>/configs/data/decaying/decaying.yaml \
  [--expe_name <NAME>]
```

**Arguments**
- `--config` (required): YAML under `configs/data/{forced|decaying}/`. The parent folder determines the generator.  
- `--expe_name` (optional): explicit experiment directory name. If omitted, a timestamp‑based name is generated: `gen_YYYYMMDD-HHMMSS_{forced|decaying}_seedXX` (seed comes from the YAML).

**Outputs**
```
<ABS_REPO>/data/ForcedTurbulence/<EXP_NAME>/{vorticity.h5, config.yaml}
<ABS_REPO>/data/DecayingTurbulence/<EXP_NAME>/{vorticity.h5, config.yaml}
```
`vorticity.h5` contains dataset `(T,H,W)` under key `vorticity` with attributes: `mean,std,var,min,max,dt,time,viscosity,inner_steps,outer_steps,final_time,solver_iteration_time,x_min,x_max,y_min,y_max,kind`.

---

## Datasets & Normalization
Module: `src/datasets/VorticityDataset.py`

- Returns pairs `(x_t, Y_t)` with `x_t ∈ ℝ^{1×H×W}` and `Y_t = (x_{t+1},…,x_{t+n}) ∈ ℝ^{n×H×W}`, where `n = nstep`.
- Normalization uses per‑file `mean/std` stored in HDF5 attributes.  
- Concatenation across experiments is supported without cross‑boundary leakage.
- For training, the loader is built with `nstep = max(curr_lr_steps)` to ensure targets exist for the largest horizon; the loop then truncates to the current horizon stage.

---

## Model (UNet)
- Encoder–decoder with four scales and skip connections. Each block: `(Conv3×3 → GroupNorm(1) → ReLU) × 2`.
- `padding_mode` supports `"circular"` to reflect periodic domains.
- **Rollout supervision** is implemented **in the training loop** (the UNet predicts 1 step; multi‑step rollout is constructed iteratively).

---

## Training
Activate **`.venv_torch`**. Launch from repo root with **`python -m`**.

### Offline (Curriculum on Rollout Horizon)
```bash
python -m src.training.train \
  --config <ABS_REPO>/configs/train/offline/offline.yaml \
  [--expe_name <RUN_NAME>]
```
**YAML highlights (typical)**
- `data`: `experiments_dirs` (list of `<ABS_REPO>/data/.../gen_...`), `h5_name`, `key`, `batch_size`, `shuffle`, `num_workers`, `pin_memory`, optional `db_size` limit.  
- `model`: `in_channels`, `num_classes`, `padding_mode`, `padding`.  
- `optim`: e.g., learning rate.  
- `curr_lr_steps`: list of rollout horizons, e.g., `[1,2,4,8]`.  
- `train`: number of `epochs` per stage.

**Curriculum (definition)**  
Given horizons \(\mathcal{H}=\{h_1<h_2<\dots<h_S\}\), training proceeds in **S stages**. At stage \(s\), supervision uses horizon \(h_s\). The DataLoader was built with \(nstep=\max(\mathcal{H})\) so that all targets are available; the loop truncates to the current \(h_s\).

**Outputs**
```
<ABS_REPO>/runs/offline/<RUN_NAME>/
  checkpoints/           # periodic + final .pth
  events/                # TensorBoard
  logs/                  # CSV + loss_epochs.png
  config.yaml            # YAML snapshot
```

### Online (Streaming Emulation)
```bash
python -m src.training.train \
  --config <ABS_REPO>/configs/train/online/online.yaml \
  [--expe_name <RUN_NAME>]
```
**Principle**  
- **Producer thread** pushes time indices into a bounded **ring buffer** (capacity `buffer_capacity`, cadence `producer_dt`).  
- **Consumer (training loop)** pops indices and builds batches on‑the‑fly from the dataset; training proceeds **without epochs**.  
- A **single** horizon is used: `curr_lr_steps: [h]`.  
- Stopping criteria: `target_loss` or `max_rounds`.

**Outputs**
```
<ABS_REPO>/runs/online/<RUN_NAME>/
  checkpoints/
  events/
  logs/                  # CSV + loss_rounds.png
  config.yaml
```

### Rollout Loss — Mathematical Formulation
Let normalized input at time \(t\) be \(x_t\). Let \(f_\theta\) be the UNet that predicts one step ahead. The multi‑step rollout is built in the loop:
\[
\hat{x}_{t+1} = f_\theta(x_t),\quad
\hat{x}_{t+2} = f_\theta(\hat{x}_{t+1}),\ \dots,\
\hat{x}_{t+k} = f_\theta(\hat{x}_{t+k-1}).
\]
For a given horizon \(h\), the per‑sample rollout loss is:
\[
\mathcal{L}_\text{rollout}^{(h)}(t;\theta) \;=\;
\sum_{k=1}^{h} w_k\;\ell\!\big(\hat{x}_{t+k},\,x_{t+k}\big),
\]
where \(\ell\) is a pointwise discrepancy (e.g., MSE) and \(w_k\ge 0\) optional weights (uniform by default). With a windowed physical penalty (e.g., TKE), one may use:
\[
\mathcal{L}^{(h)}(t;\theta)
= \sum_{k=1}^{h} w_k\,\ell_\text{MSE}(\hat{x}_{t+k}, x_{t+k})
\;+\; \lambda\,\Phi\!\big(\hat{x}_{t+1:t+h}, x_{t+1:t+h}\big),
\]
with \(\Phi\) a physical term aggregated over \([t+1,t+h]\) and \(\lambda\ge 0\).

> In the current implementation, the active loss is **MSE**. Additional losses (e.g., TKE/TKEMSE) are available in the codebase but disabled by default.

---

## Inference & Diagnostics
```bash
python -m src.inference.infer \
  --config <ABS_REPO>/configs/infer/infer.yaml
```
**YAML highlights**
- `model`: UNet hyperparameters + `checkpoint` path under `<ABS_REPO>/runs/.../checkpoints/...`  
- `rollout`: `mode: "free"` or `"chunked"`, `seed_index` (start frame), `block_n` for chunked mode  
- `data`: `experiment_dir` (e.g., `<ABS_REPO>/data/.../gen_...`), `h5_name`, `key`

**Outputs**
```
<ABS_REPO>/predictions/<RUN_NAME>/
  config.yaml
  preds/vorticity.h5
  figures/
    tke_timeseries.png
    energy_spectrum.png
    prediction_vs_simulation.gif
```

---

## Artifacts, Logs & Reproducibility
- **Config snapshot**: every run copies its YAML to the run directory.  
- **CSV logs**: per batch/epoch (offline) or per round (online).  
- **TensorBoard**: summaries under `events/` for quick inspection.  
- **Checkpoints**: periodic + final `.pth`.  
- **Figures**: standard plots saved per run under `logs/` (training) or `figures/` (inference).  
- **Determinism**: seeds are configurable in YAML for generation and training.

---

## Troubleshooting
- **ModuleNotFoundError** → Always launch from repo root with `python -m ...` (ensures package imports).  
- **HDF5 slow/hangs** (network filesystem) → set `num_workers: 0` in training YAML.  
- **CUDA not visible (PyTorch)** → check `nvidia-smi`; ensure you installed the correct PyTorch wheel for your CUDA.  
- **JAX/JAX‑CFD install issues** → prefer CPU first; then switch to GPU following JAX official instructions matching your CUDA version.  
- **Absolute path errors** → replace `<ABS_REPO>` placeholders by your real absolute path; if your code version hard‑codes roots, edit those constants to point to `<ABS_REPO>`.

---

## Acknowledgements & Citation
Data generation uses **JAX‑CFD** (spectral Navier–Stokes solver). Please cite:  
- Google Research, “**JAX‑CFD: Computational Fluid Dynamics in JAX**,” GitHub. https://github.com/google/jax-cfd (accessed YYYY‑MM‑DD).

---
<img width="3465" height="2058" alt="losses_grid_forced2" src="https://github.com/user-attachments/assets/ab98b3b3-54db-4ecf-87bd-63d4c27aa7fc" />
<img width="3529" height="2058" alt="losses_grid_decaying" src="https://github.com/user-attachments/assets/23a5e93b-e96b-45f9-99e5-577895a0770b" />
<img width="5635" height="2274" alt="final_grid_tke_64x64vorticity" src="https://github.com/user-attachments/assets/36c0f107-657f-4c3f-860c-22334bc32fc4" />
<img width="4518" height="2274" alt="final_grid_tke_256x256vorticity" src="https://github.com/user-attachments/assets/7418b143-3d7e-45a9-aae2-be2db72efc2f" />
<img width="3937" height="2337" alt="tke_rows_forced_free" src="https://github.com/user-attachments/assets/f874f8d3-458a-4c39-8a4c-82ae5dc0cc2c" />
