## 7) Dataset (vorticity sequences; single or concatenated simulations)

`VorticityDataset` yields **normalized** vorticity frame–sequences for **multi-step rollout supervision**. It reads an HDF5 file produced by the generator, pulls normalization stats from HDF5 **attributes**, and returns samples with:

- **Input** $x_t$ — shape `(1, H, W)`: one normalized vorticity frame at time $t$  
- **Target** $Y_t^{(n)}$ — shape `(n, H, W)`: the next $n$ normalized frames $[x_{t+1}, \ldots, x_{t+n}]$

**Normalization.** With attributes `mean` and `std` (denoted $\mu$ and $\sigma$), each frame is normalized as

$$
\tilde{x}_t = \frac{x_t - \mu}{\sigma}, \qquad
\tilde{Y}_t^{(n)} = [\,\tilde{x}_{t+1}, \ldots, \tilde{x}_{t+n}\,].
$$

**Batch shapes** (batch size $B$):

$$
\tilde{X} \in \mathbb{R}^{B \times 1 \times H \times W}, \qquad
\tilde{Y} \in \mathbb{R}^{B \times n \times H \times W}.
$$

### Single vs. multiple simulations
- **Single experiment directory** → one `VorticityDataset`.
- **Multiple experiment directories** → the builder returns a **`ConcatDataset`** that concatenates several `VorticityDataset` instances end-to-end; batching then interleaves samples across simulations (same `nstep`, `h5_name`, `key` across all).

### HDF5 layout and required attributes
- Dataset key (default): `vorticity` with shape `(T, H, W)`.
- Required attributes written by the generator:  
  `mean`, `std`, `var`, `min`, `max`, `dt`, `time`, `viscosity`,  
  `inner_steps`, `outer_steps`, `final_time`, `solver_iteration_time`,  
  `x_min`, `x_max`, `y_min`, `y_max`, `kind` (`"forced"` or `"decaying"`).
- The standard deviation is safely floored internally to avoid division by zero.

> **Rendering note (for GitHub/Markdown math):**  
> Use `$...$` for inline math and `$$...$$` for display equations.  
> Do **not** wrap math in backticks. Leave a blank line before/after `$$` blocks for reliable rendering.

---

## 1) Motivation (scientific context)

**Surrogate models** approximate the input–output map of a costly simulator to deliver **orders-of-magnitude speedups** while preserving task-relevant accuracy. In CFD, surrogates enable rapid design-space exploration, real‑time inference, and embedding of learned components into hybrid solvers (e.g., closures, subgrid models).

**Why online learning is (almost) necessary in CFD.** High‑fidelity CFD produces **long spatiotemporal trajectories** and **multi‑GB/s I/O**; persisting full offline corpora becomes a **storage and bandwidth bottleneck**. Training “in situ”/online—**consuming data as they are produced**—reduces disk pressure and aligns with deployment scenarios where the model must adapt on streams.

**Why decaying turbulence stresses surrogates.** Forced isotropic turbulence is **statistically stationary** (its statistics do not evolve in time under sustained forcing), while **freely decaying turbulence** is **non‑stationary** (energy, spectra, and multiscale content evolve over time). In streaming training, this non‑stationarity introduces **distribution shift** along the trajectory, and models trained sequentially tend to **forget early‑time regimes** (catastrophic forgetting). Consequently, the decaying case is a stringent testbed for **continual/online** training compared to the forced (stationary) case.

---

## 2) Features
- **Spectral data generation** (via JAX‑CFD) with HDF5 export + physical metadata.
- **Dataset API** with on‑the‑fly **normalization** (from HDF5 attributes) and **multi‑step targets** for rollout supervision.
- **UNet** surrogate; multi‑step **rollout** implemented in the **training loop** (no autoregressive wrapper).
- **Offline curriculum** on the rollout horizon (e.g., 1→2→4→8) to stabilize optimization for longer horizons.
- **Online emulation** (producer/consumer ring buffer) enabling streaming training **without epochs**, with **target‑loss** or **round‑limit** stopping.
- **Inference** (free/chunked) + **TKE** and **isotropic energy spectrum** + GIFs.
- **YAML‑driven** configuration, per‑run config snapshot, CSV logs, TensorBoard events, checkpoints, and figures.

---

## 3) Repository Structure
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

> **Execution policy:** run **from `src/`** using `python -m <package.module>` to ensure imports resolve exactly as intended.

---

## 4) System Requirements
- **Python** ≥ 3.10  
- **GPU** recommended for training (PyTorch). JAX‑CFD supports CPU/GPU.  
- Core libs: PyTorch, NumPy, h5py, PyYAML, Matplotlib, TensorBoard.  
- Generation requires JAX + JAX‑CFD (CPU or GPU build matching your platform).

> HDF5 on network filesystems: prefer `num_workers=0` in DataLoaders to avoid I/O contention.

---

## 5) Two Environments (strictly recommended)
Create **two separate virtualenvs** to avoid CUDA/cuDNN conflicts.

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

### B) Data Generation (JAX‑CFD)
```bash
# from repo root
python -m venv .venv_jax
# Linux/macOS
source .venv_jax/bin/activate
# Windows (PowerShell)
# .venv_jax\Scripts\Activate.ps1

pip install -r src/datagen/requirements_jaxcfd.txt
```

Do **not** install JAX in `.venv_torch`, nor PyTorch in `.venv_jax`.

---

## 6) Local Paths & Placeholders
Some scripts accept `--config` but it is **optional**. If `--config` is **omitted**, an **internal default path** is used; these defaults are **absolute** and machine‑specific. You have two options:
- **(Recommended)** Pass `--config` explicitly with **your** path.  
- **(Alternative)** Edit the hard‑coded defaults in the source to point to **your** clone.

**Placeholder convention** (used below):  
- `<ABS_REPO>` = **absolute path to your cloned repo**.  
  - Linux/macOS: `pwd` at repo root (e.g., `/home/alice/CFDStreamSurrogate`)  
  - PowerShell: `Get-Location` at repo root (e.g., `C:\Users\Alice\CFDStreamSurrogate`)

All commands below are launched **from `src/`**.

---

## 7) Quick Start
```bash
# 1) (JAX env) generate data — forced example
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

## 8) Data Generation (Spectral / JAX‑CFD)
Activate **`.venv_jax`**. The generator infers **forced** vs **decaying** from the **parent folder** of the YAML path.

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
  If omitted, the script uses its **internal default** path; edit it if it does not exist on your machine.
- `--expe_name` (optional): explicit output directory name. If omitted, a timestamped name is auto‑generated (`gen_YYYYMMDD-HHMMSS_{forced|decaying}_seedXX`, seed from YAML).

**Outputs**
```
<ABS_REPO>/data/ForcedTurbulence/<EXP_NAME>/{vorticity.h5, config.yaml}
<ABS_REPO>/data/DecayingTurbulence/<EXP_NAME>/{vorticity.h5, config.yaml}
```
HDF5 includes dataset `(T,H,W)` under key `vorticity` and attributes: `mean,std,var,min,max,dt,time,viscosity,inner_steps,outer_steps,final_time,solver_iteration_time,x_min,x_max,y_min,y_max,kind`.

---

## 9) Datasets & Normalization
`src/datasets/VorticityDataset.py` reads `vorticity.h5`, applies **normalization** from file attributes, and returns samples `(x_t, Y_t)` with `Y_t = (x_{t+1},…,x_{t+n})` and `n = nstep`. Concatenation across experiments is supported. For training, the loader is built with `nstep = max(curr_lr_steps)`; the loop truncates to the current curriculum horizon.

---

## 10) Model (UNet)
UNet encoder–decoder (4 scales) with skip connections. Building block: `(Conv3×3 → GroupNorm(1) → ReLU) × 2`. `padding_mode="circular"` supports periodic domains. **Rollout** is implemented in the **training loop** (UNet predicts 1 step).

---

## 11) Training from `src/`

Activate **`.venv_torch`**.

### 11.1 Offline (Curriculum)
```bash
python -m training.train \
  [--config <ABS_REPO>/configs/train/offline/offline.yaml] \
  [--expe_name <RUN_NAME>]
```
**Scenarios**
- **With `--config`**: strategy = **offline** (path under `configs/train/offline/`); the YAML controls data, model, optimizer, curriculum and epochs.  
- **Without `--config`**: the script uses its internal **offline default** path; edit it if it does not exist on your machine.

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

### 11.2 Online (Streaming Emulation)
```bash
python -m training.train \
  [--config <ABS_REPO>/configs/train/online/online.yaml] \
  [--expe_name <RUN_NAME>]
```
**Scenarios**
- **With `--config`**: strategy = **online** (path under `configs/train/online/`); the YAML controls data + `buffer_capacity`, `producer_dt`, `curr_lr_steps: [h]`, `target_loss`, `max_rounds`.  
- **Without `--config`**: the script uses its internal **online default** path; edit it if it does not exist on your machine.

**Outputs**
```
<ABS_REPO>/runs/online/<RUN_NAME>/
  checkpoints/  events/  logs/  config.yaml
```

### 11.3 Rollout Loss — Mathematical Formulation
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

## 12) Inference from `src/`
```bash
python -m inference.infer \
  [--config <ABS_REPO>/configs/infer/infer.yaml]
```
**Scenarios**
- **With `--config`**: YAML specifies UNet hyperparameters and `checkpoint` (`<ABS_REPO>/runs/.../checkpoints/...`), rollout mode (`free`/`chunked`), `seed_index`, `block_n` (if chunked), and data source (`experiment_dir`, `h5_name`, `key`).  
- **Without `--config`**: the script uses its internal default; edit it if it does not exist on your machine.

**Outputs**
```
<ABS_REPO>/predictions/<RUN_NAME>/
  config.yaml
  preds/vorticity.h5
  figures/ {tke_timeseries.png, energy_spectrum.png, prediction_vs_simulation.gif}
```

---

## 13) Artifacts & Reproducibility
- Per‑run **config snapshot**.
- **CSV logs** (batch/epoch offline; round‑based online).  
- **TensorBoard** under `events/`.  
- **Checkpoints** (periodic + final).  
- **Figures** (training loss curves; inference diagnostics).  
- Seeds configurable in YAML (generation & training).

---

## 14) Troubleshooting
- **Imports** → Always run **from `src/`** with `python -m ...`.  
- **HDF5 I/O** → Set `num_workers: 0` (esp. on NFS).  
- **PyTorch CUDA** → Check `nvidia-smi`; install the correct wheel for your CUDA.  
- **JAX/JAX‑CFD** → Start with CPU wheels; switch to GPU following official JAX instructions for your CUDA version.  
- **Defaults without `--config`** → Edit hard‑coded default paths in the source to point to your `<ABS_REPO>`.


---
<img width="3465" height="2058" alt="losses_grid_forced2" src="https://github.com/user-attachments/assets/ab98b3b3-54db-4ecf-87bd-63d4c27aa7fc" />
<img width="3529" height="2058" alt="losses_grid_decaying" src="https://github.com/user-attachments/assets/23a5e93b-e96b-45f9-99e5-577895a0770b" />
<img width="5635" height="2274" alt="final_grid_tke_64x64vorticity" src="https://github.com/user-attachments/assets/36c0f107-657f-4c3f-860c-22334bc32fc4" />
<img width="4518" height="2274" alt="final_grid_tke_256x256vorticity" src="https://github.com/user-attachments/assets/7418b143-3d7e-45a9-aae2-be2db72efc2f" />
<img width="3937" height="2337" alt="tke_rows_forced_free" src="https://github.com/user-attachments/assets/f874f8d3-458a-4c39-8a4c-82ae5dc0cc2c" />
