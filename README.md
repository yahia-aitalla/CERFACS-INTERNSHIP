# Surrogate modeling for 2D turbulence (forced & decaying turbulence) with offline training and an online‑learning emulation.

This repository delivers an end‑to‑end stack to (i) **generate** 2D turbulence datasets with a **spectral Navier–Stokes solver**, (ii) **train** a **UNet** surrogate with **multi‑step rollout supervision** (curriculum on the time horizon), (iii) **emulate** **online/streaming training**, and (iv) run **inference & diagnostics** (TKE, isotropic energy spectrum, GIFs). 

> **CFD solver:** data generation uses **JAX‑CFD** (spectral method). Please acknowledge: Google Research, “**JAX‑CFD: Computational Fluid Dynamics in JAX**,” GitHub, https://github.com/google/jax-cfd (accessed 2025-10-11).

---

## 1) Why this project (scientific motivation)

### 1.1 Surrogate models
A surrogate $\,\mathcal{S}_\theta\,$ approximates the input–output map of a CFD solver $\,\mathcal{F}\,$, delivering predictions **orders of magnitude faster** than direct numerical simulation while targeting application-level fidelity (design sweeps, real-time control, hybrid solver coupling). In modern ML-CFD, neural surrogates (e.g., UNet/CNN variants) complement reduced-order and kernel methods to accelerate end-to-end workflows.



### 1.2 Why (near-)online learning in CFD
High-fidelity simulations produce **long spatiotemporal trajectories** at **high data rates**; persisting complete snapshot corpora creates a **storage/I/O bottleneck**. In response, HPC practice couples simulation and learning **in situ / online** so that models **consume data as they are produced**, reducing disk pressure and matching deployment scenarios where adaptation on streams is required (producer/consumer data paths, minimal I/O).



### 1.3 Why decaying turbulence is hard
Under sustained forcing, isotropic turbulence can be **statistically stationary**, enabling stable sampling for training/validation. In contrast, **freely decaying** turbulence is **non-stationary**: kinetic energy and spectral content **evolve over time**, so even a single trajectory exhibits **distribution shift**. In streaming/continual settings, such shift amplifies **catastrophic forgetting** of early-time regimes. Consequently, decaying turbulence is a stringent benchmark for online/continual learning, whereas the forced case is comparatively easier.


---



## 2) What you get (features)
- **Spectral data generation** (JAX‑CFD) → HDF5 with comprehensive physical metadata.
- **Dataset API** with **on‑the‑fly normalization** (HDF5 stats) and **multi‑step targets** for rollout supervision.
- **UNet** surrogate; **rollout is implemented inside the training loop** (no external autoregressive wrapper).
- **Offline curriculum** on rollout horizon (e.g., 1 → 2 → 4 → 8) to stabilize optimization on longer horizons.
- **Online emulation** (bounded producer/consumer ring buffer) with **no epochs**, stopping by **target loss** or **max rounds**.
- **Inference** (free or chunked rollout) plus **TKE** and **isotropic energy spectrum** plots and **GIFs**.
- **YAML‑driven configuration**, per‑run config snapshot, **CSV logs**, **TensorBoard events**, **checkpoints**, and **figures**.

---

## 3) Repository layout (complete)

> **Legend:** 📁 directory · 📄 file  
> **Execution policy:** run commands **from `src/`** and always use `python -m <package.module>` so imports resolve reliably.

```text
📁 configs/
│  ├─ 📁 data/
│  │   ├─ 📁 decaying/            # JAX-CFD data-generation YAMLs (decaying turbulence)
│  │   └─ 📁 forced/              # JAX-CFD data-generation YAMLs (forced turbulence)
│  ├─ 📁 train/
│  │   ├─ 📁 offline/             # training YAMLs (offline curriculum, DataLoader setup)
│  │   └─ 📁 online/              # training YAMLs (online emulation: buffer, target_loss, …)
│  └─ 📁 infer/
│      └─ 📄 infer.yaml           # inference YAML (checkpoint, rollout mode, data source)
│
📁 data/
│  ├─ 📁 DecayingTurbulence/      # generated experiments (.h5 + config.yaml per experiment)
│  └─ 📁 ForcedTurbulence/        # generated experiments (.h5 + config.yaml per experiment)
│
📁 predictions/
│  └─ 📁 <RUN_NAME>/              # config.yaml, preds/vorticity.h5, figures/*.png, *.gif
│
📁 runs/
│  ├─ 📁 offline/
│  │   └─ 📁 <RUN_NAME>/          # checkpoints/, events/ (TensorBoard), logs/, config.yaml
│  └─ 📁 online/
│      └─ 📁 <RUN_NAME>/          # same structure as offline
│
📁 src/
│  ├─ 📁 datagen/
│  │   ├─ 📄 __init__.py
│  │   ├─ 📄 generate.py          # CLI for JAX-CFD generation
│  │   ├─ 📄 generators.py        # forced/decaying generators (spectral solver glue)
│  │   └─ 📄 src/datagen/requirements_jaxcfd.txt   # JAX/JAX-CFD env (generation)
│  ├─ 📁 datasets/
│  │   ├─ 📄 __init__.py
│  │   └─ 📄 VorticityDataset.py  # vorticity dataset + DataLoader helpers
│  │
│  ├─ 📁 inference/
│  │   ├─ 📄 __init__.py
│  │   ├─ 📄 core.py              # rollout engine (free/chunked)
│  │   ├─ 📄 infer.py             # CLI for inference
│  │   ├─ 📄 metrics.py           # TKE, isotropic spectrum, etc.
│  │   └─ 📄 plotting.py          # figures (TKE/spectrum/GIF)
│  │
│  ├─ 📁 models/
│  │   ├─ 📄 __init__.py
│  │   └─ 📄 unet.py              # 2D UNet (Conv, GroupNorm(1), ReLU), circular padding
│  │
│  └─ 📁 training/
│      ├─ 📄 __init__.py
│      ├─ 📄 losses.py            # MSE (active), TKE/TKEMSE (available, disabled by default)
│      ├─ 📄 metricslogger.py     # CSV + training loss figures
│      ├─ 📄 offline.py           # offline loop (curriculum), in-loop rollout
│      ├─ 📄 online.py            # online loop (index buffer), single horizon
│      ├─ 📄 trainer.py           # run directory creation, TB writer, artifact layout
│      └─ 📄 train.py             # CLI entry, strategy dispatch (offline/online)
│
📄 README.md
📄 requirements.txt                # PyTorch env (training/inference)

```

---

## 4) System requirements & environments

- **Python ≥ 3.10**. GPU recommended for training (PyTorch). JAX‑CFD supports CPU or GPU.

**Two separate virtual environments are strongly recommended** to avoid CUDA/cuDNN conflicts:

**A) Training / inference (PyTorch)**
```bash
# from repo root
python -m venv .venv_torch
# Linux/macOS
source .venv_torch/bin/activate
# Windows PowerShell
# .venv_torch\Scripts\Activate.ps1
pip install -r requirements.txt
```

**B) Data generation (JAX‑CFD)**
```bash
# from repo root
python -m venv .venv_jax
# Linux/macOS
source .venv_jax/bin/activate
# Windows PowerShell
# .venv_jax\Scripts\Activate.ps1
pip install -r src/datagen/requirements_jaxcfd.txt
```

> Do **not** mix stacks: keep **PyTorch** in `.venv_torch` and **JAX‑CFD** in `.venv_jax`.

---

## 5) Paths & placeholders

- `<ABS_REPO>` denotes the **absolute path to your clone**.  
  (Linux/macOS: run `pwd` at the repo root. PowerShell: `Get-Location`.)

Some scripts accept `--config` (optional). If omitted, an **internal default path** (often absolute) is used. Two options:
1) **Recommended:** pass `--config` with **your** path (shown below using `<ABS_REPO>`).  
2) **Alternative:** edit the default paths **inside the scripts** to point to your `<ABS_REPO>`.

> **All commands below are launched from `src/`.**

---

## 6) Data generation (spectral / JAX-CFD) — run from `src/` with **.venv_jax**

> **Activate the JAX-CFD environment first (installed in §4.B)**
>
> - **Linux/macOS**
>   ```bash
>   source <ABS_REPO>/.venv_jax/bin/activate
>   ```
> - **Windows (PowerShell)**
>   ```powershell
>   <ABS_REPO>\.venv_jax\Scripts\Activate.ps1
>   ```

> **Run from `src/` using the module entrypoint**
> ```bash
> cd <ABS_REPO>/src
> python -m datagen.generate [--config <PATH_TO_YAML>] [--expe_name <EXPERIMENT_NAME>]
> ```

### Behavior and parameters
- `--config` (optional)  
  - **Omitted:** the script uses its **internal default YAML for the *forced* case** and launches the **forced-turbulence generator**.  
  - **Provided:** the **turbulence type is inferred from the YAML’s parent folder**:  
    - YAML under `.../configs/data/forced/` → runs the **forced** generator  
    - YAML under `.../configs/data/decaying/` → runs the **decaying** generator
- `--expe_name` (optional)  
  - If omitted, an experiment name is **auto-generated**:  
    `gen_YYYYMMDD-HHMMSS_{forced|decaying}_seed<seed>`.

### Minimal examples
**A) Default (no `--config` → forced generator with the internal default YAML)**
```bash
source <ABS_REPO>/.venv_jax/bin/activate
cd <ABS_REPO>/src
python -m datagen.generate
```

---

## 7) Dataset

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
### 7.1 HDF5 assumptions (checked at runtime)
- The vorticity array is stored under key `vorticity` (configurable with `key`).
- Attributes required on that dataset (written by the generator):  
  `mean, std, var, min, max, solver_iteration_time, x_min, x_max, y_min, y_max, kind`.
- Normalization uses **only** these attributes; `std` is safely floored internally to avoid division by zero.

### 7.2 Single simulation **or** concatenation of multiple simulations
- **Single directory** → a single `VorticityDataset` over that experiment.
- **Multiple directories** → the builder returns a **`ConcatDataset`** that concatenates several `VorticityDataset` instances end-to-end; batching then interleaves samples across simulations transparently (same `nstep`, `h5_name`, `key` across all).

### 7.3 Builder and DataLoader (what the training loop uses)
- `build_dataset(experiments_dirs, *, h5_name="vorticity.h5", key="vorticity", nstep, db_size, dtype)`  
  Returns a `VorticityDataset` (one dir) or a `ConcatDataset` (many dirs).
- `make_dataloader(..., batch_size, shuffle, num_workers, pin_memory, ...)`  
  Wraps the dataset in a PyTorch `DataLoader`.

> **Note:** On network filesystems, prefer `num_workers: 0` to avoid HDF5 I/O contention.

### 7.4 Practical guidance for training
- **Offline curriculum:** construct the loader with `nstep = max(curr_lr_steps)` so that the largest target window is available; the training loop **truncates** to the active horizon at each stage.
- **Data types:** you can pass `dtype="float32"` (default) or a `torch.dtype`. Inputs/targets are converted accordingly.
- **Domain metadata:** `x_min, x_max, y_min, y_max` are read from attributes and exposed via dataset properties for downstream diagnostics (e.g., spectra).

**Summary of shapes per batch** (batch size $B$):
$$
\text{inputs: } X \in \mathbb{R}^{B \times 1 \times H \times W}, \qquad
\text{targets: } Y \in \mathbb{R}^{B \times n \times H \times W}.
$$


---

## 8) Model (U-Net)

A 2D UNet (four scales) with skip connections; each block is `(Conv3×3 → GroupNorm(1) → ReLU) × 2`. `padding_mode="circular"` is available for periodic domains.  
**Important:** the model is **one‑step**; **multi‑step rollout is constructed inside the training loop**.

---

## 9) Why supervised **rollout** (multi‑step) instead of **one‑step** training?

One-step training optimizes $x_t \mapsto x_{t+1}$, but inference **feeds back** predictions, creating a train–inference mismatch (exposure bias) and compounding error. Rollout supervision explicitly **mimics inference during training** by iterating the learned one-step map $f_\theta$ across a horizon $h$:

$$
\hat{x}_{t+1} = f_\theta(x_t), \qquad
\hat{x}_{t+k} = f_\theta\!\big(\hat{x}_{t+k-1}\big)\quad\text{for } 2 \le k \le h,
$$

and comparing the predicted stack
$$
\{\hat{x}_{t+1},\ldots,\hat{x}_{t+h}\}
$$
to ground truth over the same window. This reduces the train–inference gap and improves long-horizon stability—especially important on **non-stationary** trajectories such as decaying turbulence.

---

## 10) Mathematical formulation 

### 10.1 Notation
- $x_t \in \mathbb{R}^{H \times W}$: **normalized** vorticity at time $t$.
- $f_\theta : \mathbb{R}^{1 \times H \times W} \rightarrow \mathbb{R}^{1 \times H \times W}$: one-step UNet.
- **Rollout** for horizon $h \in \mathbb{N}^+$:

  $$
  \hat{x}_{t+1} = f_\theta(x_t),\quad
  \hat{x}_{t+2} = f_\theta(\hat{x}_{t+1}),\ \ldots,\
  \hat{x}_{t+k} = f_\theta(\hat{x}_{t+k-1}) \quad (1 \le k \le h).
  $$
- Stack predictions and targets:
  $$
  \hat{Y}_t^{(h)} = [\hat{x}_{t+1}, \ldots, \hat{x}_{t+h}] \in \mathbb{R}^{h \times H \times W},\quad
  Y_t^{(h)} = [x_{t+1}, \ldots, x_{t+h}] \in \mathbb{R}^{h \times H \times W}.
  $$

### 10.2 Active loss (unweighted MSE; averaged over batch, time, and pixels)
For batch size $B$, the code computes the MSE **between stacked tensors** $\hat{Y}_t^{(h)}$ and $Y_t^{(h)}$:
$$
\mathcal{L}_{\text{MSE}}^{(h)}(\theta)
= \frac{1}{B\,h\,H\,W} \sum_{b=1}^{B} \sum_{k=1}^{h} \sum_{i=1}^{H} \sum_{j=1}^{W}
\Big( \hat{x}^{(b)}_{t+k}[i,j] - x^{(b)}_{t+k}[i,j] \Big)^2.
$$

> Physical terms (TKE/TKEMSE) exist in `losses.py` but are **disabled by default** in the current CLI.

### 10.3 Curriculum learning
With horizons $\mathcal{H} = \{\, h_1 < h_2 < \cdots < h_S \,\}$, training proceeds in **stages** $s = 1,\dots,S$. At stage $s$, the loop rolls out to $h_s$ and minimizes $\mathcal{L}_{\mathrm{MSE}}^{(h_s)}$. The DataLoader is built with $n = \max(\mathcal{H})$ to ensure all targets exist; the loop then **truncates** to $h_s$ at each stage.

---

## 11) Training — **from `src/`** (PyTorch env)

### 11.1 Offline 
```bash
cd <ABS_REPO>/src
python -m training.train \
  [--config <ABS_REPO>/configs/train/offline/offline.yaml] \
  [--expe_name <RUN_NAME>]
```
**Scenarios**
- **With `--config`**: strategy = **offline** (path under `configs/train/offline/`).  
- **Without `--config`**: an **internal default** (offline training configuration) is used and launches the **offline trainer**

**Typical YAML fields**
- `data`: `experiments_dirs` (list of `<ABS_REPO>/data/.../gen_...`), `h5_name`, `key`, `batch_size`, `shuffle`, `num_workers`, `pin_memory`, optional `db_size`.  
- `model`: `in_channels`, `num_classes`, `padding_mode`, `padding`.  
- `optim`: learning rate, etc.  
- `curr_lr_steps`: horizons, e.g., `[1,2,4,8]`.  
- `train`: `epochs` per stage.

**Outputs**
```
<ABS_REPO>/runs/offline/<RUN_NAME>/
  checkpoints/  events/  logs/  config.yaml
```

### 11.2 Online (streaming emulation)
```bash
cd <ABS_REPO>/src
python -m training.train \
  --config <ABS_REPO>/configs/train/online/online.yaml \
  [--expe_name <RUN_NAME>]
```
**Principle**
- **Producer** thread pushes time indices into a bounded **ring buffer** (`producer_dt`, `buffer_capacity`).  
- **Consumer** builds batches on the fly; a **single horizon** is used (first element of `curr_lr_steps`, e.g., `[4]`).  
- Stop at `target_loss` or `max_rounds`.

**Outputs**
```
<ABS_REPO>/runs/online/<RUN_NAME>/
  checkpoints/  events/  logs/  config.yaml
```

---

## 12) Inference — **from `src/`** (PyTorch env)
```bash
cd <ABS_REPO>/src
python -m inference.infer \
  [--config <ABS_REPO>/configs/infer/infer.yaml]
```
**Scenarios**
- **With `--config`**: YAML specifies UNet hyperparameters and `checkpoint` (`<ABS_REPO>/runs/.../checkpoints/...`), rollout mode (`free`/`chunked`), `seed_index`, `block_n` (chunked), and data (`experiment_dir`, `h5_name`, `key`).  
- **Without `--config`**: the script falls back to its internal default; adapt it if necessary.

**Outputs**
```
<ABS_REPO>/predictions/<RUN_NAME>/
  config.yaml
  preds/vorticity.h5
  figures/{tke_timeseries.png, energy_spectrum.png, prediction_vs_simulation.gif}
```

---

## 13) Visualizing training with **TensorBoard**
Events are written under `runs/<strategy>/<RUN_NAME>/events/`.

**Launch**
```bash
# from any terminal
tensorboard --logdir <ABS_REPO>/runs/<strategy>/<RUN_NAME>/events --port 6006
# open http://localhost:6006
```

---

## 14) Reproducibility & artifacts
- **Config snapshot** per run (the exact YAML used).  
- **CSV logs**: per batch/epoch (offline) or per round (online).  
- **TensorBoard**: scalar summaries by batch/epoch/round.  
- **Checkpoints**: periodic + final.  
- **Figures**: loss curves (training) and TKE/spectrum/GIFs (inference).  
- **Seeding**: configurable in data generation and training YAMLs.


---
<img width="3465" height="2058" alt="losses_grid_forced2" src="https://github.com/user-attachments/assets/ab98b3b3-54db-4ecf-87bd-63d4c27aa7fc" />
<img width="3529" height="2058" alt="losses_grid_decaying" src="https://github.com/user-attachments/assets/23a5e93b-e96b-45f9-99e5-577895a0770b" />
<img width="5635" height="2274" alt="final_grid_tke_64x64vorticity" src="https://github.com/user-attachments/assets/36c0f107-657f-4c3f-860c-22334bc32fc4" />
<img width="4518" height="2274" alt="final_grid_tke_256x256vorticity" src="https://github.com/user-attachments/assets/7418b143-3d7e-45a9-aae2-be2db72efc2f" />
<img width="3937" height="2337" alt="tke_rows_forced_free" src="https://github.com/user-attachments/assets/f874f8d3-458a-4c39-8a4c-82ae5dc0cc2c" />
