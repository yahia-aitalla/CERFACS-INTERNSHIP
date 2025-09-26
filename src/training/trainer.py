from __future__ import annotations

import os, time, shutil, random
from pathlib import Path
from typing import Any, Dict, Optional
from training.metricslogger import MetricsLogger

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter


RUNS_ROOT = Path("/scratch/algo/aitalla/StageGitlab/runs")


#def set_seed(seed: int) -> None:
#    random.seed(seed)
#    np.random.seed(seed)
#    torch.manual_seed(seed)
#    torch.cuda.manual_seed_all(seed)
#    torch.backends.cudnn.benchmark = True


def prepare_run_dir(strategy: str, seed: int, turb_kind: str, expe_name: Optional[str], cfg_path: Path) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    name = expe_name.strip().replace(" ", "_") if expe_name else f"train_{ts}_{strategy}_{turb_kind}_seed{seed}"
    strategy_to_subdir = {
        "offline": "offline",
        "online":   "online",  
        }
    run_dir = RUNS_ROOT / strategy_to_subdir[strategy] / name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "events").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    # Copy config
    shutil.copyfile(cfg_path, run_dir / "config.yaml")
    return run_dir




class BaseTrainer:
    """
    Shared utilities for all trainers (offline/online)
    """
    def __init__(
        self,
        *,
        desired_steps:list,
        run_dir: Path,
        device: torch.device,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        strategy: str,
        start_global_step: int = 0,
    ) -> None:
        self.run_dir = run_dir
        self.events_dir = run_dir / "events"
        self.ckpt_dir = run_dir / "checkpoints"
        self.logs_dir = run_dir / "logs"
        self.desired_steps= desired_steps
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.strategy = strategy
        self.metrics = MetricsLogger(self.logs_dir)
        self.global_step = start_global_step
        self.best_metric = float("inf")
        self.writer = SummaryWriter(log_dir=os.fspath(self.events_dir))

    def log_scalar(self, tag: str, value: float, step: Optional[int] = None) -> None:
        self.writer.add_scalar(tag, value, self.global_step if step is None else step)

    

    #def _cuda_timers(self):
    #    """
    #    Return CUDA events (start,end) for fwd/bwd/opt if device is CUDA,
    #    otherwise return CPU no-op timers with the same interface.
    #    """
    #    if self.device.type != "cuda":
    #        class CPU:
    #            def record(self): pass
    #            def elapsed_time(self, other): return 0.0
    #        return CPU(), CPU(), CPU(), CPU(), CPU(), CPU()
    #    s1 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    #    s2 = torch.cuda.Event(enable_timing=True); e2 = torch.cuda.Event(enable_timing=True)
    #    s3 = torch.cuda.Event(enable_timing=True); e3 = torch.cuda.Event(enable_timing=True)
    #   return s1, e1, s2, e2, s3, e3

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()

    def run(self) -> None:
        raise NotImplementedError
