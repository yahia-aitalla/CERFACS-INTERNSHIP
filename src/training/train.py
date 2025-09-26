import argparse
import os
from pathlib import Path
from typing import Tuple
import torch
import yaml

from datasets.VorticityDataset import make_dataloader, build_dataset
from torch.utils.data import ConcatDataset
import torch.nn as nn
from models.unet import UNet
#from losses import get_loss
from training.trainer import prepare_run_dir
from training.offline import OfflineTrainer
from training.online import OnlineTrainer

DEFAULTS = {
    "offline": "/scratch/algo/aitalla/StageGitlab/configs/train/offline/offline.yaml",
    "online":  "/scratch/algo/aitalla/StageGitlab/configs/train/online/online.yaml",
}

OFFLINE_ROOT = Path("/scratch/algo/aitalla/StageGitlab/configs/train/offline")
ONLINE_ROOT  = Path("/scratch/algo/aitalla/StageGitlab/configs/train/online")


def read_yaml(path: str | os.PathLike) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def infer_domain_nstep(ds) -> Tuple[tuple, str]:
    """
    Return (domain, nstep) from a dataset or a ConcatDataset.
    - domain: (x_min, x_max, y_min, y_max)
    - nstep : int
    """
    # If it's a ConcatDataset, check that all sub-datasets agree
    if isinstance(ds, ConcatDataset):
        domains = {tuple(getattr(d, "domain", None)) for d in ds.datasets}
        kinds  = {getattr(d, "kind",  None) for d in ds.datasets}
        # All sub-datasets must expose domain/nstep and be identical
        if None in domains or len(domains) != 1:
            raise ValueError(f"Inconsistent or missing 'domain' across sub-datasets: {domains}")
        if None in kinds or len(kinds) != 1:
            raise ValueError(f"Inconsistent or missing 'kind' across sub-datasets: {kinds}")
        
        dom = domains.pop()
        if len(dom) != 4:
            raise ValueError(f"'domain' must be a 4-tuple (x_min,x_max,y_min,y_max), got: {dom}")
        return dom, str(kinds.pop())

    # Single dataset: must expose both attributes
    if not hasattr(ds, "domain") or not hasattr(ds, "kind"):
        raise AttributeError("Dataset must expose both 'domain' and 'kind'.")

    dom = tuple(getattr(ds, "domain"))
    if len(dom) != 4:
        raise ValueError(f"'domain' must be a 4-tuple (x_min,x_max,y_min,y_max), got: {dom}")
    return dom, str(getattr(ds, "kind"))

def parse_args():
    ap = argparse.ArgumentParser(
        description="Train UNet (offline | online) with YAML config (1 YAML per strategy). "
                    "La stratégie est déduite du dossier parent du YAML : "
                    ".../configs/train/offline/ → offline, .../configs/train/online/ → online."
    )
    ap.add_argument(
        "--config",
        type=str,
        default=None,
        help=("Chemin vers le YAML d'entraînement. "
              f"Si omis, on utilise par défaut: {DEFAULTS['offline']}")
    )
    ap.add_argument(
        "--expe_name",
        type=str,
        default=None,
        help="Nom explicite du dossier d'expérience sous runs/ (sinon nom auto)."
    )
    ap.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Chemin vers un checkpoint à reprendre (ex: runs/.../checkpoints/last.pt)."
    )
    return ap.parse_args()


def main():
    args = parse_args()

    cfg_path = Path(args.config or DEFAULTS["offline"])
    if not cfg_path.is_file():
        raise FileNotFoundError(f"YAML introuvable: {cfg_path}")

    if cfg_path.is_relative_to(OFFLINE_ROOT):
        strategy = "offline"
    elif cfg_path.is_relative_to(ONLINE_ROOT):
        strategy = "online"
    else:
        raise ValueError(
            f"Impossible d'inférer la stratégie: le YAML doit se trouver sous:\n"
            f" - {OFFLINE_ROOT}\n"
            f" - {ONLINE_ROOT}\n"
            f"Reçu: {cfg_path}"
        )

    cfg = read_yaml(cfg_path)

    #seed = int(cfg.get("seed", 42))
    #set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    data_cfg = cfg.get("data", {})


    curr_lr_steps = cfg.get("curr_lr_steps", [1,2,4,8])
    if isinstance(curr_lr_steps, int):
        curr_lr_steps = [curr_lr_steps]
    
    maxstep = max(curr_lr_steps)

    if strategy == "offline":
        dl = make_dataloader(
            experiments_dirs=data_cfg.get("experiments_dirs"),
            h5_name=data_cfg.get("h5_name","vorticity.h5"),
            key=data_cfg.get("key", "vorticity"),
            nstep=maxstep,
            db_size=data_cfg.get("db_size"),
            batch_size=int(data_cfg.get("batch_size", 8)),
            shuffle=bool(data_cfg.get("shuffle", True)),
            num_workers=int(data_cfg.get("num_workers", 0)),
            pin_memory=bool(data_cfg.get("pin_memory", False)),
            dtype="float32",
        )
        ds = dl.dataset
    else:
        ds = build_dataset(
            experiments_dirs=data_cfg.get("experiments_dirs"), 
            h5_name=data_cfg.get("h5_name", "vorticity.h5"),
            key=data_cfg.get("key", "vorticity"),
            nstep=maxstep,
            db_size=data_cfg.get("db_size"),
            dtype="float32",
        )
        dl = None

    domain, turb_kind = infer_domain_nstep(ds)

    model_cfg = cfg.get("model", {})
    model = UNet(
        in_channels=int(model_cfg.get("in_channels", 1)),
        num_classes=int(model_cfg.get("num_classes", 1)),
        padding_mode=str(model_cfg.get("padding_mode", "zeros")),
        padding=int(model_cfg.get("padding", 1)),
    ).to(device)

    optim_cfg = cfg.get("optim", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optim_cfg.get("lr", 0.0003)),
    )

    #loss_name = (cfg.get("loss", {}) or {}).get("name", "mse")
    #criterion = get_loss(
    #    loss_name,
    #    device=device,
    #    domain=domain if loss_name in ("tke", "tkemse") else None,
    #)
    criterion = nn.MSELoss()
    run_dir = prepare_run_dir(strategy, 0, turb_kind, args.expe_name, cfg_path)
    #if args.resume:
    #    ckpt = torch.load(args.resume, map_location="cpu")
    #    if "model" in ckpt:
    #        model.load_state_dict(ckpt["model"])
    #    if "optimizer" in ckpt:
    #        optimizer.load_state_dict(ckpt["optimizer"])
    #    print(f"[resume] Loaded checkpoint from {args.resume}")


    if strategy == "offline":
        epochs = int((cfg.get("train", {}) or {}).get("epochs", 300)) 
        trainer = OfflineTrainer(
            desired_steps=curr_lr_steps,
            run_dir=run_dir, device=device, model=model,
            optimizer=optimizer, criterion=criterion,
            train_loader=dl, epochs=epochs, strategy="offline", 
        )
        trainer.run()
    else:
        buffer_capacity  = int(cfg.get("buffer_capacity", 2016))
        producer_dt      = float(cfg.get("producer_dt", 1.0e-5))
        target_loss      = float(cfg.get("target_loss", 0.005))
        sample_threshold = cfg.get("sample_threshold", None)
        max_rounds       = int(cfg.get("max_rounds", 250))

        trainer = OnlineTrainer(
            desired_steps=curr_lr_steps,
            run_dir=run_dir, device=device, model=model,
            optimizer=optimizer, criterion=criterion,
            dataset=ds,
            batch_size=int(data_cfg.get("batch_size", 8)),
            buffer_capacity=buffer_capacity,
            producer_dt=producer_dt,
            target_loss=target_loss,
            sample_threshold=sample_threshold,
            max_rounds=max_rounds,
            strategy="online",
        )
        trainer.run()


if __name__ == "__main__":
    main()
