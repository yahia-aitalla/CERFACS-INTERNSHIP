import argparse
import os
from datagen.generators import  DecayingGenerator, ForcedGenerator
from pathlib import Path

DEFAULT_DECAY_CFG  = "/scratch/algo/aitalla/StageGitlab/configs/data/decaying/decaying.yaml"
DEFAULT_FORCED_CFG = "/scratch/algo/aitalla/StageGitlab/configs/data/forced/forced.yaml"

FORCED_ROOT   = Path("/scratch/algo/aitalla/StageGitlab/configs/data/forced").resolve()
DECAYING_ROOT = Path("/scratch/algo/aitalla/StageGitlab/configs/data/decaying").resolve()

def parse_args():
    ap = argparse.ArgumentParser(
        description="Generate a 2D turbulence HDF5 dataset (decaying | forced) configured via YAML. "
                    "Creates data/(DecayingTurbulence | ForcedTurbulence)/<exp>/ with a copy of the YAML and the HDF5 file."
    )
    
    ap.add_argument(
        "--config",
        type=str,
        default=None,
        help=("Path to the YAML config. If omitted, uses "
              f"{DEFAULT_FORCED_CFG} for forced.")
    )
    ap.add_argument(
        "--expe_name",
        type=str,
        default=None,
        help="Explicit name for the experiment folder (otherwise an automatic name is used)."
    )
    return ap.parse_args()

def main():
    args = parse_args()

    cfg = Path(args.config or DEFAULT_FORCED_CFG).expanduser().resolve()

    print(cfg)
    if not cfg.is_file():
        raise FileNotFoundError(f"YAML not found: {cfg}")

    if cfg.is_relative_to(DECAYING_ROOT):
        print("Decaying turbulence generation has been launched.")
        DecayingGenerator(config_path=os.fspath(cfg), expe_name=args.expe_name).run()
    elif cfg.is_relative_to(FORCED_ROOT):
        print("Forced turbulence generation has been launched.")
        ForcedGenerator(config_path=os.fspath(cfg), expe_name=args.expe_name).run()
    else:
        raise ValueError(
            f"The YAML file must be located under '{DECAYING_ROOT}' ou '{FORCED_ROOT}'. "
            f"Received: {cfg}"
        )

    

if __name__ == "__main__":
    main()
