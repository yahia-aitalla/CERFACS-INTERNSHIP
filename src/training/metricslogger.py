
import csv
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import defaultdict


class MetricsLogger:
    """CSV metrics + plotting helper shared by all trainers."""
    def __init__(self, logs_dir: Path):
        self.logs_dir = logs_dir
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.batch_csv = self.logs_dir / "metrics_batch.csv"
        self.epoch_csv = self.logs_dir / "metrics_epoch.csv"

        # Create CSVs with headers if missing
        if not self.batch_csv.exists():
            with open(self.batch_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["global_batch", "stage_n", "epoch_in_stage", "batch_idx", "loss_batch"])

        if not self.epoch_csv.exists():
            with open(self.epoch_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["global_epoch", "stage_n", "epoch_in_stage", "loss_epoch"])

    def log_batch(self, *, stage_n: int, epoch_in_stage: int, batch_idx: int,
                  global_batch: int, loss: float) -> None:
        with open(self.batch_csv, "a", newline="") as f:
            csv.writer(f).writerow([global_batch, stage_n, epoch_in_stage, batch_idx, float(loss)])

    def log_epoch(self, *, stage_n: int, epoch_in_stage: int,
                  global_epoch: int, loss: float) -> None:
        with open(self.epoch_csv, "a", newline="") as f:
            csv.writer(f).writerow([global_epoch, stage_n, epoch_in_stage, float(loss)])

    def plot_epoch_loss_by_stage(self, out_name: str = "loss_epochs.png") -> Path:
        """
        Plot a single continuous curve (x = global_epoch), coloring each curriculum
        stage (stage_n) differently. Saves into logs_dir/out_name.
        """
        xs, ys, ss = [], [], []
        with open(self.epoch_csv, "r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                xs.append(int(row["global_epoch"]))
                ys.append(float(row["loss_epoch"]))
                ss.append(int(row["stage_n"]))

        if not xs:
            return self.logs_dir / out_name  # nothing to plot

        # Matplotlib style (your preferred style)
        mpl.rcParams.update({
            "figure.figsize": (5.5, 3.2),
            "figure.dpi": 300,
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        })

        # Split continuous sequence by stage, but keep global_epoch for continuity
        by_stage = defaultdict(lambda: {"x": [], "y": []})
        for x, y, s in zip(xs, ys, ss):
            by_stage[s]["x"].append(x)
            by_stage[s]["y"].append(y)

        # Choose distinct colors per stage
        # (tab10 handles up to 10; extend if you have more)
        color_cycle = plt.get_cmap("tab10").colors

        fig, ax = plt.subplots()
        for i, s in enumerate(sorted(by_stage.keys())):
            color = color_cycle[i % len(color_cycle)]
            ax.plot(by_stage[s]["x"], by_stage[s]["y"],
                    color=color, linewidth=0.8, label=f"n={s}")

        ax.set_xlabel("epochs (global)")
        ax.set_ylabel("loss")
        ax.set_title("Training loss per epoch (curriculum stages colored)")
        ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.5)
        ax.minorticks_on()
        ax.legend(loc="best")

        out_path = self.logs_dir / out_name
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path
