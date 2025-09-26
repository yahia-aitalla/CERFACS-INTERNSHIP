from __future__ import annotations

#import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.trainer import BaseTrainer


class OfflineTrainer(BaseTrainer):
    """
    Classic offline training over a DataLoader for a fixed number of epochs.
    - Logs loss per batch ("loss/train_step") and per epoch ("loss/train_epoch") to TensorBoard
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
        train_loader: DataLoader,
        epochs: int,
        strategy: str = "offline",
        start_global_step: int = 0,
    ) -> None:
        super().__init__(desired_steps= desired_steps,
            run_dir=run_dir, device=device, model=model, optimizer=optimizer,
            criterion=criterion, strategy=strategy, start_global_step=start_global_step
        )
        self.train_loader = train_loader
        self.epochs = int(epochs)
    
    def run(self) -> None:
        # curriculum: iterate over the desired horizons
        desired_steps = self.desired_steps 
        
        global_epoch_step = 0

        train_losses = {n: [] for n in desired_steps}

        for n_cur in desired_steps:
            print(f"\n===== Stage: desired_nstep = {n_cur} =====")

            for epoch in tqdm(range(self.epochs)):
                self.model.train()
                train_running_loss = 0.0

                for idx, (img, mask_full) in enumerate(tqdm(self.train_loader, position=0, leave=True)):
                    img = img.float().to(self.device)              # (B, 1, H, W)                                         
                    mask = mask_full[:, :n_cur, :, :].float().to(self.device)              # mask: (B, n_cur, H, W)          #mask_full: (B, max(desired_steps), H, W)

                    self.optimizer.zero_grad()
                    
                    # autoregressive rollout with a plain UNet (channel=1)
                    x = img
                    preds = []
                    for _ in range(n_cur):
                        x = self.model(x)                          # (B, 1, H, W)
                        preds.append(x)
                    y_pred = torch.cat(preds, dim=1)               # (B, n_cur H, W)

                    loss = self.criterion(y_pred, mask)
                    loss.backward()
                    self.optimizer.step()

                    train_running_loss += loss.item()
                    
                    self.metrics.log_batch(
                        stage_n=n_cur,
                        epoch_in_stage=epoch,   # epoch index inside this stage
                        batch_idx=idx,          # batch index inside the epoch
                        global_batch=self.global_step,  # global batch counter
                        loss=loss.item(),
                    )

                    # TensorBoard: loss per batch (global and per-stage)
                    self.log_scalar("train/loss_batch", loss.item())
                    self.log_scalar(f"train_nstep_{n_cur}/loss_batch", loss.item())
                    self.global_step += 1

                train_loss = train_running_loss / (idx + 1)
                train_losses[n_cur].append(train_loss)
                print(f"[nstep={n_cur} | Epoch {epoch+1}] Train loss: {train_loss:.6f}")

                self.metrics.log_epoch(
                    stage_n=n_cur,
                    epoch_in_stage=epoch,      # 0..self.epochs-1
                    global_epoch=global_epoch_step,  # continuous across all stages
                    loss=train_loss,
                )

                # TensorBoard: loss per epoch (global and per-stage)
                self.log_scalar("train/loss_epoch", train_loss, global_epoch_step)
                self.log_scalar(f"train_nstep_{n_cur}/loss_epoch", train_loss, global_epoch_step)
                global_epoch_step += 1

                # periodic checkpoint per stage
                if (epoch + 1) % 20 == 0:
                    ckpt_path = self.ckpt_dir / f"ckpt_n{n_cur}_e{epoch+1}.pth"
                    torch.save(self.model.state_dict(), ckpt_path)

        self.metrics.plot_epoch_loss_by_stage(out_name="loss_epochs.png")
        
        # final checkpoint after all stages
        final_ckpt = self.ckpt_dir / "my_checkpoint_final.pth"
        torch.save(self.model.state_dict(), final_ckpt)
        self.close()
        

    