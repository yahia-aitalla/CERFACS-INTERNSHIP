from __future__ import annotations

import time, threading, random
from typing import List, Optional, Set, Tuple
from pathlib import Path
from time import perf_counter

import torch
from torch import nn

from training.trainer import BaseTrainer


class SharedIndexBuffer:
    """Thread-safe integer ring buffer with capacity limit."""
    def __init__(self, capacity:int):
        self.capacity = int(capacity)
        self.data: List[int] = []
        self.lock = threading.Lock()

    def append(self, idx: int) -> bool:
        with self.lock:
            if len(self.data) >= self.capacity:
                return False
            self.data.append(idx)
            return True

    def size(self, exclude: Optional[Set[int]] = None) -> int:
        with self.lock:
            if not exclude:
                return len(self.data)
            return sum(1 for v in self.data if v not in exclude)

    def sample_snapshot(self, k: int, exclude: Optional[Set[int]] = None) -> Tuple[List[int], List[int]]:
        with self.lock:
            allowed_pos = [i for i, v in enumerate(self.data) if not exclude or v not in exclude]
            if len(allowed_pos) < k:
                return [], []
            pos = random.sample(allowed_pos, k)
            vals = [self.data[i] for i in pos]
        return pos, vals

    def clear_buf(self, exclude: Optional[Set[int]] = None) -> List[int]:
        with self.lock:
            if not exclude:
                out = self.data[:]
                self.data.clear()
                return out
            kept = [v for v in self.data if v not in exclude]
            self.data.clear()
            return kept

    def remove_values(self, values: Set[int]):
        if not values:
            return
        with self.lock:
            s = set(values)
            self.data = [self.data[i] for i in range(len(self.data)) if self.data[i] not in s]


def producer_indices(ds_len: int, buffer: SharedIndexBuffer, dt: float,
                     stop_evt: threading.Event, done_evt: threading.Event):
    print("[producer] start", flush=True)
    i = 0
    while not stop_evt.is_set() and i < ds_len:
        buffer.append(i)
        i += 1
        time.sleep(dt)
    done_evt.set()
    print("[producer] stop", flush=True)


@torch.no_grad()
def keep_sample(model: nn.Module, time_horizon: int, dataset, device: torch.device,
                sample_indices: List[int], seuil: float):
    """
    Return (to_keep_idx, to_drop_idx):
      keep if per-sample loss >= threshold, drop otherwise.
    """
    if not sample_indices:
        return [], []
    xs, ys = [], []
    criterion = nn.MSELoss(reduction="none")
    for idx in sample_indices:
        x, y = dataset[idx]
        xs.append(x); ys.append(y)
    xb = torch.stack(xs, 0).to(device).float()
    yb = torch.stack(ys, 0).to(device).float()
    
    model.eval()
    preds = []
    for _ in range(time_horizon):
        x = model(x)                          # (B, 1, H, W)
        preds.append(x)
    y_pred = torch.cat(preds, dim=1)               # (B, n_cur H, W)
    
    loss_per_pixel = criterion(y_pred, yb)
    loss_per_sample = torch.mean(loss_per_pixel, dim=(-3, -2, -1))
    to_keep, to_drop = [], []
    for j, l in enumerate(loss_per_sample):
        (to_keep if float(l.item()) >= seuil else to_drop).append(sample_indices[j])
    return to_keep, to_drop


class OnlineTrainer(BaseTrainer):
    """
    Continual/streaming training with a threaded producer feeding a buffer.
    Loop continues until avg loss <= target_loss (or max_rounds is reached).
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
        dataset,                   # VorDataset
        batch_size: int,
        buffer_capacity: int,
        producer_dt: float,
        target_loss: float,
        sample_threshold: Optional[float] = None,  # if provided, use keep_sample
        max_rounds: int = 600,
        strategy: str = "online",
        start_global_step: int = 0,
    ) -> None:
        super().__init__(
            desired_steps= desired_steps, run_dir=run_dir, device=device, model=model, optimizer=optimizer,
            criterion=criterion, strategy=strategy, start_global_step=start_global_step
        )
        self.dataset = dataset
        self.bs = int(batch_size)
        self.buffer_capacity = int(buffer_capacity)
        self.producer_dt = float(producer_dt)
        self.target_loss = float(target_loss)
        self.sample_threshold = float(sample_threshold) if sample_threshold is not None else None
        self.max_rounds = int(max_rounds)
        

    def run(self) -> None:
        time_horizon = self.desired_steps[0] 
        buf = SharedIndexBuffer(self.buffer_capacity)
        best_avg = float("inf")
        run_count = 0

        while best_avg > self.target_loss and run_count < self.max_rounds:
            run_count += 1
            round_t0 = perf_counter()
            
            stop = threading.Event()
            done = threading.Event()
            producer = threading.Thread(
                target=producer_indices,
                args=(len(self.dataset), buf, self.producer_dt, stop, done),
                daemon=True,
            )
            producer.start()

            sum_loss = 0.0
            num_batch = 1

            while not done.is_set() or buf.size() > 0:
                print("-"*30)
                print(f"Building batch number : {num_batch}")

                # chrono "batch wall"
                batch_wall_t0 = perf_counter()
                
                batch_idx: List[int] = []
                while len(batch_idx) < self.bs:   
                    print(f"Current batch samples : {batch_idx}")
                    need = self.bs - len(batch_idx)
                    print(f"Sample number needed : {need}")
                    while buf.size(exclude=set(batch_idx)) < need and not done.is_set():
                        #print(f"Waiting for enough samples... New samples available : {buf.size(exclude=set(batch_idx))}")
                        time.sleep(self.producer_dt//1000)
                    
                    # last batch is not full 
                    if  buf.size(exclude=set(batch_idx)) < need and done.is_set():
                        leftover = buf.clear_buf(exclude=set(batch_idx))          
                        if not leftover:
                            break
                        batch_idx.extend(leftover)
                        break      
                    else:
                        pos, cand_idx = buf.sample_snapshot(need, exclude=set(batch_idx))
                        #print(f"Cand Samples : {cand_idx}")
                        if not cand_idx:
                            continue  
                        #to_keep, to_drop = keep_sample(model, DataSet, device, cand_idx, seuil)
                        #print(f"hard samples  : {to_keep}")
                        #print(f"easy samples  : {to_drop}")
                        #buf.remove_values(set(to_drop))
                        #batch_idx.extend(to_keep)
                        batch_idx.extend(cand_idx)

                if len(batch_idx) == 0:
                    break
                    
                buf.remove_values(set(batch_idx))
                print(f"batch number : {num_batch}  : {batch_idx}")
                
                self.model.train()

                xs, ys = [], []
                for idx in batch_idx[:]:
                    x, y = self.dataset[idx]
                    xs.append(x); ys.append(y)

                t_xfer0 = perf_counter()
                xb = torch.stack(xs, 0).float().to(self.device)
                yb = torch.stack(ys, 0).float().to(self.device)
                t_xfer_ms = (perf_counter() - t_xfer0) * 1000.0

                torch.cuda.synchronize()
                fwd_s = torch.cuda.Event(enable_timing=True); fwd_e = torch.cuda.Event(enable_timing=True)
                bwd_s = torch.cuda.Event(enable_timing=True); bwd_e = torch.cuda.Event(enable_timing=True)
                opt_s = torch.cuda.Event(enable_timing=True); opt_e = torch.cuda.Event(enable_timing=True)

                self.optimizer.zero_grad()
                fwd_s.record()
                
                x = xb
                preds = []
                for _ in range(time_horizon):
                    x = self.model(x)                          # (B, 1, H, W)
                    preds.append(x)
                y_pred = torch.cat(preds, dim=1)               # (B, n_cur H, W)
                
                loss = self.criterion(y_pred, yb)  
                
                fwd_e.record()

                bwd_s.record()
                loss.backward()
                bwd_e.record()

                opt_s.record()
                self.optimizer.step()
                opt_e.record()
                torch.cuda.synchronize()

                t_fwd_ms   = fwd_s.elapsed_time(fwd_e)
                t_bwd_ms   = bwd_s.elapsed_time(bwd_e)
                t_opt_ms   = opt_s.elapsed_time(opt_e)
                t_train_ms = fwd_s.elapsed_time(opt_e)

                this_loss = loss.item()


                self.metrics.log_batch(
                    stage_n=time_horizon,           # stage unique (virtuel)
                    epoch_in_stage=run_count - 1,   # rounds ≡ epochs -> 0,1,2,...
                    batch_idx=num_batch - 1,        # index du batch dans ce round
                    global_batch=self.global_step,  # batch global continu
                    loss=this_loss,
                )

                self.log_scalar("train/loss_batch", this_loss)
                #self.log_scalar(f"train_nstep_{time_horizon}/loss_batch", this_loss)
                # batch global ++
                self.global_step += 1

                sum_loss += this_loss
                #batch_count += 1
                
                batch_wall_ms = (perf_counter() - batch_wall_t0) * 1000.0

                print(f"loss={loss.item():.4f} | buf_len={buf.size()}")
                print(
                    f"time: xfer={t_xfer_ms:.1f}ms | fwd={t_fwd_ms:.1f}ms | bwd={t_bwd_ms:.1f}ms | "
                    f"opt={t_opt_ms:.1f}ms | train={t_train_ms:.1f}ms | wall={batch_wall_ms:.1f}ms"
                )
                print("\n")
                print("-" * 30)
                num_batch += 1
                     

            stop.set(); producer.join(timeout=1.0)

            avg_loss = sum_loss / max(1, (num_batch-1))
            best_avg = min(best_avg, avg_loss)

           
            self.metrics.log_epoch(
                stage_n=time_horizon,
                epoch_in_stage=run_count - 1,   # rounds ≡ epochs
                global_epoch=run_count - 1,     # global axis "epochs" = 0..R-1
                loss=avg_loss,
            )

            self.log_scalar("train/loss_epoch", avg_loss, run_count - 1)
            #self.log_scalar(f"train_nstep_{time_horizon}/loss_epoch", avg_loss, run_count - 1)


            round_ms = (time.perf_counter() - round_t0) * 1000.0
            #self.log_scalar("time/round_ms", round_ms, step=run_count-1)

            print(f"[online] round={run_count} avg_loss={avg_loss:.6f} | best={best_avg:.6f} | round_time={round_ms/1000:.2f}s")

            # checkpoints
            if (run_count + 1) % 20 == 0:
                ckpt_path = self.ckpt_dir / f"ckpt_n{time_horizon}_e{run_count+1}.pth"
                torch.save(self.model.state_dict(), ckpt_path)

        print(f"> target reached? {best_avg <= self.target_loss} | rounds={run_count} | best_avg={best_avg:.6f}")
        self.metrics.plot_epoch_loss_by_stage(out_name="loss_rounds.png")
        final_ckpt = self.ckpt_dir / "my_checkpoint_final.pth"
        torch.save(self.model.state_dict(), final_ckpt)
        self.close()
