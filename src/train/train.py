import time
import warnings
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Performance knobs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ---- Stable autocast/GradScaler shim across torch versions ----
try:
    from torch.amp import autocast as _autocast_amp
    from torch.amp.grad_scaler import GradScaler as _GradScalerAmp

    def autocast(device_type: str = "cuda", dtype=None, enabled: bool = True):
        return _autocast_amp(device_type=device_type, dtype=dtype, enabled=enabled)

    GradScaler = _GradScalerAmp
except Exception:
    from torch.cuda.amp import autocast as _autocast_cuda
    from torch.cuda.amp import GradScaler as _GradScalerCuda

    def autocast(device_type: str = "cuda", dtype=None, enabled: bool = True):
        return _autocast_cuda(dtype=dtype, enabled=enabled)

    GradScaler = _GradScalerCuda
# ----------------------------------------------------------------

from config import build_config
from data_loader import create_dataloaders
from model import GPT2TimeSeries
from scheduler import build_scheduler
from checkpoint import CheckpointManager
from logger import get_logger, TrainingLogger
from utils import set_seed, RollingAverage, format_seconds
from loss import ChronosT5Loss
from metrics import MetricsCalculator


logger = get_logger(__name__, log_dir="logs")


class Trainer:
    def __init__(self, config):
        self.config = config
        set_seed(getattr(config, "seed", 42))

        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # Data
        logger.info("Creating data loaders...")
        self.train_loader, self.val_loader = create_dataloaders(config)
        self.train_dataset = self.train_loader.dataset
        self.batches_per_epoch = len(self.train_loader)

        # Model
        logger.info("Building model...")
        self.model = GPT2TimeSeries(config).to(self.device)
        if bool(getattr(config.training, "torch_compile", False)) and hasattr(torch, "compile"):
            self.model = torch.compile(self.model, mode="max-autotune")

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {n_params/1e6:.2f}M")

        # Loss: Chronos-style CE (+ optional z-loss)
        self.criterion = ChronosT5Loss(config).to(self.device)

        # Optim/scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.optim.lr,
            betas=(config.optim.beta1, config.optim.beta2),
            weight_decay=config.optim.weight_decay,
            eps=config.optim.eps,
        )
        self.lr_scheduler = build_scheduler(self.optimizer, config)

        # AMP
        mp = str(getattr(config.training, "mixed_precision", "bf16")).lower()
        self.use_amp = torch.cuda.is_available() and mp in {"fp16", "bf16"}
        self.amp_dtype = torch.float16 if mp == "fp16" else torch.bfloat16
        self.scaler = GradScaler(enabled=(self.use_amp and mp == "fp16")) if self.use_amp else None

        # Metrics/logger
        self.metrics = MetricsCalculator(getattr(config.data, "vocab_json", ""))
        self.tlogger = TrainingLogger(config)

        # Checkpointing
        self.ckpt = CheckpointManager(config)

        # State
        self.current_epoch = 0
        self.global_step = 0
        self.epoch_step = 0
        self.best_val_loss = float("inf")

        self.loss_avg = RollingAverage(500)
        self.dir_acc_avg = RollingAverage(500)
        self.token_acc_avg = RollingAverage(500)

        # Timing
        self.epoch_start_time: Optional[float] = None
        self.recent_step_times = []
        self.total_train_seconds: float = 0.0  # persisted across restarts
        self.run_start_time: float = time.time()

        # Non-finite guard state
        self._last_nan_warn_time = 0.0
        self._nan_events = 0

        # Resume if possible
        self._load_checkpoint()

    # --------------------------
    # Checkpoint load/save
    # --------------------------
    def _load_checkpoint(self):
        ckpt_path = self.ckpt.find_latest_checkpoint()
        if ckpt_path is None:
            logger.info("No checkpoint found, starting fresh.")
            return

        logger.info(f"Loading checkpoint: {ckpt_path}")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="You are using `torch.load` with `weights_only=False`",
                category=FutureWarning,
            )
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)  # torch>=2.5
            except TypeError:
                ckpt = torch.load(ckpt_path, map_location="cpu")  # older torch

        state = ckpt.get("state", {})
        if "model" in state:
            self.model.load_state_dict(state["model"])
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            try:
                self.lr_scheduler.load_state_dict(state["scheduler"])
            except Exception:
                pass

        self.current_epoch = ckpt.get("epoch", 0)
        self.global_step = ckpt.get("step", 0)
        self.epoch_step = ckpt.get("epoch_step", 0)
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self.total_train_seconds = float(ckpt.get("total_train_seconds", 0.0))
        self.run_start_time = time.time()

        # dataset epoch progress
        if "dataset_progress" in ckpt and hasattr(self.train_dataset, "load_epoch_progress"):
            self.train_dataset.load_epoch_progress(ckpt["dataset_progress"])

        logger.info(
            f"Resumed at epoch={self.current_epoch} step={self.global_step} "
            f"(epoch_step={self.epoch_step})"
        )

    def _save_checkpoint(self, epoch: int, is_epoch_end: bool = False, is_best: bool = False):
        metrics = {
            "loss": self.loss_avg.get_average(),
            "dir_acc": self.dir_acc_avg.get_average(),
            "tok_acc": self.token_acc_avg.get_average(),
            "soft_ce": self.loss_avg.get_average(),
        }

        dataset_progress = None
        if hasattr(self.train_dataset, "get_epoch_progress"):
            dataset_progress = self.train_dataset.get_epoch_progress()

        total_elapsed = self.total_train_seconds + max(0.0, time.time() - self.run_start_time)

        ckpt = {
            "epoch": epoch,
            "step": self.global_step,
            "epoch_step": self.epoch_step,
            "state": {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.lr_scheduler.state_dict() if hasattr(self.lr_scheduler, "state_dict") else {},
            },
            "metrics": metrics,
            "best_val_loss": self.best_val_loss,
            "dataset_progress": dataset_progress,
            "config": self.config.to_dict(),
            "total_train_seconds": total_elapsed,
        }

        save_path = Path(self.config.training.run_dir) / "checkpoints"
        save_path.mkdir(parents=True, exist_ok=True)
        if is_best:
            fname = "checkpoint_best.pt"
        elif is_epoch_end:
            fname = f"checkpoint_epoch_{epoch}.pt"
        else:
            fname = f"checkpoint_step_{self.global_step}.pt"

        fp = save_path / fname
        torch.save(ckpt, fp)

        latest = save_path / "checkpoint_latest.pt"
        try:
            shutil.copy2(fp, latest)  # cheap copy avoids a second serialization
        except Exception:
            torch.save(ckpt, latest)  # fallback

        logger.info(f"Saved checkpoint: {fname}")

    # --------------------------
    # Train/val
    # --------------------------
    def _select_targets(self, batch: Dict[str, Any]) -> torch.Tensor:
        return batch.get("target_ids", batch["labels"])

    def _forward_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        batch = {k: (v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        target_ids = self._select_targets(batch).to(self.device, non_blocking=True)

        if self.use_amp:
            with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=True):
                logits = self.model(input_ids)
                loss, loss_dict = self.criterion(logits, target_ids)
        else:
            logits = self.model(input_ids)
            loss, loss_dict = self.criterion(logits, target_ids)

        return {
            "loss": loss,
            "loss_dict": loss_dict,
            "logits": logits.detach(),
            "target_ids": target_ids.detach(),
        }

    def _batch_acc(self, logits: torch.Tensor, targets: torch.Tensor) -> tuple:
        probs = torch.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1)
        token_acc = float((pred == targets).float().mean().item())
        dir_acc = float(self.metrics.direction_accuracy(logits, targets))
        return token_acc, dir_acc

    def _eta_str(self, remaining_batches: int) -> str:
        if not self.recent_step_times:
            return "--"
        avg = sum(self.recent_step_times) / len(self.recent_step_times)
        return format_seconds(remaining_batches * avg)

    def _nan_guard(self, loss_value: float):
        # Throttled warning + automatic LR backoff + scaler reset (if any)
        now = time.time()
        if now - self._last_nan_warn_time >= self.config.training.nan_warn_throttle_sec:
            logger.warning(f"Non-finite loss detected (value={loss_value}); applying LR backoff and skipping batch.")
            self._last_nan_warn_time = now
        self._nan_events += 1

        # Back off LR
        bf = float(self.config.training.nan_backoff_factor)
        floor = float(self.config.training.nan_lr_floor)
        for pg in self.optimizer.param_groups:
            new_lr = max(pg["lr"] * bf, floor)
            pg["lr"] = new_lr

        # Reset AMP scaler if used
        if self.use_amp and self.scaler is not None:
            try:
                self.scaler.update(1.0)  # reset scale
            except Exception:
                pass

        if self._nan_events >= int(self.config.training.nan_max_events):
            logger.error("Too many non-finite events; aborting to avoid wasting compute.")
            raise RuntimeError("Exceeded nan_max_events")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.loss_avg.reset()
        self.dir_acc_avg.reset()
        self.token_acc_avg.reset()

        self.model.train()

        if self.epoch_step == 0:
            self.epoch_start_time = time.time()
        epoch_loss = 0.0

        # progress bar (resume-aware). Shows % through the entire dataset (batches_per_epoch).
        pbar = tqdm(
            enumerate(self.train_loader, start=self.epoch_step),
            total=self.batches_per_epoch,
            initial=self.epoch_step,
            ncols=160,
            desc=f"Epoch {epoch}",
            bar_format="{desc} [{elapsed}<{remaining}] {bar} {n_fmt}/{total_fmt} "
                       "({percentage:3.1f}%) | L:{postfix[0]:.3f} "
                       "D:{postfix[1]:.1f}% T:{postfix[2]:.1f}% | {postfix[3]} | {postfix[4]}",
            postfix=[0.0, 0.0, 0.0, "ETA:--", "Total:0s"],
        )

        step_t0 = time.time()
        for batch_idx, batch in pbar:
            if self.epoch_step >= self.batches_per_epoch:
                break  # safety

            out = self._forward_batch(batch)
            loss = out["loss"]

            # Guard: skip non-finite before backward
            if not torch.isfinite(loss):
                self._nan_guard(float(loss.detach().cpu().item()) if loss.numel() == 1 else float("nan"))
                # advance counters to avoid repeating same step
                self.global_step += 1
                self.epoch_step += 1
                # timing/ETA upkeep
                dt = time.time() - step_t0
                self.recent_step_times.append(dt)
                if len(self.recent_step_times) > 100:
                    self.recent_step_times.pop(0)
                step_t0 = time.time()
                remaining = self.batches_per_epoch - self.epoch_step
                eta = self._eta_str(remaining)
                total_elapsed_str = format_seconds(self.total_train_seconds + (time.time() - self.run_start_time))
                pbar.postfix = [self.loss_avg.get_average(),
                                self.dir_acc_avg.get_average() * 100.0,
                                self.token_acc_avg.get_average() * 100.0,
                                f"ETA:{eta}", f"Total:{total_elapsed_str}"]
                continue

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp and self.scaler is not None:
                with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=True):
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clipping)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clipping)
                self.optimizer.step()

            self.lr_scheduler.step()
            self.global_step += 1
            self.epoch_step += 1

            # metrics
            token_acc, dir_acc = self._batch_acc(out["logits"], out["target_ids"])

            lv = float(loss.detach())
            self.loss_avg.update(lv)
            self.dir_acc_avg.update(dir_acc)
            self.token_acc_avg.update(token_acc)
            epoch_loss += lv

            # timing for ETA
            dt = time.time() - step_t0
            self.recent_step_times.append(dt)
            if len(self.recent_step_times) > 100:
                self.recent_step_times.pop(0)
            step_t0 = time.time()

            # progress strings
            remaining = self.batches_per_epoch - self.epoch_step
            eta = self._eta_str(remaining)
            total_elapsed_str = format_seconds(self.total_train_seconds + (time.time() - self.run_start_time))
            pbar.postfix = [
                self.loss_avg.get_average(),
                self.dir_acc_avg.get_average() * 100.0,
                self.token_acc_avg.get_average() * 100.0,
                f"ETA:{eta}",
                f"Total:{total_elapsed_str}",
            ]

            # logging every N
            if self.global_step % self.config.training.log_interval_steps == 0:
                # 'tau' kept for CSV compatibility; Chronos loss does not use it
                self.tlogger.log_step(
                    self.global_step,
                    self.loss_avg.get_average(),
                    self.dir_acc_avg.get_average(),
                    self.token_acc_avg.get_average(),
                    out["loss_dict"].get("tau", 0.0),
                    self.optimizer.param_groups[0]["lr"],
                )

            # checkpoint every N steps
            if self.global_step % self.config.training.save_every_n_steps == 0:
                self._save_checkpoint(epoch, is_epoch_end=False, is_best=False)

        pbar.close()

        steps_this_epoch = max(1, self.epoch_step)
        avg_loss = epoch_loss / steps_this_epoch

        # accumulate total training seconds at epoch end
        if self.epoch_start_time is not None:
            self.total_train_seconds += (time.time() - self.epoch_start_time)
            self.epoch_start_time = None

        logger.info(f"Epoch {epoch} avg loss: {avg_loss:.4f}")
        return {"loss": float(avg_loss)}

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        for batch in self.val_loader:
            batch = {k: (v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            if self.use_amp:
                with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=True):
                    logits = self.model(batch["input_ids"])
                    loss, _ = self.criterion(logits, self._select_targets(batch))
            else:
                logits = self.model(batch["input_ids"])
                loss, _ = self.criterion(logits, self._select_targets(batch))
            total_loss += float(loss.detach())
            n_batches += 1

        total_loss = total_loss / max(1, n_batches)
        return {"soft_ce": float(total_loss)}  # name kept for checkpoint compatibility

    def run(self):
        try:
            while self.config.training.train_forever or self.current_epoch < self.config.training.epochs:
                # If we're resuming mid-epoch (epoch_step > 0), keep the same epoch label.
                if self.epoch_step == 0:
                    self.current_epoch += 1
                epoch_label = self.current_epoch

                train_metrics = self.train_epoch(epoch_label)

                # End of epoch: validation and checkpointing
                val_metrics = self.validate(epoch_label)
                val_loss = float(val_metrics.get("soft_ce", train_metrics["loss"]))

                # Best?
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss

                # Save epoch & (maybe) best
                self._save_checkpoint(epoch_label, is_epoch_end=True, is_best=is_best)

                # Reset per-epoch counters for next epoch
                self.epoch_step = 0
                self.loss_avg.reset()
                self.dir_acc_avg.reset()
                self.token_acc_avg.reset()

        except KeyboardInterrupt:
            logger.info("Interrupted by user; saving emergency checkpoint...")
            # emergency save without changing epoch label
            self._save_checkpoint(self.current_epoch, is_epoch_end=False, is_best=False)
            raise


def main():
    config = build_config()
    trainer = Trainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
