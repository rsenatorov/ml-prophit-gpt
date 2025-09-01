# checkpoint.py
"""
Checkpoint management for training with enhanced resumption support.
Saves a 'checkpoint_latest.pt' pointer every time, and supports best/epoch/step files.
"""

import shutil
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Iterable

import torch

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manage model checkpoints with perfect resumption support."""

    def __init__(self, config, _logger: Optional[logging.Logger] = None):
        self.config = config
        self.run_dir = Path(config.training.run_dir)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_metric = float("inf")
        self.keep_last_k = 5

    # ---------- core API ----------
    def save_checkpoint(
        self,
        state: Dict,
        epoch: int,
        step: int,
        metrics: Dict,
        extra: Optional[Dict] = None,
        is_best: bool = False,
        is_epoch_end: bool = False,
    ) -> Path:
        """Save a checkpoint file and update 'latest' pointer."""
        if is_best:
            filename = "checkpoint_best.pt"
        elif is_epoch_end:
            filename = f"checkpoint_epoch_{epoch}.pt"
        else:
            filename = f"checkpoint_step_{step}.pt"

        filepath = self.checkpoint_dir / filename

        checkpoint = {
            "epoch": epoch,
            "step": step,
            "state": state,
            "metrics": metrics,
            "config": self.config.to_dict(),
        }
        if extra:
            checkpoint.update(extra)

        self._atomic_save(checkpoint, filepath)
        logger.info(f"Saved checkpoint: {filename}")

        # Update latest pointer for easy resume (atomic replace)
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        tmp_latest = latest_path.with_suffix(latest_path.suffix + ".tmp")
        try:
            shutil.copy2(filepath, tmp_latest)
            try:
                tmp_latest.replace(latest_path)
            except Exception:
                shutil.move(str(tmp_latest), str(latest_path))
        finally:
            if tmp_latest.exists():
                try:
                    tmp_latest.unlink(missing_ok=True)
                except Exception:
                    pass

        # Track and cleanup old step checkpoints
        if not is_best and not is_epoch_end:
            self._cleanup_old_checkpoints()

        # Track best metric
        if is_best:
            m = metrics.get("soft_ce", None)
            if m is not None:
                self.best_metric = float(m)

        return filepath

    def save(
        self,
        model,
        optimizer,
        scheduler,
        epoch: int,
        step: int,
        metrics: Optional[Dict] = None,
        extra: Optional[Dict] = None,
        is_epoch_end: bool = False,
    ):
        """Convenience wrapper called by train loop."""
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else {},
            "scheduler": scheduler.state_dict() if (scheduler is not None and hasattr(scheduler, "state_dict")) else {},
        }
        return self.save_checkpoint(
            state=state,
            epoch=epoch,
            step=step,
            metrics=metrics or {},
            extra=extra,
            is_best=False,
            is_epoch_end=is_epoch_end,
        )

    def save_best(
        self,
        model,
        optimizer,
        scheduler,
        epoch: int,
        step: int,
        metrics: Dict,
        extra: Optional[Dict] = None,
    ):
        """Save best checkpoint when validation improves."""
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else {},
            "scheduler": scheduler.state_dict() if (scheduler is not None and hasattr(scheduler, "state_dict")) else {},
        }
        self.best_metric = float(metrics.get("soft_ce", self.best_metric))
        return self.save_checkpoint(
            state=state,
            epoch=epoch,
            step=step,
            metrics=metrics,
            extra=extra,
            is_best=True,
            is_epoch_end=False,
        )

    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> Optional[Dict]:
        """Load a checkpoint dict."""
        if checkpoint_path is None:
            checkpoint_path = self.find_latest_checkpoint()
            if checkpoint_path is None:
                logger.info("No checkpoint found to resume from")
                return None

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return None

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch')} step {checkpoint.get('step')}")
        return checkpoint

    def load_if_exists(self, model, optimizer=None, scheduler=None) -> bool:
        """Restore model/opt/scheduler if a checkpoint exists."""
        ckpt = self.load_checkpoint()
        if ckpt is None:
            return False

        state = ckpt.get("state", {})
        if "model" in state:
            model.load_state_dict(state["model"])
        if optimizer is not None and "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and "scheduler" in state:
            try:
                scheduler.load_state_dict(state["scheduler"])
            except Exception:
                pass

        m = ckpt.get("metrics", {})
        if "soft_ce" in m:
            self.best_metric = float(m["soft_ce"])
        return True

    # ---------- helpers ----------
    def _cleanup_old_checkpoints(self):
        """Keep only last K step checkpoints to save disk (race-safe)."""
        step_checkpoints = self._existing(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
        if len(step_checkpoints) <= self.keep_last_k:
            return
        step_checkpoints.sort(key=self._safe_key_for_sort, reverse=True)
        for p in step_checkpoints[self.keep_last_k :]:
            try:
                p.unlink(missing_ok=True)
            except FileNotFoundError:
                pass
            except Exception:
                pass

    def _safe_key_for_sort(self, p: Path):
        e, s = self._extract_epoch_step(p.name)
        try:
            mt = p.stat().st_mtime
        except FileNotFoundError:
            mt = -1.0
        return (e, s, mt)

    def _existing(self, paths: Iterable[Path]):
        out = []
        for p in paths:
            try:
                if p.exists():
                    out.append(p)
            except FileNotFoundError:
                pass
            except Exception:
                pass
        return out

    def _atomic_save(self, obj: Dict, path: Path):
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(obj, tmp)
        try:
            tmp.replace(path)  # atomic on NTFS
        except Exception:
            shutil.move(str(tmp), str(path))
        finally:
            if tmp.exists():
                try:
                    tmp.unlink(missing_ok=True)
                except Exception:
                    pass

    def _extract_epoch_step(self, name: str) -> Tuple[int, int]:
        # Support names like checkpoint_epoch_3.pt or checkpoint_epoch_3_step_20000.pt
        epoch, step = 0, 0
        parts = name.replace(".pt", "").split("_")
        try:
            if "epoch" in parts:
                epoch = int(parts[parts.index("epoch") + 1])
            if "step" in parts:
                step = int(parts[parts.index("step") + 1])
        except Exception:
            pass
        return epoch, step

    def find_latest_checkpoint(self) -> Optional[Path]:
        """Prefer 'checkpoint_latest.pt'; else pick the newest by (epoch, step, mtime)."""
        latest = self.checkpoint_dir / "checkpoint_latest.pt"
        if latest.exists():
            return latest
        cands = self._existing(self.checkpoint_dir.glob("checkpoint*.pt"))
        if not cands:
            return None
        cands.sort(key=self._safe_key_for_sort, reverse=True)
        return cands[0] if cands else None
