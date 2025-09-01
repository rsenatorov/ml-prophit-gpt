# logger.py
"""
Logging utilities and CSV metric writers for training/validation.
"""

import csv
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict

logger = logging.getLogger(__name__)


def get_logger(name: str, log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """
    Create a module logger that logs to both console and a file in `log_dir`.
    Safe to call multiple times.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    lg = logging.getLogger(name)
    lg.setLevel(level)
    lg.propagate = False

    if not lg.handlers:
        # Console
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S"))
        lg.addHandler(ch)

        # File
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(log_path / f"{name.replace('.', '_')}_{ts}.log", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"))
        lg.addHandler(fh)

    return lg


class TrainingLogger:
    """Lightweight CSV logger for steps/epochs."""

    def __init__(self, config):
        self.config = config
        self.log_dir = Path(config.training.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{config.experiment_id}_{timestamp}"

        self.epoch_log_path = self.log_dir / f"{self.run_name}_epoch_metrics.csv"
        self.step_log_path = self.log_dir / f"{self.run_name}_step_metrics.csv"

        self._init_csv_files()

    def _init_csv_files(self):
        with open(self.epoch_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss",
                "val_loss",
                "direction_accuracy",
                "token_accuracy",
                "learning_rate",
                "timestamp",
            ])

        with open(self.step_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step",
                "loss",
                "direction_accuracy",
                "token_accuracy",
                "tau",
                "learning_rate",
                "timestamp",
            ])

    def log_step(self, step: int, loss: float, dir_acc: float, tok_acc: float, tau: float, lr: float):
        # Throttle to avoid huge CSVs (log every 500 steps)
        if step % 500 == 0:
            with open(self.step_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([step, loss, dir_acc, tok_acc, tau, lr, datetime.now().isoformat()])

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float,
                  dir_acc: float, tok_acc: float, lr: float):
        with open(self.epoch_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, dir_acc, tok_acc, lr, datetime.now().isoformat()])
