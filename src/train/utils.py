# utils.py
"""
Utility helpers for training.
"""

import logging
import random
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed to {seed}")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EMA:
    """Exponential Moving Average (optional)."""

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()

    def update(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * p.data

    def apply_shadow(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.backup[name] = p.data.clone()
                p.data = self.shadow[name]

    def restore(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad and name in self.backup:
                p.data = self.backup[name]
        self.backup = {}

    def state_dict(self):
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state_dict):
        self.shadow = state_dict["shadow"]
        self.decay = state_dict["decay"]


class RollingAverage:
    def __init__(self, window_size: int = 500):
        self.window_size = int(window_size)
        self.buf = []

    def update(self, v: float):
        self.buf.append(float(v))
        if len(self.buf) > self.window_size:
            self.buf.pop(0)

    def get_average(self) -> float:
        if not self.buf:
            return 0.0
        return float(sum(self.buf) / len(self.buf))

    def reset(self) -> None:
        self.buf.clear()


def format_seconds(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h{minutes}m"
