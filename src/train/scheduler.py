# scheduler.py - FINAL VERSION v2
"""
Cosine annealing with warmup and restarts.
CRITICAL FIX: Linear warmup instead of cubic for proper early learning
"""

import math
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWithRestartsLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, t0_steps, t_mult=1.5, min_lr_ratio=0.05, 
                 warmup_mode="linear", last_epoch=-1):
        self.warmup_steps = int(warmup_steps)
        self.t0_steps = int(t0_steps)
        self.t_mult = float(t_mult)
        self.min_lr_ratio = float(min_lr_ratio)
        self.warmup_mode = warmup_mode  # NEW: Support different warmup modes
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_step = 0
        
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        s = self.current_step
        
        # Warmup phase - CRITICAL FIX: Use linear by default
        if s < self.warmup_steps:
            progress = s / max(1, self.warmup_steps)
            
            if self.warmup_mode == "linear":
                # Linear warmup - much more aggressive early learning
                warmup_factor = progress
            elif self.warmup_mode == "cubic":
                # Original cubic (smoothstep) - too slow!
                warmup_factor = progress ** 2 * (3.0 - 2.0 * progress)
            elif self.warmup_mode == "cosine":
                # Cosine warmup - compromise between linear and cubic
                warmup_factor = 0.5 * (1.0 - math.cos(math.pi * progress))
            else:
                # Default to linear
                warmup_factor = progress
                
            # Ensure we never go below a minimum threshold
            warmup_factor = max(0.01, warmup_factor)  # At least 1% of target LR
            
            return [base * warmup_factor for base in self.base_lrs]

        # Cosine annealing with restarts after warmup
        s2 = s - self.warmup_steps
        cycle_start = 0
        cycle_len = self.t0_steps
        cycle_num = 0
        
        # Find current cycle
        while cycle_start + cycle_len <= s2:
            cycle_start += cycle_len
            cycle_len = int(cycle_len * self.t_mult)
            cycle_num += 1
        
        # Position within current cycle
        pos_in_cycle = s2 - cycle_start
        pos = pos_in_cycle / max(1, cycle_len)
        pos = min(pos, 1.0)  # Clamp to avoid numerical issues
        
        # Cosine annealing within cycle
        lrs = []
        for base_lr in self.base_lrs:
            min_lr = base_lr * self.min_lr_ratio
            # Standard cosine annealing
            lr = min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * pos))
            lrs.append(lr)
        
        return lrs

    def step(self, epoch=None):
        self.current_step += 1
        for pg, lr in zip(self.optimizer.param_groups, self.get_lr()):
            pg["lr"] = lr

    def state_dict(self):
        return {
            "current_step": self.current_step,
            "warmup_steps": self.warmup_steps,
            "t0_steps": self.t0_steps,
            "t_mult": self.t_mult,
            "min_lr_ratio": self.min_lr_ratio,
            "warmup_mode": self.warmup_mode,
            "base_lrs": self.base_lrs,
        }

    def load_state_dict(self, state_dict):
        self.current_step = int(state_dict["current_step"])
        self.warmup_steps = int(state_dict["warmup_steps"])
        self.t0_steps = int(state_dict["t0_steps"])
        self.t_mult = float(state_dict["t_mult"])
        self.min_lr_ratio = float(state_dict["min_lr_ratio"])
        self.warmup_mode = state_dict.get("warmup_mode", "linear")
        self.base_lrs = list(state_dict["base_lrs"])


def build_scheduler(optimizer, config):
    return CosineAnnealingWithRestartsLR(
        optimizer,
        warmup_steps=config.training.warmup_steps,
        t0_steps=config.training.scheduler_t0_steps,
        t_mult=config.training.scheduler_t_mult,
        min_lr_ratio=config.training.min_lr_ratio,
        warmup_mode=getattr(config.training, "warmup_mode", "linear"),  # Use linear by default
    )