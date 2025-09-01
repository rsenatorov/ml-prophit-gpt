# config.py - FINAL VERSION v2
"""
Configuration management for GPT-2 time series model
CRITICAL FIXES APPLIED:
- Much shorter warmup (1000 steps)
- Lower learning rate (3e-5)
- Disabled most regularization initially
- Linear warmup instead of cubic
"""

import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Any
import torch


# ----------------------------
# Data
# ----------------------------
@dataclass
class DataConfig:
    tokens_dir: str = "data/tokens"
    centers_path: str = "tokenizer/kmeans_centers.npy"
    vocab_json: str = "tokenizer/vocab.json"

    sequence_length: int = 512
    batch_size: int = 8
    grad_accum_steps: int = 1  # kept simple

    # DataLoader performance knobs
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 4
    persistent_workers: bool = True
    multiprocessing_context: str = "spawn"

    # Optional on-disk cache for memmap'd tokens
    cache_dir: Optional[str] = "data/tokens_cache"

    # Date-based split
    train_val_boundary: str = "2025-05-01"
    val_start: str = "2025-05-01"
    val_end: str = "2025-07-31"
    test_start: str = "2025-08-01"

    # Optional conditioning
    add_volatility_bucket: bool = False
    volatility_num_buckets: int = 8
    volatility_window_steps: int = 96
    volatility_embedding_dim: int = 8
    
    # CRITICAL: Disable curriculum learning initially
    curriculum_learning: bool = False  # CHANGED: Was True


# ----------------------------
# Model
# ----------------------------
@dataclass
class ModelConfig:
    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12
    d_ff: int = 3072
    activation: str = "swiglu"
    norm: str = "rmsnorm"
    dropout_attn: float = 0.0  # CHANGED: Was 0.05, now disabled
    dropout_ffn: float = 0.0   # CHANGED: Was 0.05, now disabled
    dropout_residual: float = 0.0  # CHANGED: Was 0.05, now disabled
    dropout_embedding: float = 0.0
    rope_base_theta: float = 10000.0
    max_seq_length: int = 512
    tie_embeddings: bool = True
    stochastic_depth_prob: float = 0.0  # CHANGED: Was 0.05, now disabled completely
    vocab_size: int = 1024


# ----------------------------
# Loss
# ----------------------------
@dataclass
class LossConfig:
    name: str = "chronos_ce"
    eps_floor: float = 1e-8
    
    # Minimal regularization
    label_smoothing: float = 0.0  # CHANGED: Was 0.05, now disabled
    
    # Minimal z-loss
    z_loss_weight: float = 1e-5  # CHANGED: Was 1e-3, greatly reduced
    
    # Ignore index for padding
    ignore_index: int = -100

    # Legacy fields kept for compatibility
    tau_start: float = 0.08
    tau_end: float = 0.05
    tau_warmup_steps: int = 10_000
    tau_t0_steps: int = 300_000
    tau_t_mult: float = 1.5
    alpha_start: float = 0.7
    alpha_end: float = 0.3
    beta_start: float = 0.3
    beta_end: float = 0.7


# ----------------------------
# Optim
# ----------------------------
@dataclass
class OptimConfig:
    lr: float = 3e-5  # CHANGED: Was 1e-4, reduced for stability
    beta1: float = 0.9
    beta2: float = 0.999  # CHANGED: Was 0.98, increased for stability
    weight_decay: float = 0.1  # CHANGED: Was 0.01, increased
    eps: float = 1e-8


# ----------------------------
# Training
# ----------------------------
@dataclass
class TrainingConfig:
    train_forever: bool = True
    epochs: int = 10_000
    val_every_epochs: int = 1
    val_every_steps: int = 0

    save_every_n_steps: int = 5000  # CHANGED: Was 10000, save more frequently
    log_interval_steps: int = 100
    eval_interval_steps: int = 0

    # LR scheduler base - CRITICAL CHANGES
    lr_peak: float = 3e-5  # CHANGED: Was 1e-4, must match OptimConfig.lr
    betas: tuple = (0.9, 0.999)  # CHANGED: Match OptimConfig
    weight_decay: float = 0.1  # CHANGED: Match OptimConfig
    eps: float = 1e-8

    warmup_steps: int = 1000  # CHANGED: Was 4000, much shorter warmup
    warmup_mode: str = "linear"  # NEW: Use linear instead of cubic warmup
    scheduler_t0_steps: int = 50000  # CHANGED: Was 300000, shorter cycles
    scheduler_t_mult: float = 1.2  # CHANGED: Was 1.5, gentler increases
    min_lr_ratio: float = 0.3  # CHANGED: Was 0.1, don't let LR drop too low

    gradient_clipping: float = 1.0  # CHANGED: Was 0.5, allow larger gradients
    mixed_precision: str = "fp32"  # CHANGED: Was bf16, disable for debugging
    ema_enabled: bool = False
    ema_decay: float = 0.999

    run_dir: str = "runs/gpt2ts"
    runs_dir: str = "runs"
    run_name: Optional[str] = None

    log_dir: str = "logs"

    # for scheduling in loss
    max_steps: int = 1_000_000

    # compile toggle
    torch_compile: bool = False

    # ---------- Stability / NaN guard ----------
    nan_backoff_factor: float = 0.5
    nan_lr_floor: float = 1e-6  # CHANGED: Was 1e-7
    nan_warn_throttle_sec: float = 5.0
    nan_max_events: int = 50


# ----------------------------
# Inference
# ----------------------------
@dataclass
class InferenceConfig:
    horizon_steps: int = 64
    temperature: float = 0.9
    top_p: float = 0.9
    top_k: int = 0
    repetition_penalty: float = 1.0
    num_samples: int = 64


# ----------------------------
# Root config
# ----------------------------
@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    experiment_id: str = "exp_gpt2ts_v2_stable_fixed"  # CHANGED: New experiment ID

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_config(config_path: Optional[str] = None) -> Config:
    if config_path is None:
        return Config()
    p = Path(config_path)
    if not p.exists():
        return Config()
    if p.suffix.lower() in {".yml", ".yaml"}:
        import yaml  # lazy import
        with open(p, "r") as f:
            d = yaml.safe_load(f)
    else:
        with open(p, "r") as f:
            d = json.load(f)
    return Config(
        data=DataConfig(**d.get("data", {})),
        model=ModelConfig(**d.get("model", {})),
        loss=LossConfig(**d.get("loss", {})),
        training=TrainingConfig(**d.get("training", {})),
        inference=InferenceConfig(**d.get("inference", {})),
        optim=OptimConfig(**d.get("optim", {})),
        seed=d.get("seed", 42),
        device=d.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        experiment_id=d.get("experiment_id", "exp_gpt2ts_v2_stable_fixed"),
    )