# loss.py - FINAL VERSION v2
"""
Simplified loss function for stable training.
CRITICAL FIXES:
- Removed label smoothing by default
- Minimal z-loss regularization
- Simpler, more stable computation
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChronosT5Loss(nn.Module):
    """
    Simple and stable cross-entropy loss for next-token prediction.
    
    Components:
      - Standard cross-entropy loss (no label smoothing by default)
      - Optional minimal z-loss for preventing logit explosion
      - Numerical stability guards
    """

    def __init__(self, config, *, vocab_size: Optional[int] = None):
        super().__init__()
        self.label_smoothing = float(getattr(config.loss, "label_smoothing", 0.0))
        self.z_loss_weight = float(getattr(config.loss, "z_loss_weight", 1e-5))
        self.ignore_index = int(getattr(config.loss, "ignore_index", -100))
        self.vocab_size = vocab_size or getattr(config.model, "vocab_size", 1024)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Args:
          logits:  [B, T, V] - model predictions
          targets: [B, T] - ground truth token indices
          
        Returns:
          loss: scalar tensor
          loss_dict: metrics dictionary
        """
        B, T, V = logits.shape
        
        # Reshape for loss computation
        logits_flat = logits.reshape(-1, V)
        targets_flat = targets.reshape(-1)
        
        # Filter out ignore indices
        mask = targets_flat != self.ignore_index
        if not mask.any():
            # No valid targets
            zero_loss = torch.tensor(0.0, device=logits.device)
            return zero_loss, {
                "hard_ce": 0.0,
                "label_smoothed_ce": 0.0,
                "z_loss": 0.0,
                "tau": 0.0,
                "num_valid_tokens": 0,
                "perplexity_hard": float("inf"),
            }
        
        # Apply mask
        logits_valid = logits_flat[mask]
        targets_valid = targets_flat[mask]
        
        # Compute cross-entropy loss in float32 for stability
        with torch.cuda.amp.autocast(enabled=False):
            logits_valid = logits_valid.float()
            
            # Standard cross-entropy
            ce_loss = F.cross_entropy(
                logits_valid,
                targets_valid,
                reduction="mean",
                label_smoothing=self.label_smoothing
            )
            
            # Z-loss: regularize logits to prevent explosion
            if self.z_loss_weight > 0:
                # log(sum(exp(logits))) - for numerical stability, use logsumexp
                z = torch.logsumexp(logits_valid, dim=-1)
                z_loss = self.z_loss_weight * (z ** 2).mean()
                total_loss = ce_loss + z_loss
            else:
                z_loss = torch.tensor(0.0, device=logits.device)
                total_loss = ce_loss
        
        # Convert back to model dtype
        total_loss = total_loss.to(logits.dtype)
        
        # Compute metrics
        with torch.no_grad():
            # Hard cross-entropy (without label smoothing) for perplexity
            hard_ce = F.cross_entropy(logits_valid.float(), targets_valid, reduction="mean")
            hard_ce_value = float(hard_ce.item())
            
            # Perplexity
            if hard_ce_value < 20:
                perplexity = math.exp(hard_ce_value)
            else:
                perplexity = float("inf")
            
            # Token accuracy
            preds = logits_valid.argmax(dim=-1)
            token_acc = float((preds == targets_valid).float().mean().item())
        
        loss_dict = {
            "hard_ce": hard_ce_value,
            "label_smoothed_ce": float(ce_loss.item()),
            "z_loss": float(z_loss.item()),
            "tau": 0.0,  # For compatibility
            "num_valid_tokens": int(mask.sum().item()),
            "perplexity_hard": perplexity,
            "token_accuracy": token_acc,
        }
        
        # Final safety check
        if not torch.isfinite(total_loss):
            print(f"WARNING: Non-finite loss detected! CE={ce_loss.item():.4f}, Z={z_loss.item():.4f}")
            # Return a large but finite loss
            total_loss = torch.tensor(10.0, device=logits.device, dtype=logits.dtype)
        
        return total_loss, loss_dict