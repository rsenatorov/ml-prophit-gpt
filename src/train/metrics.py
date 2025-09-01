# metrics.py
from typing import Dict, Optional
import json
import math
import torch
import torch.nn.functional as F


class MetricsCalculator:
    """
    Computes token/direction metrics using the tokenizer vocab (if provided).
    """

    def __init__(self, vocab_json_path: Optional[str] = None):
        self.vocab_json_path = vocab_json_path
        self._directions = None
        self._loaded = False

        if vocab_json_path:
            try:
                with open(vocab_json_path, "r") as f:
                    vocab = json.load(f)
                dirs = []
                for i in range(len(vocab)):
                    d = vocab.get(str(i), {}).get("direction", "neutral")
                    if d == "bullish":
                        dirs.append(1)
                    elif d == "bearish":
                        dirs.append(-1)
                    else:
                        dirs.append(0)
                self._directions = torch.tensor(dirs, dtype=torch.long)
                self._loaded = True
            except Exception:
                self._directions = None
                self._loaded = False

    def hard_ce(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        B, T, V = logits.shape
        fl = logits.reshape(B * T, V)
        ft = targets.reshape(B * T)
        return float(F.cross_entropy(fl, ft, reduction="mean").detach().cpu().item())

    def token_top1(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        pred = logits.argmax(dim=-1)
        return float((pred == targets).float().mean().item())

    def direction_accuracy(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        if self._directions is None:
            return 0.0
        dirs = self._directions.to(logits.device)
        pred = logits.argmax(dim=-1)  # [B,T]
        # bounds check
        if pred.max() >= dirs.size(0) or targets.max() >= dirs.size(0):
            return 0.0
        pred_dir = dirs[pred]
        true_dir = dirs[targets]
        return float((pred_dir == true_dir).float().mean().item())

    def summarize(self, loss_dict: Dict[str, float], logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        hard = float(loss_dict.get("hard_ce", self.hard_ce(logits, targets)))
        perp = float(math.exp(hard)) if hard < 20.0 else float("inf")
        tok = self.token_top1(logits, targets)
        dir_acc = self.direction_accuracy(logits, targets)
        return {
            "soft_ce": float(loss_dict.get("soft_ce", 0.0)),
            "val_nll_hard": hard,
            "perplexity_hard": perp,
            "token_top1_acc": tok,
            "token_direction_accuracy": dir_acc,
        }
