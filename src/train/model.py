# model.py - FINAL VERSION v2
"""
GPT-2 architecture for time series.
CRITICAL FIXES:
- Removed stochastic depth completely
- Simplified attention computation
- Better weight initialization
- More stable normalization
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------
# Norm & Activation
# ---------------------
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        # Compute in float32 for stability
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_in: int, d_hidden: int):
        super().__init__()
        self.w1 = nn.Linear(d_in, d_hidden, bias=False)
        self.w2 = nn.Linear(d_in, d_hidden, bias=False)
        self.w3 = nn.Linear(d_hidden, d_in, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


# ---------------------
# RoPE
# ---------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, max_seq_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute for efficiency
        t = torch.arange(max_seq_len).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, :, None, :])
        self.register_buffer("sin_cached", emb.sin()[None, :, None, :])

    def forward(self, x):
        # x: [B, T, H, D]
        B, T, H, D = x.shape
        
        if D % 2 != 0:
            # Handle odd dimensions by passing through the last dimension
            x_even = x[..., :-1]
            x_odd = x[..., -1:]
            
            x_even = self._apply_rotary(x_even, T)
            return torch.cat([x_even, x_odd], dim=-1)
        else:
            return self._apply_rotary(x, T)
    
    def _apply_rotary(self, x, seq_len):
        # x: [B, T, H, D] where D is even
        B, T, H, D = x.shape
        x = x.reshape(B, T, H, D // 2, 2)
        
        cos = self.cos_cached[:, :seq_len, :, :D//2].to(x.dtype)
        sin = self.sin_cached[:, :seq_len, :, :D//2].to(x.dtype)
        
        # Apply rotation
        x0 = x[..., 0]
        x1 = x[..., 1]
        
        out = torch.stack([
            x0 * cos - x1 * sin,
            x0 * sin + x1 * cos
        ], dim=-1)
        
        return out.flatten(-2)


# ---------------------
# Attention
# ---------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # Create causal mask buffer
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(1, 1, 1024, 1024), diagonal=1).bool()
        )
        
    def forward(self, x, rope: RotaryEmbedding):
        B, T, C = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim)
        
        # Apply RoPE
        q = rope(q)
        k = rope(k)
        
        # Transpose for attention: [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores in float32 for stability
        q = q.float()
        k = k.float()
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        mask = self.mask[:, :, :T, :T]
        scores = scores.masked_fill(mask, float('-inf'))
        
        # Softmax
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(v.dtype)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.o_proj(out)
        
        return out


# ---------------------
# Transformer Block
# ---------------------
class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        rope: RotaryEmbedding = None,
    ):
        super().__init__()
        
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.rope = rope
        
    def forward(self, x):
        # Pre-norm attention block
        h = self.ln1(x)
        h = self.attn(h, self.rope)
        h = self.dropout(h)
        x = x + h
        
        # Pre-norm MLP block
        h = self.ln2(x)
        h = self.mlp(h)
        h = self.dropout(h)
        x = x + h
        
        return x


# ---------------------
# GPT-2 Time Series Model
# ---------------------
class GPT2TimeSeries(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Model dimensions
        d_model = config.model.d_model
        n_layers = config.model.n_layers
        n_heads = config.model.n_heads
        d_ff = config.model.d_ff
        dropout = config.model.dropout_attn  # Use same dropout everywhere
        max_seq_len = config.model.max_seq_length
        
        # Load vocabulary
        centers = np.load(config.data.centers_path)
        self.vocab_size = centers.shape[0]
        
        # Token embeddings
        self.token_emb = nn.Embedding(self.vocab_size, d_model)
        self.drop_emb = nn.Dropout(config.model.dropout_embedding)
        
        # Shared RoPE for all layers
        head_dim = d_model // n_heads
        self.rope = RotaryEmbedding(head_dim, config.model.rope_base_theta, max_seq_len)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                rope=self.rope,
            )
            for _ in range(n_layers)
        ])
        
        # Output layers
        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, self.vocab_size, bias=False)
        
        # Tie embeddings if specified
        if config.model.tie_embeddings:
            self.lm_head.weight = self.token_emb.weight
        
        self.max_seq_length = max_seq_len
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Xavier uniform initialization
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Normal initialization for embeddings
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids):
        B, T = input_ids.shape
        
        # Token embeddings
        x = self.token_emb(input_ids)
        x = self.drop_emb(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm and output projection
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
    ):
        """Generate tokens autoregressively."""
        self.eval()
        
        for _ in range(max_new_tokens):
            # Get logits for the last position
            idx_cond = input_ids[:, -self.max_seq_length:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token in input_ids[0].unique():
                    logits[:, token] /= repetition_penalty
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Find cutoff
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Set low prob tokens to 0
                for i in range(probs.shape[0]):
                    indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                    probs[i, indices_to_remove] = 0
                
                # Renormalize
                probs = probs / probs.sum(dim=-1, keepdim=True)
            
            # Sample
            idx_next = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, idx_next), dim=1)
        
        return input_ids