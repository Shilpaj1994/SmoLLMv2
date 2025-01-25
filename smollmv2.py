#! /usr/bin/env python
"""
SmollmV2 model implementation
Author: Shilpaj Bhalerao
Date: 2025-01-19
"""
# Third-Party Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Local Imports
from config import SmollmConfig, RoPEConfig


class RoPEAttention:
    """
    Rotary Position Embedding attention with support for different Q/K dimensions
    """
    def __init__(self, head_dim, kv_dim, base=RoPEConfig.base):
        """
        Initialize rotary embeddings
        Args:
            head_dim: Dimension of query head
            kv_dim: Dimension of key/value head
            base: Base for the angle calculations (default: 10000)
        """
        super().__init__()
        
        # Generate theta parameter for rotary embeddings for both Q and K dimensions
        inv_freq_k = 1.0 / (base ** (torch.arange(0, kv_dim, 2).float() / kv_dim))
        self.register_buffer('inv_freq_k', inv_freq_k)
        
        self.head_dim = head_dim
        self.kv_dim = kv_dim
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def _update_cos_sin_cache(self, x, seq_len):
        """Update cached cos and sin values for given sequence length"""
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq_k)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq_k)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]

    def _rotate_half(self, x):
        """Rotate half the hidden dims of the input."""
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def __call__(self, q, k):
        """
        Apply rotary embeddings to input queries and keys
        Args:
            q: Query tensor of shape (batch, n_head, seq_len, head_dim)
            k: Key tensor of shape (batch, n_head, seq_len, kv_dim)
        Returns:
            q_rot: Rotated query tensor
            k_rot: Rotated key tensor
        """
        seq_len = q.shape[2]
        self._update_cos_sin_cache(k, seq_len)
        
        # Apply rotary embeddings to keys
        k_cos = self.cos_cached[..., :self.kv_dim]
        k_sin = self.sin_cached[..., :self.kv_dim]
        k_rot = (k * k_cos) + (self._rotate_half(k) * k_sin)
        
        # For queries, we only apply rotation to the part that interacts with keys
        q_part = q[..., :self.kv_dim]
        q_cos = self.cos_cached[..., :self.kv_dim]
        q_sin = self.sin_cached[..., :self.kv_dim]
        q_rot_part = (q_part * q_cos) + (self._rotate_half(q_part) * q_sin)
        
        # Combine rotated part with unrotated parts for query
        q_rot = torch.cat([q_rot_part, q[..., self.kv_dim:]], dim=-1)
        
        return q_rot, k_rot

    def register_buffer(self, name, tensor):
        """Helper function to register a buffer"""
        setattr(self, name, tensor)


class CausalSelfAttention(nn.Module):
    """
    Causal self-attention mechanism with reduced KV dimensions and RoPE
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Calculate dimensions
        self.head_dim = config.n_embd // config.n_head  # 576/9 = 64
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        # Make kv_dim divisible by n_head (189 is closest to 192 that's divisible by 9)
        self.kv_dim = 189  # 189 = 9 * 21, closest to 192 that's divisible by 9
        self.kv_dim_per_head = self.kv_dim // self.n_head  # 21
        
        # Separate projections with reduced dimensions for k,v
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, self.kv_dim, bias=False)  # 189 dimensions
        self.v_proj = nn.Linear(config.n_embd, self.kv_dim, bias=False)  # 189 dimensions
        
        # output projection
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        # rotary embeddings
        self.rope = RoPEAttention(self.head_dim, self.kv_dim_per_head)

    def forward(self, x):
        B, T, C = x.size()
        
        # calculate query, key, values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # reshape with exact dimensions
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.kv_dim_per_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.kv_dim_per_head).transpose(1, 2)

        # apply rotary embeddings
        q, k = self.rope(q, k)

        # pad k and v to match q dimension for attention
        k_pad = torch.zeros_like(q)
        v_pad = torch.zeros_like(q)
        k_pad[..., :self.kv_dim_per_head] = k
        v_pad[..., :self.kv_dim_per_head] = v

        # flash attention
        y = F.scaled_dot_product_attention(q, k_pad, v_pad, is_causal=True)

        # reshape back
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # output projection
        y = self.o_proj(y)
        return y


class MLP(nn.Module):
    """
    MLP (Multi-Layer Perceptron) layer with gate/up/down projection structure
    """
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(config.n_embd * config.mlp_ratio) - 1
        self.gate_proj = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.down_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        # SwiGLU activation as used in PaLM, Llama, etc.
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        x = F.silu(gate) * up
        x = self.down_proj(x)
        return x


class Block(nn.Module):
    """
    Transformer block
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=False)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=False)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class SmollmV2(nn.Module):
    """
    SmollmV2 model
    """
    def __init__(self, config=SmollmConfig()):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, bias=False),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing
        self.transformer.wte.weight = self.lm_head.weight

        # weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std = 0.04)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
