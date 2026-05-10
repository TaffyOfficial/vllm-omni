# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared InternLM2-style transformer building blocks.

Both ``language_model`` and ``action_model`` in the GO-1-Air checkpoint use
the InternLM2 parameter naming convention (``attention.wqkv`` combined Q/K/V
projection, ``feed_forward.w1/w2/w3`` SwiGLU MLP, ``attention_norm`` /
``ffn_norm`` RMSNorm, ``attention.wo`` output projection). These primitives
are factored here so the LM and the action expert decoder share the same
building blocks while remaining independent ``nn.Module`` trees.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class InternLM2BlockSpec:
    """Hyperparameters for one InternLM2-style transformer stack."""

    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: Optional[int] = None
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1_000_000.0
    rope_scaling_factor: float = 2.0
    rope_scaling_type: str = "dynamic"
    max_position_embeddings: int = 32_768
    attn_implementation: str = "eager"

    def resolved_head_dim(self) -> int:
        return self.head_dim if self.head_dim is not None else self.hidden_size // self.num_attention_heads

    @property
    def num_kv_groups(self) -> int:
        # Number of query heads sharing one KV head.
        return self.num_attention_heads // self.num_key_value_heads


class InternLM2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * x).to(input_dtype)


def _build_inv_freq(head_dim: int, base: float, device: torch.device) -> torch.Tensor:
    return 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))


class InternLM2RotaryEmbedding(nn.Module):
    """RoPE with optional NTK-aware dynamic scaling.

    Dynamic scaling adjusts the base when the sequence exceeds the trained
    context window. For GO-1-Air's typical input length (vision + state +
    action tokens, well under 32K) this short-circuits to the static path.
    """

    def __init__(self, head_dim: int, spec: InternLM2BlockSpec) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = float(spec.rope_theta)
        self.scaling_factor = float(spec.rope_scaling_factor)
        self.scaling_type = spec.rope_scaling_type
        self.max_position_embeddings = int(spec.max_position_embeddings)
        inv_freq = _build_inv_freq(head_dim, self.base, torch.device("cpu"))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        device = position_ids.device
        seq_max = int(position_ids.max().item()) + 1 if position_ids.numel() > 0 else 1
        if self.scaling_type == "dynamic" and seq_max > self.max_position_embeddings:
            scale = (
                self.scaling_factor * seq_max / self.max_position_embeddings
                - (self.scaling_factor - 1.0)
            )
            new_base = self.base * (scale ** (self.head_dim / (self.head_dim - 2)))
            inv_freq = _build_inv_freq(self.head_dim, new_base, device)
        else:
            inv_freq = self.inv_freq.to(device)

        freqs = position_ids.float().unsqueeze(-1) * inv_freq.unsqueeze(0)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot.to(q.dtype), k_rot.to(k.dtype)


def repeat_kv(x: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return x
    b, h, t, d = x.shape
    return x.unsqueeze(2).expand(b, h, repeats, t, d).reshape(b, h * repeats, t, d)


class InternLM2Attention(nn.Module):
    """Self-attention with combined ``wqkv`` + grouped-query attention.

    The combined projection lays out per kv-group as ``num_kv_groups`` query
    heads followed by one K head and one V head, then flattens. We undo that
    layout in :meth:`forward` to obtain Q/K/V tensors compatible with
    standard scaled dot-product attention.
    """

    def __init__(self, spec: InternLM2BlockSpec) -> None:
        super().__init__()
        self.spec = spec
        self.head_dim = spec.resolved_head_dim()
        self.num_q_heads = spec.num_attention_heads
        self.num_kv_heads = spec.num_key_value_heads
        self.num_kv_groups = spec.num_kv_groups
        per_group_heads = self.num_kv_groups + 2
        self.wqkv = nn.Linear(
            spec.hidden_size,
            self.num_kv_heads * per_group_heads * self.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(self.num_q_heads * self.head_dim, spec.hidden_size, bias=False)

    def split_qkv(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seqlen, _ = hidden_states.shape
        per_group_heads = self.num_kv_groups + 2
        qkv = self.wqkv(hidden_states).view(
            bsz,
            seqlen,
            self.num_kv_heads,
            per_group_heads,
            self.head_dim,
        )
        # Q: first num_kv_groups slices per kv-group → fold groups into a single head axis.
        q = qkv[..., : self.num_kv_groups, :].reshape(
            bsz, seqlen, self.num_kv_heads * self.num_kv_groups, self.head_dim
        )
        k = qkv[..., -2, :]
        v = qkv[..., -1, :]
        # (B, H, T, D)
        return (
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_emb: InternLM2RotaryEmbedding,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = self.split_qkv(hidden_states)
        cos, sin = rotary_emb(position_ids)
        q, k = apply_rope(q, k, cos, sin)
        attn_out = scaled_dot_product(
            q,
            k,
            v,
            num_kv_groups=self.num_kv_groups,
            mask=attention_mask,
            implementation=self.spec.attn_implementation,
        )
        bsz, _, seqlen, _ = q.shape
        out = attn_out.transpose(1, 2).reshape(bsz, seqlen, -1)
        return self.wo(out), k, v


def scaled_dot_product(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    num_kv_groups: int,
    mask: Optional[torch.Tensor],
    implementation: str,
) -> torch.Tensor:
    k_full = repeat_kv(k, num_kv_groups)
    v_full = repeat_kv(v, num_kv_groups)
    if implementation == "sdpa":
        return F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=mask, is_causal=mask is None)
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k_full.transpose(-2, -1)) * scale
    if mask is not None:
        scores = scores + mask
    probs = scores.softmax(dim=-1).to(v_full.dtype)
    return torch.matmul(probs, v_full)


class InternLM2FeedForward(nn.Module):
    """SwiGLU MLP with InternLM2 parameter naming (w1 gate, w3 up, w2 down)."""

    def __init__(self, spec: InternLM2BlockSpec) -> None:
        super().__init__()
        self.w1 = nn.Linear(spec.hidden_size, spec.intermediate_size, bias=False)
        self.w3 = nn.Linear(spec.hidden_size, spec.intermediate_size, bias=False)
        self.w2 = nn.Linear(spec.intermediate_size, spec.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class InternLM2Block(nn.Module):
    """One pre-norm decoder block: attention → FFN, both residual."""

    def __init__(self, spec: InternLM2BlockSpec) -> None:
        super().__init__()
        self.attention_norm = InternLM2RMSNorm(spec.hidden_size, eps=spec.rms_norm_eps)
        self.attention = InternLM2Attention(spec)
        self.ffn_norm = InternLM2RMSNorm(spec.hidden_size, eps=spec.rms_norm_eps)
        self.feed_forward = InternLM2FeedForward(spec)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_emb: InternLM2RotaryEmbedding,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_input = self.attention_norm(hidden_states)
        attn_out, k, v = self.attention(attn_input, position_ids, rotary_emb, attention_mask)
        hidden_states = hidden_states + attn_out
        ffn_input = self.ffn_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(ffn_input)
        return hidden_states, k, v
