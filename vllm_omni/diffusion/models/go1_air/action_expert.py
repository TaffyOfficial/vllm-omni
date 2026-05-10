# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GO-1-Air action-expert building blocks.

Provides the layer classes (input adaptors, time/freq embedders, decoder
block with VLM cross-attention, final action head) plus the squared-cosine
schedule helper. The orchestration (``predict_clean`` / ``sample_actions``)
lives on :class:`Go1Air` so all submodules sit at the checkpoint root, e.g.
``action_model.*``, ``time_embedder.*``, ``state_adaptor.*`` — matching the
safetensors index layout published by the upstream checkpoint.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import Go1AirConfig
from .internlm2_layers import (
    InternLM2BlockSpec,
    InternLM2FeedForward,
    InternLM2RMSNorm,
    InternLM2RotaryEmbedding,
    apply_rope,
    scaled_dot_product,
)


def action_block_spec(config: Go1AirConfig) -> InternLM2BlockSpec:
    return InternLM2BlockSpec(
        hidden_size=config.act_hidden_size,
        intermediate_size=config.act_intermediate_size,
        num_attention_heads=config.act_num_attention_heads,
        num_key_value_heads=config.act_num_key_value_heads,
        head_dim=config.act_head_dim,
        rms_norm_eps=config.act_rms_norm_eps,
        rope_theta=config.act_rope_theta,
        rope_scaling_factor=config.act_rope_scaling_factor,
        rope_scaling_type=config.act_rope_scaling_type,
        max_position_embeddings=config.act_max_position_embeddings,
        attn_implementation=config.attn_implementation,
    )


def squared_cos_cap_v2_alpha_bar(num_train_timesteps: int, device: torch.device) -> torch.Tensor:
    """Cumulative ``alpha_bar`` schedule used by diffusers' ``squaredcos_cap_v2``.

    ``alpha_bar(t) = cos((t/T + s)/(1+s) · π/2)^2`` with ``s = 0.008``,
    normalised so ``alpha_bar(0) = 1``.
    """
    s = 0.008
    steps = torch.arange(num_train_timesteps + 1, device=device, dtype=torch.float64) / num_train_timesteps
    f = torch.cos((steps + s) / (1.0 + s) * math.pi / 2.0).pow(2)
    return (f / f[0]).to(torch.float32)


class SinusoidalScalarEmbedding(nn.Module):
    """Sinusoidal embedding for a scalar input (timestep, frequency)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Sinusoidal embedding dim must be even; got {dim}.")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        result = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return result.to(t.dtype if t.dtype.is_floating_point else torch.float32)


class TimestepEmbedder(nn.Module):
    """Sinusoidal -> 2-layer MLP with parameter names ``mlp.0`` and ``mlp.2``.

    The sinusoidal embedding has a fixed 256-dim output; the MLP then lifts
    it to ``hidden_size``. This matches the upstream checkpoint shape
    ``mlp.0.weight = [hidden_size, 256]``.
    """

    SINUSOIDAL_DIM = 256

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.sinusoidal = SinusoidalScalarEmbedding(self.SINUSOIDAL_DIM)
        self.mlp = nn.Sequential(
            nn.Linear(self.SINUSOIDAL_DIM, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Sinusoidal output is fp32 when ``t`` is integer; cast to match the
        # MLP's parameter dtype so the Linear matmul doesn't reject the input.
        out = self.sinusoidal(t)
        return self.mlp(out.to(self.mlp[0].weight.dtype))


def make_state_action_adaptor(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    """3-layer MLP with parameter indices 0/2/4 (interleaved with SiLU)."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim, bias=True),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim, bias=True),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim, bias=True),
    )


class FinalLayer(nn.Module):
    """Output head: RMSNorm -> 2-layer MLP that maps to ``action_dim``.

    The intermediate width equals ``hidden_size`` (not the action expert's
    ``intermediate_size``), matching the upstream checkpoint shape
    ``fc1.weight = [hidden_size, hidden_size]`` / ``fc2.weight = [action_dim, hidden_size]``.
    """

    def __init__(self, hidden_size: int, action_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm_final = InternLM2RMSNorm(hidden_size, eps=eps)
        self.ffn_final = nn.Sequential()
        self.ffn_final.add_module("fc1", nn.Linear(hidden_size, hidden_size, bias=True))
        self.ffn_final.add_module("act", nn.SiLU())
        self.ffn_final.add_module("fc2", nn.Linear(hidden_size, action_dim, bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn_final(self.norm_final(x))


class ActionExpertAttention(nn.Module):
    """Action-side self-attention that concatenates VLM K/V into the joint key/value."""

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
        q = qkv[..., : self.num_kv_groups, :].reshape(
            bsz, seqlen, self.num_kv_heads * self.num_kv_groups, self.head_dim
        )
        k = qkv[..., -2, :]
        v = qkv[..., -1, :]
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        action_position_ids: torch.Tensor,
        vlm_position_ids: torch.Tensor,
        rotary_emb: InternLM2RotaryEmbedding,
        vlm_k: torch.Tensor,
        vlm_v: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q, k_act, v_act = self.split_qkv(hidden_states)
        cos_act, sin_act = rotary_emb(action_position_ids)
        q, k_act = apply_rope(q, k_act, cos_act, sin_act)

        # Re-stamp absolute VLM positions onto the projected VLM keys so action
        # queries can use cross-modal positional information. Cos/sin from the
        # rotary embedding are fp32; cast the result back to vlm_k's dtype so
        # the joint K stays in the same precision as the action-side K.
        orig_dtype = vlm_k.dtype
        cos_vlm, sin_vlm = rotary_emb(vlm_position_ids)
        cos_vlm = cos_vlm.unsqueeze(1)
        sin_vlm = sin_vlm.unsqueeze(1)
        half = vlm_k.shape[-1] // 2
        rotated = torch.cat((-vlm_k[..., half:], vlm_k[..., :half]), dim=-1)
        vlm_k = ((vlm_k * cos_vlm) + (rotated * sin_vlm)).to(orig_dtype)

        joint_k = torch.cat([vlm_k, k_act], dim=2)
        joint_v = torch.cat([vlm_v, v_act], dim=2)

        attn_out = scaled_dot_product(
            q,
            joint_k,
            joint_v,
            num_kv_groups=self.num_kv_groups,
            mask=attention_mask,
            implementation=self.spec.attn_implementation,
        )
        bsz, _, seqlen, _ = q.shape
        out = attn_out.transpose(1, 2).reshape(bsz, seqlen, -1)
        return self.wo(out)


class ActionExpertBlock(nn.Module):
    """Action-side decoder block: cross-attention into joint VLM/action keys, then FFN."""

    def __init__(self, spec: InternLM2BlockSpec) -> None:
        super().__init__()
        self.attention_norm = InternLM2RMSNorm(spec.hidden_size, eps=spec.rms_norm_eps)
        self.attention = ActionExpertAttention(spec)
        self.ffn_norm = InternLM2RMSNorm(spec.hidden_size, eps=spec.rms_norm_eps)
        self.feed_forward = InternLM2FeedForward(spec)

    def forward(
        self,
        hidden_states: torch.Tensor,
        action_position_ids: torch.Tensor,
        vlm_position_ids: torch.Tensor,
        rotary_emb: InternLM2RotaryEmbedding,
        vlm_k: torch.Tensor,
        vlm_v: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_out = self.attention(
            self.attention_norm(hidden_states),
            action_position_ids,
            vlm_position_ids,
            rotary_emb,
            vlm_k,
            vlm_v,
            attention_mask,
        )
        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.feed_forward(self.ffn_norm(hidden_states))
        return hidden_states
