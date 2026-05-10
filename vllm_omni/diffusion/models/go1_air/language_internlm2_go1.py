# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""InternLM2-GO1 language stack (the ``language_model.*`` checkpoint subtree).

The trunk reuses ``InternLM2Block`` from :mod:`internlm2_layers`. We layer a
thin token-embedding + final-norm wrapper on top, and during forward we
collect the per-layer K/V tensors so the action expert can cross-attend
to them through its ``k_proj_layers`` / ``v_proj_layers`` projections.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import Go1AirConfig
from .internlm2_layers import (
    InternLM2Block,
    InternLM2BlockSpec,
    InternLM2RMSNorm,
    InternLM2RotaryEmbedding,
)


@dataclass
class LanguageStackOutput:
    last_hidden_state: torch.Tensor
    layer_kv: list[tuple[torch.Tensor, torch.Tensor]]


def language_block_spec(config: Go1AirConfig) -> InternLM2BlockSpec:
    return InternLM2BlockSpec(
        hidden_size=config.llm_hidden_size,
        intermediate_size=config.llm_intermediate_size,
        num_attention_heads=config.llm_num_attention_heads,
        num_key_value_heads=config.llm_num_key_value_heads,
        rms_norm_eps=config.llm_rms_norm_eps,
        rope_theta=config.llm_rope_theta,
        rope_scaling_factor=config.llm_rope_scaling_factor,
        rope_scaling_type=config.llm_rope_scaling_type,
        max_position_embeddings=config.llm_max_position_embeddings,
        attn_implementation=config.attn_implementation,
    )


class InternLM2Trunk(nn.Module):
    """The ``language_model.model.*`` subtree: embeddings + blocks + final norm."""

    def __init__(self, config: Go1AirConfig) -> None:
        super().__init__()
        self.config = config
        self.spec = language_block_spec(config)
        self.tok_embeddings = nn.Embedding(config.llm_vocab_size, config.llm_hidden_size)
        self.layers = nn.ModuleList([InternLM2Block(self.spec) for _ in range(config.llm_num_hidden_layers)])
        self.norm = InternLM2RMSNorm(config.llm_hidden_size, eps=config.llm_rms_norm_eps)
        head_dim = self.spec.resolved_head_dim()
        self.rotary_emb = InternLM2RotaryEmbedding(head_dim, self.spec)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> LanguageStackOutput:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids or inputs_embeds.")
        hidden = inputs_embeds if inputs_embeds is not None else self.tok_embeddings(input_ids)
        bsz, seqlen, _ = hidden.shape
        position_ids = torch.arange(seqlen, device=hidden.device).expand(bsz, seqlen)

        # Decoder-only InternLM2 needs causal masking; combine with padding
        # mask (if provided) into a (B, 1, T, T) additive attention mask.
        neg = torch.finfo(hidden.dtype).min
        causal = torch.triu(
            torch.full((seqlen, seqlen), neg, device=hidden.device, dtype=hidden.dtype),
            diagonal=1,
        )
        attn_bias = causal.unsqueeze(0).unsqueeze(0).expand(bsz, 1, seqlen, seqlen).contiguous()
        if attention_mask is not None and attention_mask.dim() == 2:
            invalid = (attention_mask == 0).view(bsz, 1, 1, seqlen)
            attn_bias = attn_bias.masked_fill(invalid, neg)

        layer_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for block in self.layers:
            hidden, k, v = block(hidden, position_ids, self.rotary_emb, attn_bias)
            layer_kv.append((k, v))
        hidden = self.norm(hidden)
        return LanguageStackOutput(last_hidden_state=hidden, layer_kv=layer_kv)

    def set_attention_implementation(self, implementation: str) -> None:
        self.spec.attn_implementation = implementation
        for block in self.layers:
            block.attention.spec.attn_implementation = implementation


class InternLM2GO1LanguageModel(nn.Module):
    """The full ``language_model`` subtree: trunk + vocabulary head."""

    def __init__(self, config: Go1AirConfig) -> None:
        super().__init__()
        self.config = config
        self.model = InternLM2Trunk(config)
        self.output = nn.Linear(
            config.llm_hidden_size,
            config.llm_vocab_size,
            bias=False,
        )
        if config.llm_tie_word_embeddings:
            self.output.weight = self.model.tok_embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> LanguageStackOutput:
        return self.model(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask)

    def set_attention_implementation(self, implementation: str) -> None:
        self.model.set_attention_implementation(implementation)
