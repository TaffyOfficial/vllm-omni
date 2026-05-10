# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""InternViT-6B-style vision encoder used by the GO-1-Air checkpoint.

Implements a vanilla Vision Transformer with Layer Scale (the ``ls1`` /
``ls2`` parameters) and the parameter naming convention found in the GO-1
safetensors index: ``vision_model.embeddings.*`` and
``vision_model.encoder.layers.<i>.{attn.qkv, attn.proj, mlp.fc1, mlp.fc2,
norm1, norm2, ls1, ls2}``.

We deliberately keep this self-contained rather than inheriting from a
HuggingFace base class because the checkpoint's parameter naming differs
from ``transformers.models.internvl`` — direct subclassing would force a
runtime key-rewrite layer and obscure the implementation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .config import Go1AirConfig


@dataclass
class InternViTSpec:
    image_size: int
    patch_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    qkv_bias: bool
    qk_normalization: bool
    norm_type: str

    @classmethod
    def from_config(cls, config: Go1AirConfig) -> InternViTSpec:
        return cls(
            image_size=config.image_resolution[0],
            patch_size=config.vision_patch_size,
            hidden_size=config.vision_hidden_size,
            intermediate_size=config.vision_intermediate_size,
            num_hidden_layers=config.vision_num_hidden_layers,
            num_attention_heads=config.vision_num_attention_heads,
            qkv_bias=config.vision_qkv_bias,
            qk_normalization=config.vision_qk_normalization,
            norm_type=config.vision_norm_type,
        )

    @property
    def num_patches(self) -> int:
        side = self.image_size // self.patch_size
        return side * side

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


def _build_norm(spec: InternViTSpec) -> nn.Module:
    if spec.norm_type == "layer_norm":
        return nn.LayerNorm(spec.hidden_size, eps=1e-6)
    if spec.norm_type == "rms_norm":
        return nn.RMSNorm(spec.hidden_size, eps=1e-6)
    raise ValueError(f"Unsupported InternViT norm type: {spec.norm_type}")


class InternViTPatchEmbeddings(nn.Module):
    def __init__(self, spec: InternViTSpec) -> None:
        super().__init__()
        self.spec = spec
        self.class_embedding = nn.Parameter(torch.zeros(1, 1, spec.hidden_size))
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=spec.hidden_size,
            kernel_size=spec.patch_size,
            stride=spec.patch_size,
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, spec.num_patches + 1, spec.hidden_size))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch = pixel_values.shape[0]
        patches = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        cls = self.class_embedding.expand(batch, -1, -1)
        tokens = torch.cat([cls, patches], dim=1)
        return tokens + self.position_embedding[:, : tokens.shape[1]].to(tokens.dtype)


class InternViTAttention(nn.Module):
    def __init__(self, spec: InternViTSpec) -> None:
        super().__init__()
        self.spec = spec
        self.num_heads = spec.num_attention_heads
        self.head_dim = spec.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv = nn.Linear(spec.hidden_size, spec.hidden_size * 3, bias=spec.qkv_bias)
        self.proj = nn.Linear(spec.hidden_size, spec.hidden_size, bias=True)
        if spec.qk_normalization:
            self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
            self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = hidden_states.shape
        qkv = self.qkv(hidden_states).view(bsz, seqlen, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        if self.spec.qk_normalization:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(1, 2).reshape(bsz, seqlen, -1)
        return self.proj(out)


class InternViTMLP(nn.Module):
    def __init__(self, spec: InternViTSpec) -> None:
        super().__init__()
        self.fc1 = nn.Linear(spec.hidden_size, spec.intermediate_size, bias=True)
        self.fc2 = nn.Linear(spec.intermediate_size, spec.hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class InternViTLayer(nn.Module):
    def __init__(self, spec: InternViTSpec) -> None:
        super().__init__()
        self.norm1 = _build_norm(spec)
        self.attn = InternViTAttention(spec)
        self.norm2 = _build_norm(spec)
        self.mlp = InternViTMLP(spec)
        self.ls1 = nn.Parameter(torch.ones(spec.hidden_size))
        self.ls2 = nn.Parameter(torch.ones(spec.hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1 * self.attn(self.norm1(x))
        x = x + self.ls2 * self.mlp(self.norm2(x))
        return x


class InternViTEncoder(nn.Module):
    def __init__(self, spec: InternViTSpec) -> None:
        super().__init__()
        self.layers = nn.ModuleList([InternViTLayer(spec) for _ in range(spec.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        select_layer: int,
    ) -> torch.Tensor:
        # ``select_layer`` follows the InternVL convention: -1 means take the
        # last layer's output; positive indices select an explicit layer.
        if select_layer < 0:
            select_layer = len(self.layers) + select_layer
        for idx, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states)
            if idx == select_layer:
                break
        return hidden_states


class InternVisionModel(nn.Module):
    """GO-1-Air vision tower; output keeps the ``[CLS]`` token (consumer drops it)."""

    def __init__(self, spec: InternViTSpec) -> None:
        super().__init__()
        self.spec = spec
        self.embeddings = InternViTPatchEmbeddings(spec)
        self.encoder = InternViTEncoder(spec)

    def forward(self, pixel_values: torch.Tensor, select_layer: int = -1) -> torch.Tensor:
        tokens = self.embeddings(pixel_values)
        return self.encoder(tokens, select_layer=select_layer)


def pixel_shuffle(features: torch.Tensor, scale: float, version: str = "v2") -> torch.Tensor:
    """Spatial-to-channel shuffle used to compress vision tokens into the LLM stream.

    For ``scale=0.5`` this groups every 2x2 patch block into a single token
    while quadrupling the channel dimension. Mirrors the InternVL ``ps_v2``
    layout (the input tile order is rotated before reshape so adjacent
    patches stay adjacent in memory).
    """
    if scale >= 1.0:
        return features
    bsz, num_tokens, channels = features.shape
    side = int(math.isqrt(num_tokens))
    if side * side != num_tokens:
        raise ValueError(f"pixel_shuffle expects a square token grid; got {num_tokens}.")
    new_side = int(side * scale)
    block = int(round(1.0 / scale))
    grid = features.view(bsz, side, side, channels)
    if version == "v2":
        grid = grid.permute(0, 2, 1, 3).contiguous()
    grid = grid.view(bsz, side, new_side, channels * block)
    grid = grid.permute(0, 2, 1, 3).contiguous()
    grid = grid.view(bsz, new_side, new_side, channels * (block * block))
    if version == "v2":
        grid = grid.permute(0, 2, 1, 3).contiguous()
    return grid.view(bsz, new_side * new_side, channels * (block * block))
