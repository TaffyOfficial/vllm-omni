# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GO-1-Air policy and top-level model.

The submodule layout mirrors the upstream safetensors index so a
``model.load_state_dict`` call lines up directly: ``vision_model.*``,
``mlp1.*``, ``language_model.*``, ``action_model.*``, ``k_proj_layers.*``,
``v_proj_layers.*``, ``time_embedder.*``, ``freq_embedder.*``,
``state_adaptor.*``, ``action_adaptor.*``, ``final_layer.*`` are all direct
children of :class:`Go1Air`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn
from vllm.logger import init_logger

from .action_expert import (
    ActionExpertBlock,
    FinalLayer,
    TimestepEmbedder,
    action_block_spec,
    make_state_action_adaptor,
    squared_cos_cap_v2_alpha_bar,
)
from .config import OBS_IMAGES, OBS_STATE, OBS_TASK, Go1AirConfig
from .internlm2_layers import InternLM2RMSNorm, InternLM2RotaryEmbedding
from .language_internlm2_go1 import InternLM2GO1LanguageModel
from .vision_internvit import InternVisionModel, InternViTSpec, pixel_shuffle

logger = init_logger(__name__)


def _build_mlp1(vision_dim: int, scale: float, llm_hidden: int) -> nn.Sequential:
    """Vision-to-LLM projector matching the safetensors layout (LN, Linear, GELU, Linear)."""
    inv_scale_sq = int(round(1.0 / scale)) ** 2
    projector_in = vision_dim * inv_scale_sq
    return nn.Sequential(
        nn.LayerNorm(projector_in),
        nn.Linear(projector_in, llm_hidden, bias=True),
        nn.GELU(),
        nn.Linear(llm_hidden, llm_hidden, bias=True),
    )


class Go1Air(nn.Module):
    """Top-level GO-1-Air model.

    Forward path: vision_model -> pixel_shuffle -> mlp1 -> inject into LLM
    embeddings at ``img_context_token_id`` -> language_model (collect layer KV)
    -> action expert (state/action/time/freq embeddings + 24 cross-attention
    blocks + final_layer) -> diffusion sampling loop -> ``[B, chunk, action_dim]``.
    """

    def __init__(self, config: Go1AirConfig) -> None:
        super().__init__()
        self.config = config

        vision_spec = InternViTSpec.from_config(config)
        self.vision_model = InternVisionModel(vision_spec)
        self.mlp1 = _build_mlp1(
            vision_dim=config.vision_hidden_size,
            scale=config.downsample_ratio,
            llm_hidden=config.llm_hidden_size,
        )
        self.language_model = InternLM2GO1LanguageModel(config)

        spec = action_block_spec(config)
        adaptor_hidden = config.act_hidden_size
        self.state_adaptor = make_state_action_adaptor(
            in_dim=config.max_state_dim,
            hidden_dim=adaptor_hidden,
            out_dim=config.act_hidden_size,
        )
        self.action_adaptor = make_state_action_adaptor(
            in_dim=config.max_action_dim,
            hidden_dim=adaptor_hidden,
            out_dim=config.act_hidden_size,
        )
        self.time_embedder = TimestepEmbedder(config.act_hidden_size)
        self.freq_embedder = TimestepEmbedder(config.act_hidden_size)

        self.action_model = nn.ModuleDict(
            {
                "layers": nn.ModuleList(
                    [ActionExpertBlock(spec) for _ in range(config.act_num_hidden_layers)]
                ),
                "norm": InternLM2RMSNorm(config.act_hidden_size, eps=config.act_rms_norm_eps),
            }
        )

        # Per-layer projections from LLM KV space → action expert KV space.
        llm_kv_dim = (config.llm_hidden_size // config.llm_num_attention_heads) * config.llm_num_key_value_heads
        action_kv_dim = spec.resolved_head_dim() * config.act_num_key_value_heads
        self.k_proj_layers = nn.ModuleList(
            [nn.Linear(llm_kv_dim, action_kv_dim, bias=True) for _ in range(config.act_num_hidden_layers)]
        )
        self.v_proj_layers = nn.ModuleList(
            [nn.Linear(llm_kv_dim, action_kv_dim, bias=True) for _ in range(config.act_num_hidden_layers)]
        )

        self.final_layer = FinalLayer(
            hidden_size=config.act_hidden_size,
            intermediate_size=config.act_intermediate_size,
            action_dim=config.max_action_dim,
            eps=config.act_rms_norm_eps,
        )

        self._action_spec = spec
        self._action_rotary_emb = InternLM2RotaryEmbedding(spec.resolved_head_dim(), spec)
        self.register_buffer(
            "alpha_bar",
            squared_cos_cap_v2_alpha_bar(config.num_train_timesteps, torch.device("cpu")),
            persistent=False,
        )

    # ----- runtime knobs --------------------------------------------------

    def set_attention_implementation(self, implementation: str) -> None:
        self.config.attn_implementation = implementation
        self._action_spec.attn_implementation = implementation
        self.language_model.set_attention_implementation(implementation)
        for block in self.action_model["layers"]:
            block.attention.spec.attn_implementation = implementation

    # ----- vision side ----------------------------------------------------

    def encode_vision(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run vision tower, drop CLS, apply pixel shuffle, project to LLM hidden size."""
        features = self.vision_model(pixel_values, select_layer=self.config.select_layer)
        # Drop the [CLS] token before pixel-shuffling so the patch grid is square.
        patches = features[:, 1:, :]
        compressed = pixel_shuffle(patches, self.config.downsample_ratio, self.config.pixel_shuffle_version)
        return self.mlp1(compressed)

    # ----- language prefix ------------------------------------------------

    def _inject_vision(
        self,
        input_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        vision_features: torch.Tensor,
    ) -> torch.Tensor:
        """Replace embeddings at ``img_context_token_id`` positions with vision features."""
        out = input_embeds.clone()
        flat_mask = (input_ids == self.config.img_context_token_id)
        # vision_features: (B, V, D); ``flat_mask`` selects exactly V positions per batch row.
        flat_features = vision_features.reshape(-1, vision_features.shape[-1]).to(out.dtype)
        out[flat_mask] = flat_features[: flat_mask.sum().item()]
        return out

    def encode_prefix(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """Embed text + inject vision tokens, run LLM, return last hidden state + per-layer KV."""
        embeds = self.language_model.model.tok_embeddings(input_ids)
        if pixel_values is not None:
            vision_features = self.encode_vision(pixel_values)
            embeds = self._inject_vision(embeds, input_ids, vision_features)
        out = self.language_model.model(inputs_embeds=embeds, attention_mask=attention_mask)
        return out.last_hidden_state, out.layer_kv, embeds

    # ----- action expert --------------------------------------------------

    def _project_vlm_kv(
        self,
        layer_idx: int,
        vlm_k: torch.Tensor,
        vlm_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, _, vlm_len, _ = vlm_k.shape
        flat_k = vlm_k.transpose(1, 2).reshape(bsz, vlm_len, -1)
        flat_v = vlm_v.transpose(1, 2).reshape(bsz, vlm_len, -1)
        proj_k = self.k_proj_layers[layer_idx](flat_k)
        proj_v = self.v_proj_layers[layer_idx](flat_v)
        head_dim = self._action_spec.resolved_head_dim()
        proj_k = proj_k.view(bsz, vlm_len, self.config.act_num_key_value_heads, head_dim).transpose(1, 2)
        proj_v = proj_v.view(bsz, vlm_len, self.config.act_num_key_value_heads, head_dim).transpose(1, 2)
        return proj_k, proj_v

    def _build_joint_tokens(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        control_freq: torch.Tensor,
    ) -> torch.Tensor:
        bsz = actions.shape[0]
        state_token = self.state_adaptor(state).unsqueeze(1).expand(bsz, self.config.state_token_num, -1)
        action_tokens = self.action_adaptor(actions)
        time_tok = self.time_embedder(timesteps).unsqueeze(1)
        freq_tok = self.freq_embedder(control_freq).unsqueeze(1)
        return torch.cat([freq_tok, time_tok, state_token, action_tokens], dim=1)

    def predict_clean(
        self,
        actions: torch.Tensor,
        state: torch.Tensor,
        timesteps: torch.Tensor,
        control_freq: torch.Tensor,
        vlm_layer_kv: list[tuple[torch.Tensor, torch.Tensor]],
        vlm_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        joint = self._build_joint_tokens(state, actions, timesteps, control_freq)
        bsz, joint_len, _ = joint.shape
        prefix_len = 2 + self.config.state_token_num
        vlm_len = vlm_layer_kv[0][0].shape[2]

        action_position_ids = torch.arange(joint_len, device=joint.device).expand(bsz, joint_len) + vlm_len
        vlm_position_ids = torch.arange(vlm_len, device=joint.device).expand(bsz, vlm_len)

        attention_mask = self._build_joint_attention_mask(
            joint_len=joint_len,
            vlm_len=vlm_len,
            vlm_attention_mask=vlm_attention_mask,
            device=joint.device,
            dtype=joint.dtype,
        )

        hidden = joint
        for idx, block in enumerate(self.action_model["layers"]):
            proj_k, proj_v = self._project_vlm_kv(idx, vlm_layer_kv[idx][0], vlm_layer_kv[idx][1])
            hidden = block(
                hidden,
                action_position_ids,
                vlm_position_ids,
                self._action_rotary_emb,
                proj_k,
                proj_v,
                attention_mask,
            )

        hidden = self.action_model["norm"](hidden)
        action_hidden = hidden[:, prefix_len:, :]
        return self.final_layer(action_hidden)

    def _build_joint_attention_mask(
        self,
        *,
        joint_len: int,
        vlm_len: int,
        vlm_attention_mask: torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if vlm_attention_mask is None:
            return None
        bsz = vlm_attention_mask.shape[0]
        total_kv = vlm_len + joint_len
        neg = torch.finfo(dtype).min
        mask = torch.zeros((bsz, 1, joint_len, total_kv), device=device, dtype=dtype)
        invalid = (vlm_attention_mask == 0).view(bsz, 1, 1, vlm_len).expand(bsz, 1, joint_len, vlm_len)
        mask[..., :vlm_len] = mask[..., :vlm_len].masked_fill(invalid, neg)
        return mask

    @torch.inference_mode()
    def sample_actions(
        self,
        state: torch.Tensor,
        control_freq: torch.Tensor,
        vlm_layer_kv: list[tuple[torch.Tensor, torch.Tensor]],
        vlm_attention_mask: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz = state.shape[0]
        device = state.device
        dtype = state.dtype if state.is_floating_point() else torch.float32
        shape = (bsz, self.config.chunk_size, self.config.max_action_dim)
        if noise is None:
            noise = torch.randn(shape, device=device, dtype=dtype)
        x = noise.to(device=device, dtype=dtype)

        alpha_bar = self.alpha_bar.to(device=device)
        train_steps = self.config.num_train_timesteps
        infer_steps = self.config.num_inference_steps
        idx = torch.linspace(train_steps - 1, 0, infer_steps, device=device).round().long()

        for i in range(infer_steps):
            t = idx[i].expand(bsz)
            x0_pred = self.predict_clean(
                x,
                state,
                t,
                control_freq,
                vlm_layer_kv,
                vlm_attention_mask,
            ).to(dtype)
            if i < infer_steps - 1:
                t_next = idx[i + 1]
                a_t = alpha_bar[t[0]].clamp(min=1e-8)
                a_next = alpha_bar[t_next].clamp(min=1e-8)
                eps_pred = (x - a_t.sqrt() * x0_pred) / (1.0 - a_t).clamp(min=1e-8).sqrt()
                x = a_next.sqrt() * x0_pred + (1.0 - a_next).clamp(min=0.0).sqrt() * eps_pred
            else:
                x = x0_pred

        return x


class Go1AirPolicy(nn.Module):
    """Pipeline-facing wrapper: holds a tokenizer + ``Go1Air`` model."""

    def __init__(
        self,
        config: Go1AirConfig,
        *,
        processor_model_name: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.processor_model_name = processor_model_name
        self.model = Go1Air(config)
        self._tokenizer = None
        self._has_weights = False

    # ----- construction ---------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        config: Go1AirConfig,
        processor_model_name: str | None = None,
        strict: bool = False,
    ) -> Go1AirPolicy:
        policy = cls(config, processor_model_name=processor_model_name)
        policy._load_weights(Path(model_dir), strict=strict)
        policy._maybe_load_tokenizer(Path(model_dir))
        return policy

    def _load_weights(self, model_dir: Path, *, strict: bool) -> None:
        index_path = model_dir / "model.safetensors.index.json"
        single_path = model_dir / "model.safetensors"
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError("safetensors is required to load GO-1-Air checkpoints.") from exc

        state_dict: dict[str, torch.Tensor] = {}
        if index_path.exists():
            with open(index_path, encoding="utf-8") as f:
                index = json.load(f)
            shards = sorted(set(index["weight_map"].values()))
            for shard in shards:
                state_dict.update(load_file(str(model_dir / shard)))
        elif single_path.exists():
            state_dict.update(load_file(str(single_path)))
        else:
            logger.warning("Go1AirPolicy: no safetensors found under %s; running uninitialised.", model_dir)
            return

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        self._has_weights = True
        if missing:
            logger.warning("Go1AirPolicy: %d missing weights (showing first 5): %s", len(missing), missing[:5])
        if unexpected:
            logger.warning("Go1AirPolicy: %d unexpected weights (showing first 5): %s", len(unexpected), unexpected[:5])
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"Go1AirPolicy strict load failed: {len(missing)} missing, {len(unexpected)} unexpected."
            )

    def _maybe_load_tokenizer(self, model_dir: Path) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError:
            logger.warning("transformers not available; GO-1-Air policy will run without a tokenizer.")
            return
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(model_dir),
                trust_remote_code=True,
            )
        except Exception as exc:
            logger.warning("Go1AirPolicy: tokenizer load failed (%s); falling back to stub mode.", exc)
            self._tokenizer = None

    # ----- forward --------------------------------------------------------

    def forward(
        self,
        batch_inputs: dict[str, Any],
        *,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz = self._infer_batch_size(batch_inputs)
        device = self._infer_device(batch_inputs)
        dtype = getattr(torch, self.config.dtype) if isinstance(self.config.dtype, str) else self.config.dtype

        # Stub path: no weights loaded → return zero actions of the right shape.
        # Keeps pipeline plumbing exercisable end-to-end without a checkpoint.
        if not self._has_weights:
            return torch.zeros(
                (bsz, self.config.chunk_size, self.config.max_action_dim),
                device=device,
                dtype=noise.dtype if noise is not None else torch.float32,
            )

        state = batch_inputs.get(OBS_STATE)
        if not isinstance(state, torch.Tensor):
            raise ValueError(f"Go1AirPolicy expects '{OBS_STATE}' as a tensor in batch_inputs.")

        pixel_values, vlm_attention_mask, input_ids = self._build_llm_inputs(batch_inputs, device=device, dtype=dtype)
        _, layer_kv, _ = self.model.encode_prefix(input_ids, pixel_values, vlm_attention_mask)

        control_freq = batch_inputs.get("control_freq")
        if not isinstance(control_freq, torch.Tensor):
            control_freq = torch.full((bsz,), 30.0, device=device, dtype=torch.float32)

        return self.model.sample_actions(
            state=state,
            control_freq=control_freq,
            vlm_layer_kv=layer_kv,
            vlm_attention_mask=vlm_attention_mask,
            noise=noise,
        )

    # ----- input plumbing -------------------------------------------------

    def _infer_batch_size(self, batch_inputs: dict[str, Any]) -> int:
        state = batch_inputs.get(OBS_STATE)
        if isinstance(state, torch.Tensor):
            return int(state.shape[0])
        for key, value in batch_inputs.items():
            if key.startswith(f"{OBS_IMAGES}.") and not key.endswith("_mask"):
                if isinstance(value, torch.Tensor) and value.ndim >= 1:
                    return int(value.shape[0])
        return 1

    def _infer_device(self, batch_inputs: dict[str, Any]) -> torch.device:
        for value in batch_inputs.values():
            if isinstance(value, torch.Tensor):
                return value.device
        return torch.device(self.config.device)

    def _build_llm_inputs(
        self,
        batch_inputs: dict[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        """Collect pixel_values + tokenize task into input_ids with image-context placeholders."""
        if self._tokenizer is None:
            raise RuntimeError(
                "Go1AirPolicy.forward requires a tokenizer for the language prefix; "
                "load via from_pretrained(model_dir) or run the pipeline in stub mode."
            )

        # Stack all observation.images.* image tensors into a single (B*N, 3, H, W) batch.
        image_tensors: list[torch.Tensor] = []
        for key, value in batch_inputs.items():
            if key.startswith(f"{OBS_IMAGES}.") and not key.endswith("_mask") and isinstance(value, torch.Tensor):
                # Expected shape (B, history, 3, H, W) — flatten history into batch.
                bsz, history = value.shape[:2]
                image_tensors.append(value.reshape(bsz * history, *value.shape[2:]).to(dtype))
        pixel_values = torch.cat(image_tensors, dim=0).to(device=device) if image_tensors else None

        # Tokenize the task strings with one image-context placeholder per image
        # in the batch row. The tokenizer's vocabulary is expected to contain
        # the special id ``img_context_token_id`` (true for InternLM2 tokenizers
        # configured for the InternVL family).
        task_strings = batch_inputs.get(OBS_TASK) or [""]
        if isinstance(task_strings, str):
            task_strings = [task_strings]

        encoded = self._tokenizer(
            list(task_strings),
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        return pixel_values, attention_mask, input_ids
