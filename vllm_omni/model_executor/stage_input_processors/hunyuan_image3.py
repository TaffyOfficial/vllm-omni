# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processor for HunyuanImage3: AR → Diffusion transition.

In IT2I (image editing) mode:
  - Stage 0 (AR) receives (image + edit instruction), generates CoT/latent tokens
  - Stage 1 (DiT) receives the AR output + original image, denoises → edited image

The ar2diffusion function bridges these two stages, following the same
signature pattern as glm_image.ar2diffusion.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniTokensPrompt

logger = init_logger(__name__)

# AR emits `<img_size_BASE><img_ratio_Y>` after `</recaption>` in IT2I/T2I
# (see `HunyuanImage3ForCausalMM.sample` and `_stage_transitions`). The
# ratio_index resolves to a (height, width) bucket via ResolutionGroup, which
# is the official upstream's mechanism for AR-driven output aspect — without
# this lookup the DiT pipeline falls back to the user-provided width/height
# (in the `/v1/images/edits` path that defaults to `pil_images[0].size`,
# i.e. the first reference image's bucket — usually square, see
# api_server.py:1808-1811).
_DEFAULT_HUNYUAN_IMAGE3_MODEL = "tencent/HunyuanImage-3.0-Instruct"


@lru_cache(maxsize=4)
def _build_ratio_size_table(base_size: int) -> list[tuple[int, int]]:
    """Return `[(height, width)]` indexed by ratio_index for HunyuanImage-3.

    Mirrors `HunyuanImage3ImageProcessor.build_image_info`'s
    `reso_group[ratio_index]` reverse lookup. Cached because the table
    is constant per `base_size`.
    """
    from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
        HUNYUAN_IMAGE3_EXTRA_RESOLUTIONS,
        Resolution,
        ResolutionGroup,
    )

    reso_group = ResolutionGroup(
        base_size=base_size,
        extra_resolutions=[Resolution(s) for s in HUNYUAN_IMAGE3_EXTRA_RESOLUTIONS],
    )
    return [(int(r.height), int(r.width)) for r in reso_group.data]


@lru_cache(maxsize=4)
def _build_cot_end_token_ids(model_name_or_path: str) -> dict[str, int]:
    """Return `{'</recaption>': id, '</think>': id}` for cot-boundary
    truncation. Empty dict on lookup failure so callers degrade to a
    pure text-based search.
    """
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    except Exception as e:  # pragma: no cover - environment-dependent
        logger.warning("[ar2diffusion] failed to load tokenizer for cot-end lookup: %s", e)
        return {}

    result: dict[str, int] = {}
    for marker in ("</recaption>", "</think>"):
        tid = tokenizer.convert_tokens_to_ids(marker)
        if tid is not None and tid != tokenizer.unk_token_id:
            result[marker] = int(tid)
    return result


def _truncate_at_cot_end(
    generated_text: str,
    generated_token_ids,
    model_name_or_path: str,
) -> tuple[str, list[int]]:
    """Truncate AR output at first `</recaption>` (or `</think>` fallback).

    Mirrors `HunyuanImage3ForCausalMM.generate_image` in the official
    upstream, which decodes only `generated_tokens[0, :end_pos + 1]` as
    `cot_text` for DiT. The trailing `<answer><boi><img_size_*><img_ratio_*>`
    sequence is a stage-transition trigger consumed via `image_size` /
    height/width — it must NOT be forwarded to DiT's prompt builder, or
    the extra `<boi>` and ratio tokens drift the DiT's own prompt
    structure.
    """
    token_list = list(generated_token_ids) if generated_token_ids is not None else []

    end_ids = _build_cot_end_token_ids(model_name_or_path)

    for marker in ("</recaption>", "</think>"):
        idx = generated_text.find(marker)
        if idx == -1:
            continue
        text_end = idx + len(marker)
        truncated_text = generated_text[:text_end]

        truncated_tokens = token_list
        end_id = end_ids.get(marker)
        if end_id is not None and token_list:
            try:
                token_end = token_list.index(end_id)
                truncated_tokens = token_list[: token_end + 1]
            except ValueError:
                pass
        return truncated_text, truncated_tokens

    return generated_text, token_list


@lru_cache(maxsize=4)
def _build_ratio_id_lookup(model_name_or_path: str) -> dict[int, int]:
    """Return `{token_id: ratio_index}` for `<img_ratio_*>` in the tokenizer.

    Loads the tokenizer once per model path and walks the contiguous
    `<img_ratio_0>..<img_ratio_32>` plus the extra slice
    `<img_ratio_33>..<img_ratio_36>` (the same shape
    `HunyuanImage3ForCausalMM.__init__` registers at lines 1523-1531).
    Empty dict on lookup failure so callers can degrade gracefully.
    """
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    except Exception as e:  # pragma: no cover - environment-dependent
        logger.warning("[ar2diffusion] failed to load tokenizer for ratio token lookup: %s", e)
        return {}

    def _id(name: str) -> int | None:
        tid = tokenizer.convert_tokens_to_ids(name)
        return None if tid is None or tid == tokenizer.unk_token_id else int(tid)

    ratio_0 = _id("<img_ratio_0>")
    ratio_32 = _id("<img_ratio_32>")
    ratio_33 = _id("<img_ratio_33>")
    ratio_36 = _id("<img_ratio_36>")
    if None in (ratio_0, ratio_32, ratio_33, ratio_36):
        logger.warning("[ar2diffusion] tokenizer is missing one of <img_ratio_{0,32,33,36}> tokens")
        return {}

    table: dict[int, int] = {}
    for i in range(ratio_32 - ratio_0 + 1):
        table[ratio_0 + i] = i
    base_idx = ratio_32 - ratio_0 + 1
    for j in range(ratio_36 - ratio_33 + 1):
        table[ratio_33 + j] = base_idx + j
    return table


def _extract_ratio_index(generated_token_ids, model_name_or_path: str) -> int | None:
    """Resolve the AR-predicted ratio_index from this stage's output.

    `HunyuanImage3ForCausalMM`'s `_stage_transitions` forces the AR to emit
    exactly one `<img_ratio_*>` token after `</recaption><answer><boi>
    <img_size_*>`, so we scan the token stream from the tail for the first
    id that maps to a ratio. Token-ids are the source of truth — text-side
    regex is unreliable because most deploy yamls run AR with
    `skip_special_tokens: True` (special tokens are stripped from text but
    still present in `cumulative_token_ids`).
    """
    if generated_token_ids is None:
        return None
    table = _build_ratio_id_lookup(model_name_or_path)
    if not table:
        return None
    for tid in reversed(list(generated_token_ids)):
        idx = table.get(int(tid))
        if idx is not None:
            return idx
    return None


def ar2diffusion(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | list | None = None,
    requires_multimodal_data: bool = False,
) -> list[dict[str, Any]]:
    """Process AR stage outputs to create Diffusion stage inputs.

    Args:
        prompt: Original user prompt (may contain multimodal data).
        requires_multimodal_data: Whether to forward multimodal data.

    Returns:
        List of dicts, each consumable by the HunyuanImage3 diffusion pipeline.
    """
    ar_outputs = source_outputs
    diffusion_inputs = []

    # Normalize prompt to list
    if not isinstance(prompt, list):
        prompt = [prompt] if prompt is not None else [{}]

    for i, ar_output in enumerate(ar_outputs):
        output = ar_output.outputs[0]
        generated_token_ids = output.cumulative_token_ids
        generated_text = getattr(output, "text", "") or ""

        # Get original prompt info
        original_prompt = prompt[i] if i < len(prompt) else {}
        if isinstance(original_prompt, dict):
            pass
        elif hasattr(original_prompt, "_asdict"):
            original_prompt = original_prompt._asdict()
        elif hasattr(original_prompt, "__dict__"):
            original_prompt = vars(original_prompt)
        else:
            original_prompt = {}

        height = original_prompt.get("height", 1024)
        width = original_prompt.get("width", 1024)
        text_prompt = original_prompt.get("prompt", "")
        use_system_prompt = original_prompt.get("use_system_prompt")
        custom_system_prompt = original_prompt.get("system_prompt")

        # Prefer the AR's predicted output aspect (`<img_size_*><img_ratio_*>`
        # tail emitted by `HunyuanImage3ForCausalMM.sample` under the
        # ratio-restriction logits processor) over the carried-through
        # height/width, which the serving layer fills with the first
        # reference image's bucket and so collapses non-square targets to
        # square in the multi-image / mismatched-aspect case. Mirrors the
        # official upstream where `reso_group[ratio_index]` is the
        # canonical source of the diffusion target shape.
        model_name_or_path = original_prompt.get("model") or os.environ.get(
            "VLLM_OMNI_HUNYUAN_IMAGE3_MODEL", _DEFAULT_HUNYUAN_IMAGE3_MODEL
        )
        ratio_idx = _extract_ratio_index(generated_token_ids, model_name_or_path)
        ar_predicted = False
        if ratio_idx is not None:
            base_size = int(original_prompt.get("image_base_size", 1024))
            size_table = _build_ratio_size_table(base_size)
            if 0 <= ratio_idx < len(size_table):
                height, width = size_table[ratio_idx]
                ar_predicted = True
            else:
                logger.warning(
                    "[ar2diffusion] Request %d: ratio_index=%d out of range [0,%d), keeping prompt size %dx%d",
                    i,
                    ratio_idx,
                    len(size_table),
                    height,
                    width,
                )

        # Truncate the AR output at `</recaption>` (or `</think>`) before
        # passing to DiT. Mirrors official `generate_image` which keeps
        # `cot_text` clean and routes size/ratio via `image_size` only —
        # we already extracted `ratio_idx` above and translated it into
        # `height` / `width`, so the `<answer><boi><img_size_*><img_ratio_*>`
        # tail has no remaining job and would only contaminate DiT's
        # prompt builder if forwarded.
        cot_text_for_dit, cot_token_ids_for_dit = _truncate_at_cot_end(
            generated_text, generated_token_ids, model_name_or_path
        )

        logger.info(
            "[ar2diffusion] Request %d: AR generated %d tokens, text length=%d, "
            "cot_text length=%d, target size=%dx%d (%s)",
            i,
            len(generated_token_ids),
            len(generated_text),
            len(cot_text_for_dit),
            height,
            width,
            f"AR ratio_idx={ratio_idx}" if ar_predicted else "from prompt (no AR ratio token)",
        )

        token_tensor = torch.tensor(cot_token_ids_for_dit, dtype=torch.long)

        diffusion_input: dict[str, Any] = {
            "prompt": text_prompt,
            "height": height,
            "width": width,
            "extra": {
                "ar_token_ids": token_tensor,
                "ar_generated_text": cot_text_for_dit,
            },
        }

        # Forward use_system_prompt so the DiT can build the same system prefix.
        # Also forward the custom system prompt body when sys_type=custom so
        # DiT's `get_system_prompt(use, "image", body)` doesn't fall back to
        # an empty prefix and silently diverge from AR.
        if use_system_prompt is not None:
            diffusion_input["use_system_prompt"] = use_system_prompt
        if custom_system_prompt is not None:
            diffusion_input["system_prompt"] = custom_system_prompt

        # Forward multimodal data (original image for IT2I conditioning).
        # The diffusion pre_process_func reads multi_modal_data["image"], which
        # matches vLLM's standard prompt schema, so we only need to pass it once.
        mm_data = original_prompt.get("multi_modal_data")
        if mm_data:
            prompt_images = mm_data.get("image")
            if prompt_images is None:
                prompt_images = mm_data.get("images")
            if prompt_images is not None:
                diffusion_input["multi_modal_data"] = {"image": prompt_images}

        # Forward multimodal output from AR (if any)
        if hasattr(ar_output, "multimodal_output") and ar_output.multimodal_output:
            mm_output = ar_output.multimodal_output
            if isinstance(mm_output, dict):
                diffusion_input["extra"]["ar_multimodal_output"] = mm_output

        # Forward sampling params
        for key in ["seed", "num_inference_steps", "guidance_scale", "negative_prompt"]:
            if key in original_prompt:
                diffusion_input[key] = original_prompt[key]

        diffusion_inputs.append(diffusion_input)

    return diffusion_inputs
