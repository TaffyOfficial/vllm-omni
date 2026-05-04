# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify the IT2I AR-prefill prompt matches the official HF chat-template output.

PR #3107 builds the AR prefill via
:func:`vllm_omni.diffusion.models.hunyuan_image3.prompt_utils.build_prompt_tokens`,
which segment-tokenizes the canonical Instruct chat template (`<|startoftext|>`
+ `{system}\\n\\n` + `User: [<img>]{user_prompt}\\n\\nAssistant: {trigger?}`).

The official HunyuanImage-3.0-Instruct repo ships a Jinja `chat_template` in
its tokenizer config and an `image_processor.py` whose `process_image`
defines the same VAE/VIT preprocessing the diffusion pipeline uses on the
condition image. To prevent silent drift between the AR's input distribution
and what the model was actually trained on, this test asserts:

1. ``build_prompt_tokens`` token-id sequence equals the HF reference produced
   by ``tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)``
   for the same `(system, user_prompt, image)` triple.
2. The image-tensor produced by the diffusion-side ``_resize_and_crop_center``
   is byte-identical to the AR-side ``HunyuanImage3Processor._resize_and_crop``
   output (i.e. AR and DiT preprocess the IT2I condition image identically).

Both checks need the official tokenizer/image-processor classes; we gate on
``HF_HOME`` cache availability so the suite stays runnable on machines
without the model weights.
"""

from __future__ import annotations

import os
import pathlib

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


_HUNYUAN_MODEL_ID = "tencent/HunyuanImage-3.0-Instruct"


def _hf_cached(model_id: str) -> bool:
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    snap_dir = os.path.join(hf_home, "hub", f"models--{model_id.replace('/', '--')}", "snapshots")
    return os.path.isdir(snap_dir) and any(os.scandir(snap_dir))


def _snapshot_dir(model_id: str) -> pathlib.Path:
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    snap_root = pathlib.Path(hf_home) / "hub" / f"models--{model_id.replace('/', '--')}" / "snapshots"
    snap = next(iter(snap_root.iterdir()))
    return snap


@pytest.mark.skipif(
    not _hf_cached(_HUNYUAN_MODEL_ID),
    reason=f"{_HUNYUAN_MODEL_ID} not in HF cache",
)
def test_ar_prefill_tokens_match_official_hf_apply_chat_template_for_it2i():
    """The AR-side prefill we build must be token-id-identical to what
    the official HunyuanImage3TokenizerFast's ``apply_chat_template`` emits
    for the same (system, user_prompt, image) triple under
    ``sequence_template="instruct"`` + ``bot_task="think"``.

    The Instruct template canonical contract for IT2I (``it2i_think``):

      [BOS] + system_prompt + "\\n\\n" + "User: " + <img> + user_prompt
      + "\\n\\nAssistant: " + <think>

    This is what the model was trained on; our segment-by-segment
    ``build_prompt_tokens`` (PR #3243) was hand-derived to match. If
    upstream changes the canonical template (extra space, BOS dropped,
    trigger moved) or our segment list drifts, the AR prefill no longer
    matches the model's training distribution and IT2I output collapses
    into repetition garbage -- same failure mode PR #3243 fixed for T2I.

    We bypass HF's generic ``apply_chat_template`` (the model's
    tokenizer config has no ``chat_template`` field; the canonical
    template lives on the custom ``HunyuanImage3TokenizerFast`` class loaded
    via ``trust_remote_code``). The custom method has a non-standard
    signature -- batch lists, ``sequence_template`` switch, ``mode``,
    ``bot_task`` -- so we adapt the call site rather than using
    ``apply_chat_template(messages, ...)`` directly.
    """
    from transformers import AutoTokenizer

    from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import (
        _TASK_PRESETS,
        build_prompt_tokens,
    )
    from vllm_omni.diffusion.models.hunyuan_image3.system_prompt import get_system_prompt

    user_prompt = "Add a cute orange cat sitting in the foreground."
    task = "it2i_think"

    # The standard tokenizer (used to encode our segments) and the custom
    # HunyuanImage3TokenizerFast (used to render the canonical Instruct
    # template) are loaded separately. AutoTokenizer with trust_remote_code
    # returns the standard fast tokenizer because the model's
    # ``tokenizer_config.json`` declares ``tokenizer_class:
    # PreTrainedTokenizerFast`` -- the custom class is shipped as a
    # standalone module rather than via auto-mapping.
    fast_tok = AutoTokenizer.from_pretrained(_HUNYUAN_MODEL_ID, trust_remote_code=True)
    ours = build_prompt_tokens(user_prompt, fast_tok, task=task)

    preset_sys_type, preset_bot_task, _ = _TASK_PRESETS[task]
    sys_text = get_system_prompt(preset_sys_type, preset_bot_task, None) or ""

    tok_mod, _img_mod = _import_official_snapshot_modules()
    if tok_mod is None or not hasattr(tok_mod, "HunyuanImage3TokenizerFast"):
        pytest.skip("Official HunyuanImage3TokenizerFast module not loadable")

    snap = _snapshot_dir(_HUNYUAN_MODEL_ID)
    try:
        official_tok = tok_mod.HunyuanImage3TokenizerFast.from_pretrained(
            str(snap),
            sequence_template="instruct",
        )
    except Exception as exc:
        pytest.skip(f"Official HunyuanImage3TokenizerFast.from_pretrained failed: {exc}")

    try:
        out = official_tok.apply_chat_template(
            batch_prompt=[user_prompt],
            batch_system_prompt=[sys_text],
            mode="gen_text",
            bot_task="think",
            sequence_template="instruct",
            cfg_factor=1,
            add_assistant_prefix=True,
        )
    except Exception as exc:
        pytest.skip(
            f"Official apply_chat_template raised at runtime: {exc}"
        )

    # ``apply_chat_template`` returns ``{"output": <SpecialOutput>, "sections": ...}``
    # where ``SpecialOutput.tokens`` is a tensor of shape
    # ``(batch * cfg_factor, seq_len)``. Extract the single-batch list.
    if isinstance(out, dict):
        output_obj = out.get("output", out)
    else:
        output_obj = out
    tokens = getattr(output_obj, "tokens", None)
    if tokens is None and isinstance(out, dict):
        tokens = out.get("input_ids")
    if tokens is None:
        pytest.skip(
            "apply_chat_template returned an object without `tokens`/`input_ids`"
        )
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    if tokens and isinstance(tokens[0], list):
        hf_ids = list(tokens[0])
    else:
        hf_ids = list(tokens)

    # The official template includes the `<img>` placeholder once for the
    # condition image; our `build_prompt_tokens(task='it2i_think')` also
    # inserts a single `<img>` slot for the IT2I case. Both lists should
    # therefore align directly without padding.
    assert ours == hf_ids, (
        "AR-prefill token-id sequence drifted from official apply_chat_template:\n"
        f"  ours length={len(ours)}, hf length={len(hf_ids)}\n"
        f"  first divergent index="
        f"{next((i for i, (a, b) in enumerate(zip(ours, hf_ids)) if a != b), 'tail-diff')}\n"
        "  -> the IT2I AR prefill no longer matches the model's training "
        "  distribution (same class of bug PR #3243 fixed for T2I)."
    )


_OFFICIAL_PKG = "_hunyuan_image_3_official_snapshot"


def _import_official_snapshot_modules():
    """Register the HunyuanImage-3.0-Instruct snapshot as a fake package so
    its ``image_processor.py`` (which does ``from .tokenization_hunyuan_image_3
    import ...``) can be loaded with relative imports intact.

    Returns ``(tokenization_module, image_processor_module)`` or ``(None, None)``
    if either fails (e.g. snapshot missing, optional dep like diffusers absent).
    """
    import importlib.util
    import sys
    import types

    if _OFFICIAL_PKG in sys.modules:
        pkg = sys.modules[_OFFICIAL_PKG]
        return (
            sys.modules.get(f"{_OFFICIAL_PKG}.tokenization_hunyuan_image_3"),
            sys.modules.get(f"{_OFFICIAL_PKG}.image_processor"),
        )

    snap = _snapshot_dir(_HUNYUAN_MODEL_ID)
    if not (snap / "image_processor.py").is_file():
        return None, None

    pkg = types.ModuleType(_OFFICIAL_PKG)
    pkg.__path__ = [str(snap)]
    sys.modules[_OFFICIAL_PKG] = pkg

    def _load(name: str):
        full = f"{_OFFICIAL_PKG}.{name}"
        spec = importlib.util.spec_from_file_location(full, snap / f"{name}.py")
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception:
            del sys.modules[full]
            return None
        return mod

    tok_mod = _load("tokenization_hunyuan_image_3")
    if tok_mod is None:
        return None, None
    img_mod = _load("image_processor")
    return tok_mod, img_mod


@pytest.mark.skipif(
    not _hf_cached(_HUNYUAN_MODEL_ID),
    reason=f"{_HUNYUAN_MODEL_ID} not in HF cache",
)
def test_dit_condition_image_preprocessing_byte_matches_official_hf():
    """The diffusion pipeline's ``_resize_and_crop_center`` (used to feed
    the VAE encoder for IT2I conditioning) must produce byte-identical
    pixels to the **official** HuggingFace
    ``image_processor.resize_and_crop`` (loaded straight out of the
    HunyuanImage-3.0-Instruct snapshot's bundled ``image_processor.py``)
    at ``crop_type='center'``.

    Bounty-hunter's PR #3107 review flagged that the DiT-side helper had
    drifted from the AR-side processor on rounding boundaries; PR #3107
    commit ``0a7e0e6f`` aligned the DiT helper to the AR-side algorithm.
    AR and DiT both *claim* to mirror the HF reference, so the actual
    contract is "DiT (and AR) match the HF reference verbatim". We
    enforce that contract here by comparing directly to the HF function
    rather than to a sibling vllm-omni copy.
    """
    import numpy as np
    from PIL import Image

    from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
        _resize_and_crop_center,
    )

    _tok_mod, official_module = _import_official_snapshot_modules()
    if official_module is None or not hasattr(official_module, "resize_and_crop"):
        pytest.skip("Official HunyuanImage3 image_processor.py not loadable")
    official_resize_and_crop = official_module.resize_and_crop

    rng = np.random.default_rng(seed=42)
    src_size_pairs = [(640, 1024), (1024, 1024), (1280, 720), (480, 800)]
    target_size_pairs = [(1024, 1024), (1024, 768), (768, 1024)]

    for src_w, src_h in src_size_pairs:
        src_arr = rng.integers(0, 256, size=(src_h, src_w, 3), dtype=np.uint8)
        src = Image.fromarray(src_arr, mode="RGB")
        for tw, th in target_size_pairs:
            ref_out = official_resize_and_crop(
                src,
                target_size=(tw, th),
                resample=Image.Resampling.LANCZOS,
                crop_type="center",
            )
            dit_out = _resize_and_crop_center(src, tw, th)
            assert ref_out.size == dit_out.size == (tw, th), (
                f"size mismatch for src={(src_w, src_h)} target={(tw, th)}: "
                f"hf_official={ref_out.size} dit={dit_out.size}"
            )
            ref_pixels = np.asarray(ref_out)
            dit_pixels = np.asarray(dit_out)
            assert np.array_equal(ref_pixels, dit_pixels), (
                f"DiT condition-image preprocessing diverged from HF "
                f"image_processor.resize_and_crop at src={(src_w, src_h)} "
                f"target={(tw, th)}: max abs diff = "
                f"{int(np.abs(ref_pixels.astype(int) - dit_pixels.astype(int)).max())}"
            )
