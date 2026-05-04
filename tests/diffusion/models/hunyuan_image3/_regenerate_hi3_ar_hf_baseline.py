#!/usr/bin/env python3
"""Regenerate the HF AR-output baseline JSON consumed by
``tests/e2e/accuracy/test_hunyuan_image3_it2i_ar_output.py``.

Run this from a venv with the model's pinned HF stack:

    transformers==4.57.1
    torch==2.8.0+cu128 + torchvision==0.23.0
    diffusers==0.35.2 + einops + accelerate

with the two manual patches applied to the snapshot's
``modeling_hunyuan_image_3.py`` (RoPE broadcast / 2D attention_mask SDPA
fall-through) -- see ``workflow-starter/memory/hf/hf_baseline_runbook.md``.

The fixed ``(prompt, image_seed, image_shape, max_new_tokens)`` are
hard-coded here so re-running on a clean cache produces a byte-identical
JSON for the same model snapshot.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODEL = "tencent/HunyuanImage-3.0-Instruct"
PROMPT = "Describe the content of the picture."
IMAGE_SEED = 42
IMAGE_SHAPE = (512, 512)
MAX_NEW = 64


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "_hi3_ar_hf_baseline.json",
    )
    args = parser.parse_args()

    print("[regen] loading HF model...")
    t0 = time.perf_counter()
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"[regen] loaded in {time.perf_counter() - t0:.1f}s")

    tok_cls = None
    for key, mod in sys.modules.items():
        if "tokenization_hunyuan_image_3" in key and inspect.ismodule(mod):
            if hasattr(mod, "HunyuanImage3TokenizerFast"):
                tok_cls = mod.HunyuanImage3TokenizerFast
                break
    if tok_cls is None:
        raise RuntimeError(
            "HunyuanImage3TokenizerFast not auto-imported with trust_remote_code"
        )
    model._tokenizer = tok_cls.from_pretrained(MODEL)
    model._validate_model_kwargs = lambda kw: None

    rng = np.random.default_rng(seed=IMAGE_SEED)
    image_arr = rng.integers(0, 256, IMAGE_SHAPE + (3,), dtype=np.uint8)
    image = Image.fromarray(image_arr, mode="RGB")

    mi = model.prepare_model_inputs(
        prompt=PROMPT,
        image=image,
        mode="gen_text",
        bot_task="think",
        max_new_tokens=MAX_NEW,
    )
    prefill = mi["input_ids"][0].cpu().tolist()
    print(f"[regen] prefill len: {len(prefill)}")

    with torch.no_grad():
        out = model.generate(
            do_sample=False,
            temperature=0.0,
            decode_text=False,
            verbose=0,
            **mi,
        )

    if isinstance(out, torch.Tensor):
        full = out[0].cpu().tolist()
    elif isinstance(out, list) and out:
        head = out[0]
        full = head.tolist() if hasattr(head, "tolist") else list(head)
    else:
        raise RuntimeError(f"Unexpected HF generate return type: {type(out)}")
    generated = full[len(prefill):]
    decoded = model._tokenizer.decode(generated, skip_special_tokens=False)
    print(f"[regen] generated {len(generated)} tokens")

    import transformers as _tx

    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "image_seed": IMAGE_SEED,
        "image_shape": list(IMAGE_SHAPE),
        "max_new_tokens": MAX_NEW,
        "prefill_input_ids": prefill,
        "generated_token_ids": generated,
        "decoded_text": decoded,
        "transformers_version": _tx.__version__,
        "torch_version": torch.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload), encoding="utf-8")
    print(f"[regen] wrote {args.output}")
    print(f"[regen] decoded[:200]: {decoded[:200]!r}")


if __name__ == "__main__":
    main()
