# `_hi3_ar_hf_baseline.json`

Recorded HF AR-output baseline for the IT2I `test_omni_ar_output_aligns_with_hf_reference_greedy`
regression in `tests/e2e/accuracy/test_hunyuan_image3_it2i_ar_output.py`.

## What it is

Output of running the official HuggingFace HunyuanImage-3.0-Instruct AR path in
greedy mode (`do_sample=False, temperature=0.0`) on a fixed
`(image_seed, prompt, max_new_tokens)` triple, captured offline so the regression
test only needs to bring up vllm-omni's AR stage and compare token sequences.

| field | meaning |
|-------|---------|
| `model` | HF repo id |
| `prompt` | user prompt (str) |
| `image_seed` | numpy seed used to generate the synthetic 512x512 condition image |
| `image_shape` | `[H, W]` of the synthetic image |
| `max_new_tokens` | greedy decode budget |
| `prefill_input_ids` | ids out of `model.prepare_model_inputs(...)["input_ids"][0]` |
| `generated_token_ids` | the AR-generated tokens (greedy) |
| `decoded_text` | `model._tokenizer.decode(generated)` |
| `transformers_version` / `torch_version` | provenance of the recording stack |

## Why a recorded baseline rather than running HF live

vllm-omni and the HF reference want incompatible Python environments:

- vllm-omni: `transformers>=5.x`, `torch>=2.10`, vllm-bundled CUDA libs
- HF reference: `transformers==4.57.1` (the model's own `requirements.txt` pin),
  `torch==2.8.0+cu128`, two manual patches to `modeling_hunyuan_image_3.py`

Trying to satisfy both pins in one venv forces transformers downgrades that
re-introduce other bugs. The project's standing pattern (per
`workflow-starter/memory/hf/hf_omni_alignment_method.md`) is two separate
venvs: `/root/venv_hf` (HF reference) and the omni venv. CI can replicate
either, but doing both in one job is fragile.

The recorded baseline lets the regression test run with just one stack while
still asserting "vllm-omni's AR output aligns with HF reference".

## How to regenerate

On a machine that can run the HF reference (transformers==4.57.1 venv with the
two model patches applied):

```bash
HF_HOME=/path/to/hf_cache /root/venv_hf/bin/python tests/diffusion/models/hunyuan_image3/_regenerate_hi3_ar_hf_baseline.py \
    --output tests/diffusion/models/hunyuan_image3/_hi3_ar_hf_baseline.json
```

The capture script reuses `scripts/bench/bench_ar_hf.py`'s loader pattern;
running it with the same `(prompt, image_seed, max_new_tokens)` produces a
byte-identical JSON.
