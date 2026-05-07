# HunyuanImage-3.0-Instruct (Online Serving)

OpenAI-compatible online serving for HunyuanImage-3.0-Instruct via `vllm serve --omni`.
For offline (Python `Omni()`) usage, see [`examples/offline_inference/hunyuan_image3/`](../../offline_inference/hunyuan_image3/).

## Architecture

HunyuanImage-3.0-Instruct is a two-stage AR + DiT model:

| Stage | Role | Topology choices |
| :---- | :--- | :--------------- |
| Stage 0 (AR / Thinker) | Text + image understanding, emits CoT and latent tokens | TP=2 / 4 |
| Stage 1 (DiT) | Denoising loop + VAE decode | TP=2 / 4 |

Stage 0 sends KV cache to Stage 1 via the SharedMemoryConnector (single-node) so the DiT
conditions on the AR's prefill state without re-encoding. Both stages support all four
modalities: `text2img`, `img2img`, `img2text`, `text2text` (the latter two skip stage 1).

## GPU Topology Configs

| Config YAML | Stages | GPUs | Notes |
| :---------- | :----- | :--- | :---- |
| `hunyuan_image3_t2i.yaml` | DiT only | 4 | Pairs with external AR for development |
| `hunyuan_image3_moe.yaml` | AR + DiT | 8 (4+4) | Default; verified on 8x L40S 48GB |
| `hunyuan_image3_t2i_4gpu.yaml` | AR + DiT | 4 (2+2) | Verified on 4x L20X 144GB |
| `hunyuan_image3_t2i_2gpu.yaml` | AR only | 2 | For comprehension benchmarks (not full t2i) |
| `hunyuan_image3_moe_dit_2gpu_fp8.yaml` | DiT only | 2 (FP8) | DiT-side, requires external AR; for 2x H200 |

All YAMLs live under `vllm_omni/model_executor/stage_configs/`.

## Launch the Server

### Quick start (8-GPU default)

```bash
cd examples/online_serving/hunyuan_image3
bash run_server.sh
```

### 4-GPU layout (2 AR + 2 DiT)

```bash
STAGE_CONFIG=$(python -c "import vllm_omni, os; print(os.path.join(os.path.dirname(vllm_omni.__file__), 'model_executor/stage_configs/hunyuan_image3_t2i_4gpu.yaml'))") \
bash run_server.sh
```

### Ada-class GPUs (sm89: L20 / L40 / L40S / RTX 6000 Ada)

vLLM's auto MoE backend picks `flashinfer_cutlass`, which JIT-compiles a sm90-only kernel
and fails on sm89. Force triton:

```bash
MOE_BACKEND=triton bash run_server.sh
```

If you hit `FileNotFoundError: ninja` during model load, install `ninja` in the active
environment (`pip install ninja`) — flashinfer's JIT path needs it even when ultimately
unused. With `MOE_BACKEND=triton` set early enough the JIT path is bypassed entirely.

### Stage-level timing breakdown

Set `ENABLE_PROFILER=1` to enable both `--enable-diffusion-pipeline-profiler` and
`--enable-ar-profiler`. The server then attaches `stage_durations` to each chat
completion response, including `HunyuanImage3Pipeline.model.forward` (DiT total),
`HunyuanImage3Pipeline.vae.decode`, `ar_stage_0`, `stage_0_gen_ms`, `stage_1_gen_ms`,
`queue_wait_ms`, and `preprocess_ms`. Use this with
[`benchmarks/diffusion/diffusion_benchmark_serving.py`](../../../benchmarks/diffusion/diffusion_benchmark_serving.py).

```bash
ENABLE_PROFILER=1 bash run_server.sh
```

### Manual `vllm serve` invocation

```bash
vllm serve tencent/HunyuanImage-3.0-Instruct --omni \
    --port 8091 \
    --stage-configs-path vllm_omni/model_executor/stage_configs/hunyuan_image3_moe.yaml \
    --chat-template examples/online_serving/hunyuan_image3/chat_template.jinja \
    --moe-backend triton    # only on Ada-class GPUs
```

## Why a custom `chat_template.jinja`?

HunyuanImage-3.0-Instruct's HF tokenizer does not ship a `chat_template`. Without one,
`/v1/chat/completions` returns `400 ChatTemplateResolutionError`. The shipped
[`chat_template.jinja`](chat_template.jinja) is a minimal template that emits the
`<|startoftext|>... User: ... Assistant: <think>` sequence expected by the
text-to-image-with-CoT path (`t2i_think` task in `prompt_utils.py`).

It does **not** embed the official `unified_system_prompt_en` literal — that 6 KB
string belongs to the model repo, not vllm-omni. For non-benchmark / quality-sensitive
runs, pass it via a `system` role message (the `--system-prompt-file` flag in
[`openai_chat_client.py`](openai_chat_client.py) wires this up).

For pure-text comprehension (`i2t` / `t2t`) where the `<think>` trigger is wrong,
prefer the offline `end2end.py` path — it builds prompts via
`build_prompt_tokens()` and selects the right trigger per task.

## Send Requests

### Text → Image (text2img)

```bash
python openai_chat_client.py \
    --prompt "A cute cat sitting on a windowsill watching the sunset" \
    --output cat.png \
    --steps 50 \
    --height 1024 --width 1024
```

curl:

```bash
curl http://localhost:8091/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [{"role": "user", "content": "A cute cat sitting on a windowsill"}],
        "modalities": ["image"],
        "height": 1024, "width": 1024,
        "num_inference_steps": 50, "seed": 42
    }' | jq -r '.choices[0].message.content[0].image_url.url' \
       | sed 's/^data:image\/[^;]*;base64,//' | base64 -d > cat.png
```

> **Resolution buckets**: HunyuanImage-3.0 has `image_base_size=1024` hard-coded; any
> request resolution is rounded to the nearest aspect-ratio bucket with area ≈ 1024².
> Sending `width=496, height=496` produces a 1024×1024 image — sub-1024 generations are
> not supported by the model. `width=1280, height=720` rounds to 1280×768.

### Image → Image (img2img)

```bash
python openai_chat_client.py \
    --prompt "Make the petals neon pink" \
    --image-url input.png \
    --output edited.png \
    --modality img2img --steps 50
```

### Image → Text (img2text) and Text → Text (text2text)

For comprehension tasks the `<think>` trigger emitted by the default chat template is
wrong (it forces the AR into chain-of-thought mode). Two options:

1. Use the offline path: `python examples/offline_inference/hunyuan_image3/end2end.py --modality img2text ...`
2. Pass a different chat template to `vllm serve` via `--chat-template` that omits the
   trigger (out of scope for this example).

## Client Arguments

| Argument | Default | Description |
| :------- | :------ | :---------- |
| `--prompt` / `-p` | `A cute cat ...` | Text prompt |
| `--output` / `-o` | `hyimage3_output.png` | Output path (image only) |
| `--server` / `-s` | `http://localhost:8091` | Server URL |
| `--image-url` / `-i` | None | Input image URL or local path (img2img / img2text) |
| `--modality` / `-m` | `text2img` | `text2img` / `img2img` / `img2text` / `text2text` |
| `--height` / `--width` | 1024 | Output image size (rounded to model's aspect-ratio bucket) |
| `--steps` | 50 | Inference steps |
| `--guidance-scale` | 5.0 | CFG scale |
| `--seed` | 42 | Random seed |
| `--negative` | None | Negative prompt |
| `--system-prompt-file` | None | Optional file containing the unified_system_prompt_en text |

## VRAM Footprint

Approximate per-stage VRAM (bf16, no quantization):

| Stage | VRAM (4-GPU 2+2) | VRAM (8-GPU 4+4) |
| :---- | :--------------- | :--------------- |
| Stage 0 (AR) | ~84 GiB / GPU | ~30 GiB / GPU |
| Stage 1 (DiT) | ~83 GiB / GPU | ~30 GiB / GPU |

For 48 GB cards (L40S / RTX 6000 Ada) the 4-GPU layout overflows; use the 8-GPU layout
or enable `--vae-use-tiling`.

## FAQ

- **`ChatTemplateResolutionError`**: pass `--chat-template chat_template.jinja`.
- **`FileNotFoundError: ninja`** during load: `pip install ninja` in the env.
- **`flashinfer cutlass MoE` build failure**: set `MOE_BACKEND=triton` (Ada-class GPUs).
- **OOM on stage 1**: pass `--vae-use-tiling`, lower `gpu_memory_utilization` in the
  stage YAML, or move to a topology with more DiT GPUs.
- **Stage 0 hangs at startup**: in two-process layouts, Stage 0 waits for Stage 1's
  worker to connect — this is expected.
