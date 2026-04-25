# HunyuanImage-3.0-Instruct DiT Image Generation on 4x GPU

> DiT-only text-to-image recipe for HunyuanImage-3.0-Instruct with FP8,
> tensor parallelism, sequence parallelism, and CFG parallelism.

## Summary

- Vendor: Tencent Hunyuan
- Model: `tencent/HunyuanImage-3.0-Instruct`
- Task: Text-to-image generation
- Mode: Online serving and performance benchmarking, DiT stage only
- Maintainer: Community

## When to use this recipe

Use this recipe when you want to run or benchmark the HunyuanImage-3.0 DiT
stage directly. This is the recommended first setup when validating DiT
throughput, memory, FP8 kernels, sequence parallelism, or CFG parallelism before
adding the autoregressive HunyuanImage front-end stage.

The recipe covers three 4-GPU configurations:

| Configuration | Parallelism | Notes |
| --- | --- | --- |
| `tp4_fp8` | TP=4 | Lowest per-GPU memory, higher communication overhead |
| `tp2_fp8_sp2` | TP=2, SP=2, Ulysses=2 | Splits sequence work across two GPUs per TP group |
| `tp2_fp8_cfgp2` | TP=2, CFG=2 | Runs CFG branches in parallel; fastest validated DiT setup |

## References

- Model: <https://huggingface.co/tencent/HunyuanImage-3.0-Instruct>
- Offline example:
  [`examples/offline_inference/hunyuan_image3`](../../examples/offline_inference/hunyuan_image3)
- Stage config docs:
  [`docs/configuration/stage_configs.md`](../../docs/configuration/stage_configs.md)
- Performance benchmark configs:
  [`tests/dfx/perf/tests/test_hunyuan_image_tp4_fp8.json`](../../tests/dfx/perf/tests/test_hunyuan_image_tp4_fp8.json),
  [`tests/dfx/perf/tests/test_hunyuan_image_tp2_fp8_sp2.json`](../../tests/dfx/perf/tests/test_hunyuan_image_tp2_fp8_sp2.json),
  [`tests/dfx/perf/tests/test_hunyuan_image_tp2_fp8_cfgp2.json`](../../tests/dfx/perf/tests/test_hunyuan_image_tp2_fp8_cfgp2.json)
- Related PRs:
  [#2495](https://github.com/vllm-project/vllm-omni/pull/2495) for DiT performance CI,
  [#3055](https://github.com/vllm-project/vllm-omni/pull/3055) for GEBench accuracy CI,
  [#3104](https://github.com/vllm-project/vllm-omni/pull/3104) for the T2I L3 dummy guard, and
  [#2949](https://github.com/vllm-project/vllm-omni/pull/2949) for full AR-to-DiT KV reuse.

## Hardware Support

## GPU

### 4x H100/H800 80GB

#### Environment

- OS: Linux
- Python: 3.10+
- Driver / runtime: CUDA-capable runtime matching the repository build
- vLLM version: Match the repository requirements for your checkout
- vLLM-Omni version or commit: Use the commit you are deploying from
- Optional environment variables:

```bash
export CACHE_DIT_VERSION=1.3.0
```

HunyuanImage-3.0 sets the diffusion attention backend to `TORCH_SDPA`
internally because the model mixes causal and full attention.

#### Stage Configs

Create one of the following stage config files and pass it to
`--stage-configs-path`.

**TP=4 + FP8**

```yaml
# hunyuan_image3_dit_tp4_fp8.yaml
stage_args:
  - stage_id: 0
    stage_type: diffusion
    engine_args:
      enforce_eager: true
      distributed_executor_backend: mp
      quantization: fp8
      parallel_config:
        tensor_parallel_size: 4
    final_output: true
    final_output_type: image
```

**TP=2 + FP8 + SP=2**

```yaml
# hunyuan_image3_dit_tp2_fp8_sp2.yaml
stage_args:
  - stage_id: 0
    stage_type: diffusion
    engine_args:
      enforce_eager: true
      distributed_executor_backend: mp
      quantization: fp8
      parallel_config:
        tensor_parallel_size: 2
        sequence_parallel_size: 2
        ulysses_degree: 2
    final_output: true
    final_output_type: image
```

**TP=2 + FP8 + CFG=2**

```yaml
# hunyuan_image3_dit_tp2_fp8_cfgp2.yaml
stage_args:
  - stage_id: 0
    stage_type: diffusion
    engine_args:
      enforce_eager: true
      distributed_executor_backend: mp
      quantization: fp8
      parallel_config:
        tensor_parallel_size: 2
        cfg_parallel_size: 2
    final_output: true
    final_output_type: image
```

#### Command

Start the DiT-only server with the selected stage config:

```bash
vllm serve tencent/HunyuanImage-3.0-Instruct \
  --omni \
  --port 8091 \
  --stage-configs-path hunyuan_image3_dit_tp2_fp8_cfgp2.yaml \
  --enable-diffusion-pipeline-profiler
```

Generate one 1024x1024 image:

```bash
curl -s http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "A cinematic photo of a glass observatory on Mars at sunrise"}
    ],
    "extra_body": {
      "height": 1024,
      "width": 1024,
      "num_inference_steps": 50,
      "guidance_scale": 5.0,
      "seed": 42
    }
  }' | jq -r '.choices[0].message.content[0].image_url.url' \
     | cut -d',' -f2- \
     | base64 -d > hunyuan_image3_output.png
```

#### Benchmark

The repository includes CI-style benchmark configs for the same DiT-only
settings. Run the configs individually:

```bash
export DIFFUSION_BENCHMARK_DIR=tests/dfx/perf/results
export CACHE_DIT_VERSION=1.3.0

pytest -s -v tests/dfx/perf/scripts/run_diffusion_benchmark.py \
  --test-config-file tests/dfx/perf/tests/test_hunyuan_image_tp4_fp8.json

pytest -s -v tests/dfx/perf/scripts/run_diffusion_benchmark.py \
  --test-config-file tests/dfx/perf/tests/test_hunyuan_image_tp2_fp8_sp2.json

pytest -s -v tests/dfx/perf/scripts/run_diffusion_benchmark.py \
  --test-config-file tests/dfx/perf/tests/test_hunyuan_image_tp2_fp8_cfgp2.json
```

#### Verification

Check that:

- The server responds on `http://localhost:8091/health`.
- The generation request writes a valid PNG file.
- Logs include `Selected CutlassFP8ScaledMMLinearKernel` for dense FP8
  linear layers and `Using TRITON Fp8 MoE` for MoE layers.
- With `--enable-diffusion-pipeline-profiler`, logs include per-stage timings
  such as `model.forward`, `patch_embed.forward`, `final_layer.forward`, and
  `vae.decode`.

Validated benchmark characteristics for 1024x1024, 50 denoising steps,
batch size 1:

| Configuration | Latency | Peak memory |
| --- | ---: | ---: |
| `tp4_fp8` | about 13.7s | about 47 GB |
| `tp2_fp8_sp2` | about 12.1s | about 66 GB |
| `tp2_fp8_cfgp2` | about 10.0s | about 66 GB |

#### Related Accuracy Smoke Data

PR [#3055](https://github.com/vllm-project/vllm-omni/pull/3055) adds a
DiT-only GEBench smoke setup for CI accuracy validation. Its validated
configuration was:

- Hardware: 4x H100.
- Runtime: TP=4 with expert parallel enabled, `bfloat16`,
  `distributed_executor_backend=mp`, `max_num_seqs=1`,
  `gpu_memory_utilization=0.95`, `enforce_eager=True`.
- Task scope: T2I-only GEBench type3/type4, 4 samples per type, 28 denoising
  steps.
- Judge: `QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ`.

Validated score summary:

| Metric | Score |
| --- | ---: |
| overall_mean | 0.955 |
| type3 overall_mean | 0.91 |
| type4 overall_mean | 1.00 |

The CI assertion threshold is `overall_mean >= 0.45`, so the smoke result is
comfortably above the gate. The generate server and judge server run
sequentially through the `OmniServer` fixture, and
`VLLM_TEST_CLEAN_GPU_MEMORY=1` is used to wait for memory cleanup between
server lifetimes.

The lower-cost 2-GPU Instruct setup was tried for this smoke path but did not
fit in memory. A previous 2-GPU experiment used the base HunyuanImage-3.0
checkpoint with FP8, but that base checkpoint is not available in the CI HF
cache. The validated CI-ready Instruct setup is therefore 4x H100 TP=4 with
expert parallel.

#### Related Functional Guard

PR [#3104](https://github.com/vllm-project/vllm-omni/pull/3104) adds an L3
dummy guard for the T2I request path. The guard exercises
`HunyuanImage3Pipeline.forward()` without loading the full checkpoint by
stubbing `prepare_model_inputs()` and `_generate()`. It verifies propagation
of:

- prompt and system prompt selection;
- output image size;
- inference steps and guidance scale;
- request generator;
- image `DiffusionOutput` and `stage_durations`.

#### Full AR-to-DiT KV-Reuse Reference

This recipe starts with DiT-only serving. For the full HunyuanImage
AR-to-DiT image-editing path, the KV-reuse work in PR #2949 provides useful
reference numbers and caveats.

Measured setup:

- Hardware: 4x NVIDIA L20X, 143 GB, driver 570.133.20.
- Parallelism: TP=2 for the AR stage and TP=2 for the DiT stage.
- Request: 1216x832 IT2I, 50 denoising steps, guidance 5.0, seed 42.
- Prompt/image: official HunyuanImage-3.0-Instruct demo image and prompt.

Performance summary:

| Path | AR generate | DiT denoise | End-to-end |
| --- | ---: | ---: | ---: |
| non-reuse baseline | 424 tokens | 57.3 s | 195 s |
| KV reuse | 443-481 tokens | 26.5-27.7 s | 169-174 s |

This is about `2.05-2.16x` faster for DiT denoising and about `1.40x` faster
end to end. The measured KV transfer cost for a single request was about
420 MB for the primary KV at roughly 1000 MB/s (about 0.4 s), plus about
70 MB for the CFG companion KV at roughly 270 MB/s.

The speedup comes from avoiding repeated text-prefix projection inside the
DiT. At 1216x832, the joint DiT sequence is about 10.7k tokens: roughly
6.7k text-prefix tokens and 4k image tokens. After KV injection, denoising
steps project only the image region and reuse the text KV, reducing the
attention-side per-step work by roughly `10.7k / 4k ~= 2.7x`.

Precision notes from the same PR:

| Measurement | PSNR | Interpretation |
| --- | ---: | --- |
| greedy KV-reuse, run 1 vs run 2 | inf dB | KV-reuse path is bitwise reproducible |
| greedy non-reuse, run 1 vs run 2 | 26.70 dB | non-reuse DiT determinism floor |
| greedy non-reuse with CFG expansion vs KV reuse | 10.22 dB | KV-reuse algorithmic drift |
| non-reuse vs KV reuse with same AR seed | 10.45 dB | total observed gap |
| non-reuse seed 42 vs seed 123 | 12.40 dB | AR sampling diversity reference |

The full AR-to-DiT path also fixed an IT2I regression where empty AR text made
the DiT ignore the source image. In the validated path, AR produced
non-empty text (`text length=749` in the reported run), and the DiT conditioned
on both the source image and edit prompt.

#### Notes

- This recipe intentionally starts with the DiT stage only. Full
  HunyuanImage AR-to-DiT generation uses the stage configs in
  `examples/offline_inference/hunyuan_image3` and
  `vllm_omni/model_executor/stage_configs`.
- `tp2_fp8_cfgp2` is usually fastest because CFG branches run in parallel.
  Individual layer timing can still look slower than `tp4_fp8` because each
  CFG branch uses TP=2, so each GPU owns a larger shard than in TP=4.
- `tp4_fp8` has the lowest per-GPU memory because weights are sharded across
  all four GPUs, but it pays more all-reduce communication overhead.
- `tp2_fp8_sp2` can improve model-forward latency by splitting sequence work,
  while adding all-to-all communication overhead.
- If you see OOM on 80GB GPUs, reduce image size or lower
  `gpu_memory_utilization` in the stage config before increasing batch size.
