# Step Execution

Step execution is an opt-in diffusion execution mode enabled with
`step_execution=True` when constructing `Omni`.

It is not a generic diffusion toggle for every pipeline. Only pipelines that
implement the stepwise contract support it today.

## Quick Start

### Python API

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(
    model="Qwen/Qwen-Image",
    step_execution=True,
)

outputs = omni.generate(
    "A cat sitting on a windowsill",
    OmniDiffusionSamplingParams(
        num_inference_steps=50,
    ),
)
```

### Serving

```bash
vllm serve Qwen/Qwen-Image --omni \
  --port 8091 \
  --step-execution \
  --max-num-seqs 8
```

For serving, `--step-execution` enables the step-wise runtime. Continuous
batching only becomes relevant when `--max-num-seqs > 1`.

For HunyuanImage3 DiT-stage experiments, keep the default deploy YAMLs
unchanged and opt in explicitly on the DiT stage with `step_execution: true`
and a small `max_num_seqs`, such as `2` or `4`.

## Supported Pipelines

| Pipeline | Example models | Step execution |
|----------|----------------|----------------|
| `QwenImagePipeline` | `Qwen/Qwen-Image`, `Qwen/Qwen-Image-2512` | Yes |
| `HunyuanImage3Pipeline` | HunyuanImage-3 DiT stage | Experimental: same-resolution grouped DiT batching only |
| All other diffusion pipelines | `QwenImageEditPipeline`, `QwenImageEditPlusPipeline`, `QwenImageLayeredPipeline`, GLM-Image, Wan, Flux, etc. | No |

!!! warning "Experimental continuous batching"
    When `--step-execution` is enabled and `max_num_seqs > 1` is configured,
    the step-wise path can batch
    compatible requests together. This is experimental. Requests with
    incompatible sampling parameters are intentionally kept in separate batches,
    and `max_num_seqs=1` remains the conservative default.

## Current Limitations

- Continuous batching under `step_execution` is experimental and only batches
  compatible requests.
- HunyuanImage3 step execution currently requires one prompt per request and
  batches only same-resolution requests with matching denoise step counts and
  guidance settings. Per-request prompt encoding remains independent, and the
  DiT step-wise merge right-pads variable prompt-token sequence fields within
  the active batch. Custom timesteps/sigmas are rejected, and staggered
  AR-to-DiT arrivals may still run as separate DiT batches. Sequence
  parallelism, CFG parallelism, and multi-output-per-prompt are rejected in
  this path.
- `cache_backend` is not supported together with step execution.
- Unsupported pipelines fail early during model loading.
- Request-mode extras such as KV transfer are not wired into step mode yet.

## When To Use It

Use step execution only when you specifically need the pipeline to run through
its stepwise request state machine. For normal diffusion inference, leave it
disabled unless your workflow depends on this mode.

For Qwen-Image online serving, the usual progression is:

- start with `--step-execution --max-num-seqs 1` if you only need the step-wise path
- increase `--max-num-seqs` after that if you want the experimental compatible-request batching behavior

If you are looking for general diffusion speedups, see
[Diffusion Features Overview](../diffusion_features.md).

## Troubleshooting

If model loading fails with a message mentioning `prepare_encode()`,
`denoise_step()`, `step_scheduler()`, and `post_decode()`, the selected
pipeline does not support step execution.

## For Model Authors

If you want to add step execution support to a new diffusion pipeline, see the
implementation guide:
[Diffusion Step Execution Design](../../design/feature/diffusion_step_execution.md).

If you also want that pipeline to participate in the experimental batched
step-wise path, see:
[Continuous Batching for Step-Wise Diffusion](../../design/feature/diffusion_continuous_batching.md).
