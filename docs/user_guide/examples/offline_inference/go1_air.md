# GO-1-Air

> Source repository: <https://github.com/OpenDriveLab/AgiBot-World>
>
> Weights: <https://huggingface.co/agibot-world/GO-1-Air>

This example runs offline inference for **AgiBot GO-1-Air**, an open-source
Vision-Language-Latent-Action (ViLLA) policy. The released checkpoint ships
with the Latent Planner disabled, so this integration covers the
`Vision → Language → diffusion action` path only; an `[B, 30, 16]` action
chunk is produced with the upstream squared-cosine diffusion schedule and
5-step DPM-Solver sampling.

## Quick start

```bash
# (Optional) point at a checkpoint directory containing config.json
# and model.safetensors[.index.json].
export GO1_AIR_MODEL_DIR=/path/to/GO-1-Air

bash examples/offline_inference/go1_air/run.sh
```

The smoke test prints `[smoke] OK action shape=(1, 30, 16)` on success.

## Input contract

GO-1-Air consumes repository-side tensors through
`sampling_params.extra_args["batch_inputs"]`:

| Key | Shape / type | Notes |
| --- | --- | --- |
| `observation.state` | `torch.Tensor[B, 16]` | Robot state vector. |
| `observation.task` | `str` for `B=1`, or `list[str]` / `tuple[str, ...]` of length `B` | Language instruction. |
| `observation.images.<camera>` | `torch.Tensor[B, history, 3, 448, 448]` | RGB image history, already resized to the model resolution. |
| `observation.images.<camera>_mask` | optional scalar, `torch.Tensor[B]`, or `torch.Tensor[B, history]` | Marks valid cameras/history frames before vision tokens are allocated. |
| `control_freq` | optional `torch.Tensor[B]` or scalar tensor | Defaults to 30 Hz when omitted. |

`extra_args["noise"]` may be provided for deterministic debugging and must
match `[B, 30, 16]`.

## Notes

* The full open-loop evaluation harness (dataset loader, deterministic noise,
  result archiving) is added in a follow-up PR — see
  `examples/offline_inference/internvla_a1/` for the structure that will be
  mirrored.
* The GO-1-Air weights are licensed under **CC BY-NC-SA 4.0**; the
  vllm-omni integration code is Apache-2.0 and contains no upstream model
  code. Downstream commercial deployment of the weights is governed by
  AgiBot's license — see the model card before shipping.
