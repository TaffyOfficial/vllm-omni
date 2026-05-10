# GO-1-Air offline inference

Minimal smoke test for the `Go1AirPipeline` registration. The full open-loop
evaluation harness (dataset loader, deterministic noise, result archiving)
lives in a follow-up PR mirroring `examples/offline_inference/internvla_a1/`.

## Run

```bash
# Stub mode (no checkpoint, validates import + pipeline shape contract).
bash run.sh

# With weights (sets the env var; same script).
export GO1_AIR_MODEL_DIR=/path/to/GO-1-Air
bash run.sh
```

The smoke succeeds when it prints `[smoke] OK action shape=(1, 30, 16)`.

## Upstream license note

The GO-1-Air weights on HuggingFace (`agibot-world/GO-1-Air`) are released under
**CC BY-NC-SA 4.0** (NonCommercial + ShareAlike). The vllm-omni integration
code in this repository is Apache-2.0 and contains no upstream model code —
it loads weights at runtime and runs them through clean-room implementations
of the architecture. Downstream commercial use of the weights is governed by
AgiBot's license, not by Apache-2.0; consult the GO-1-Air model card before
deploying.
