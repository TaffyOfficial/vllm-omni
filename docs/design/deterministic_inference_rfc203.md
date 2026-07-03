# Deterministic Inference Support for verl-omni RFC 203

## Background

verl-omni issue 203 asks for bitwise reproducible end-to-end training. The
vLLM-Omni part is narrower than the whole RFC: vLLM-Omni must make rollout
inference deterministic when the caller provides deterministic request
metadata.

Trainer configuration, Ray runtime environment propagation, reward-model
serialization, and Hydra schema changes live in verl-omni and are not
implemented here.

## Goals

- Preserve existing behavior by default.
- Carry request `priority` from public generate APIs into scheduler-visible
  request objects.
- Honor `VLLM_BATCH_INVARIANT=1` as an opt-in deterministic scheduling mode.
- Preserve per-request `SamplingParams.seed` on AR paths.
- Make diffusion requests reject missing seeds under deterministic mode instead
  of silently assigning random fallback seeds.
- Add focused tests for priority, seed, and deterministic scheduling semantics.

## Non-Goals

- Do not add verl-omni config keys in this repository.
- Do not add Ray runtime environment propagation here.
- Do not implement reward-model serialization here.
- Do not claim cross-hardware bitwise equality.
- Do not add model-path-specific environment variables for heavy E2E tests.

## Owner Paths

### Public API Plumbing

Owner files:

- `vllm_omni/entrypoints/async_omni.py`
- `vllm_omni/engine/async_omni_engine.py`
- `vllm_omni/engine/messages.py`
- `vllm_omni/engine/orchestrator.py`
- `vllm_omni/engine/stage_pool.py`
- `vllm_omni/diffusion/stage_diffusion_client.py`
- `vllm_omni/diffusion/inline_stage_diffusion_client.py`
- `vllm_omni/diffusion/stage_diffusion_proc.py`
- `vllm_omni/request.py`
- `vllm_omni/engine/__init__.py`

Expected behavior:

- `AsyncOmni.generate(..., priority=N)` passes `N` to
  `AsyncOmniEngine.add_request_async`.
- Orchestrator messages and downstream stage submissions preserve the same
  priority for the request.
- Diffusion clients carry priority across inline and subprocess execution.
- Existing callers that omit priority continue to use priority `0`.

### AR Sampling Seed

Owner files:

- `vllm_omni/worker/gpu_model_runner.py`
- `vllm_omni/core/sched/output.py`

Expected behavior:

- When `SamplingParams.seed` produces `SamplingType.RANDOM_SEED`, the model
  runner creates a per-request generator from that seed.
- Omni scheduler wrappers preserve `sampling_params.seed`.

### Deterministic Scheduling

Owner files:

- `vllm_omni/core/sched/omni_generation_scheduler.py`
- `vllm_omni/core/sched/omni_ar_scheduler.py`
- `vllm_omni/diffusion/sched/base_scheduler.py`
- shared scheduler helper if both schedulers need the same logic

Expected behavior when `VLLM_BATCH_INVARIANT=1`:

- New waiting requests are considered in stable request-priority order, not
  arrival order.
- Priority follows vLLM semantics: lower numeric values are handled earlier.
- Ties are deterministic, using request id as the tie breaker.
- vLLM-stage schedulers use an FCFS backing queue in this mode so the explicit
  deterministic ordering is not overridden by the priority queue's arrival-time
  tie breaker.
- Schedulers cap concurrent running requests at one. This is the conservative
  vLLM-Omni guarantee until every Omni kernel, multimodal preprocessing path,
  and diffusion denoise pipeline has proven batch-composition invariance.
- The deterministic mode is opt-in only; existing scheduler behavior remains
  unchanged when the env var is absent or false.
- Pure diffusion schedulers use the same stable ordering as vLLM-stage
  schedulers.

### Diffusion Request Seeding

Owner files:

- `vllm_omni/diffusion/request.py`
- `vllm_omni/diffusion/worker/diffusion_model_runner.py`

Expected behavior:

- Normal serving keeps auto-randomized seeds for backward compatibility.
- Under deterministic mode, a diffusion request without explicit seed or
  generator is rejected instead of silently assigning a random seed.
- When a seed is present, generator creation remains per request and process
  safe.
- For multi-prompt diffusion requests in deterministic mode, the runner expands
  the request seed into one generator per sample using stable `request_id`
  derivation, so noise assignment is not tied to batch position.

## Test Plan

- `tests/entrypoints/test_async_omni.py`: verify
  `AsyncOmni.generate(..., priority=N)` forwards `N` into
  `engine.add_request_async`.
- `tests/core/sched/test_omni_scheduler_mixin.py`: verify deterministic mode
  selects waiting requests by `(priority, request_id)` and default mode keeps
  arrival-order behavior.
- `tests/diffusion/test_diffusion_request.py`: verify deterministic mode
  rejects missing diffusion seed and default mode still auto-assigns different
  seeds.
- `tests/diffusion/test_diffusion_model_runner.py`: verify deterministic mode
  derives per-sample generators for multi-prompt diffusion requests while
  default mode keeps the historical single-generator behavior.
- `tests/diffusion/test_diffusion_scheduler.py`: verify pure diffusion
  scheduler ordering in deterministic mode.
- `tests/diffusion/test_inline_stage_diffusion_client.py`,
  `tests/diffusion/test_multiproc_engine_concurrency.py`, and
  `tests/diffusion/test_stage_diffusion_proc.py`: verify priority survives
  inline dispatch, ZMQ payloads, and subprocess request reconstruction.
- `tests/engine/test_orchestrator_kv_sender_info.py`: verify orchestrator
  diffusion submissions preserve priority while still attaching KV sender info.
- Run targeted pytest files, `ruff check` on touched Python files, and
  `git diff --check`.

## Rollout Order

1. Add a small deterministic-mode helper.
2. Fix priority propagation from `AsyncOmni.generate`.
3. Apply deterministic request ordering to vLLM-stage schedulers.
4. Apply the same ordering to pure diffusion schedulers.
5. Add diffusion seed guard under deterministic mode.
6. Run focused tests and document any remaining heavy E2E gap.

## Open Questions

- Whether diffusion full batch invariance needs model-pipeline-specific changes
  beyond scheduler order and seed handling. Those need model-specific tests
  rather than a generic silent claim.
