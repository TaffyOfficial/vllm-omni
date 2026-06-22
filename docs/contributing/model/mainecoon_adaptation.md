# MaineCoon Adaptation Mini Spec

This note records the current vLLM-Omni intake for
[`catnip-ai-tech/MaineCoon`](https://huggingface.co/catnip-ai-tech/MaineCoon).
It is intentionally not a supported-model entry until a runnable checkpoint and
loader contract are public.

## Current Upstream State

- Hugging Face model id: `catnip-ai-tech/MaineCoon`
- Hugging Face revision checked: `a4b688a77382bf71ee64132a7b5e4abb82e56771`
- Published files checked: `.gitattributes`, `README.md`,
  `MaineCoon_Technical_Report.pdf`
- Public GitHub repository checked:
  [`catnip-ai-tech/MaineCoon`](https://github.com/catnip-ai-tech/MaineCoon)
- Public GitHub scope: project README, report, demos, and links; no model
  source code or inference loader is published.

The published Hugging Face repository has no `config.json`, `model_index.json`,
tokenizer files, model subfolders, or weight shards. vLLM-Omni therefore cannot
auto-detect it as either a Transformers-style model or a Diffusers-style
pipeline.

## Mini Spec

- Goal: add MaineCoon as a real-time, streaming text/audio-to-audio-video
  generation model only after a runnable artifact is available.
- Checkpoint layout:
  - Runnable model id: none public as of the revision above.
  - Raw/upstream model id: `catnip-ai-tech/MaineCoon` is a report/model-card
    repository, not a runnable checkpoint.
  - Required files/subfolders before implementation:
    - root `config.json` with `model_type` and `architectures`, or a
      `model_index.json` with a loadable pipeline class;
    - transformer block configs and weight shards for the causal audio-visual
      generator;
    - tokenizer/processor assets for text, audio, and video conditioning;
    - audio/video latent tokenizer or VAE configs and weights;
    - any streaming cache manager, sliding-window decode, and audio
      reconstruction code needed to match official inference.
- Public entrypoints:
  - Offline text/audio-to-audio-video: pending runnable checkpoint and official
    input contract.
  - Online streaming generation: pending official streaming protocol and
    chunk/cadence contract.
  - Diffusers adapter: not applicable to the current public repo because
    `DiffusionPipeline.from_pretrained()` has no `model_index.json` or pipeline
    files to load.
  - Supported models table: do not list until the runnable artifact is
    validated.
- Request fields:
  - Ingress: likely prompt text plus optional audio/video context, but the
    public report does not define a stable API schema.
  - Default semantics: unknown for chunk size, FPS, resolution, seed,
    prompt-planning, cache budget, audio codec, and conditioning windows.
  - Owner: model-specific pipeline should own streaming/chunk/cache semantics;
    OpenAI-compatible serving should only pass user-visible fields through.
  - Consumers: causal audio-visual generator, KV-cache manager, sliding-window
    VAE/audio decode path, and streaming response writer.
  - Failure policy: fail fast until all required official artifacts are
    available; do not silently route to LTX-2.3 or the generic Diffusers
    adapter.
- Path parity:
  - Normal path: a native multi-stage or single-stage streaming pipeline must
    share the same request parser between offline and serving paths.
  - Variant paths: non-streaming smoke, streaming online, and long-horizon
    cache-managed generation need explicit parity tests.
  - Shared helper or split: parsing should be shared; cache commitment,
    eviction, and byte-stream emission are MaineCoon-specific.
- Validation tiers:
  - Unit: checkpoint layout detection, request normalization, chunk/cache shape
    contracts, post-processing/muxing contracts.
  - Public smoke: one official prompt at an official runnable checkpoint, with
    generated video+audio artifact and server log.
  - Formal perf: only after latest-head smoke passes; include resolution,
    duration, FPS, chunk size, hardware, command, result artifact, and whether
    cache manager is enabled.
- PR evidence:
  - Latest-head: pending runnable checkpoint.
  - Historical: technical report only; not valid as runtime evidence.
  - Pending: official source/weights, public input schema, and at least one
    smoke artifact.
- Non-goals:
  - Do not claim support from the current model-card-only HF repository.
  - Do not alias MaineCoon to LTX-2.3. The report says MaineCoon initializes
    from LTX-2.3, but its causal audio-visual generator, KV-cache reuse, and
    agentic streaming inference are different runtime contracts.
  - Do not publish latency, VRAM, or FPS numbers in vLLM-Omni docs until they
    come from a latest-head vLLM-Omni run.

## Implementation Plan Once Artifacts Are Published

1. Verify checkpoint layout locally or through the Hugging Face API:
   `config.json` / `model_index.json`, component subfolders, tokenizer and
   processor assets, and all weight shards.
2. Classify the runtime:
   - If official artifacts are Diffusers-compatible, first try the existing
     Diffusers backend adapter and only add a native pipeline when vLLM-Omni
     features such as streaming, cache management, or parallelism require it.
   - If artifacts are a custom autoregressive audio-visual model, add a native
     omni pipeline under `vllm_omni/model_executor/models/mainecoon/` with a
     `PipelineConfig`, deploy YAML, model registry entries, and stage input
     processors.
3. Keep the first public PR narrow:
   load/config detection, offline smoke, and one serving path. Long-horizon
   agentic planning, formal performance, and advanced cache controls should be
   follow-up PRs unless official smoke requires them.
4. Add tests only at behavior owners:
   registry/config tests for model detection, parser tests for request fields,
   and model-specific tests for cache/chunk contracts.
5. Update `docs/models/supported_models.md` only after the runnable model id has
   a passing latest-head smoke.
