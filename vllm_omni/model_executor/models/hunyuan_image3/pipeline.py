# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HunyuanImage-3.0 pipeline topologies (frozen).

HunyuanImage-3.0 is a multi-task model. One HF checkpoint serves five
distinct pipeline topologies; deploy YAML (or PR #2383 CLI) picks which
one to use. Each ``PipelineConfig`` below declares one topology:

  * DIT_ONLY (default)          Stage 0 DiT only, prompt -> image (no AR rewrite)
  * T2I  (text-to-image)        Stage 0 AR -> Stage 1 DiT
  * IT2I (image+text-to-image)  Stage 0 AR (multimodal) -> Stage 1 DiT
  * I2T  (image-to-text)        Stage 0 AR only, multimodal, text out
  * T2T  (text-to-text)         Stage 0 AR only, text out

The HF ``model_type`` reported by the checkpoint is ``hunyuan_image_3_moe``.
We register **DIT_ONLY under that exact model_type** so it wins HF
auto-detection — DIT_ONLY is the only topology that reliably works today
(see ``docs/design/hunyuan_image3_it2i_gap.md``: AR→DiT bridge has 5
code-level gaps, so the AR stage does not actually affect DiT output).
The other four topologies are registered under synthetic suffixes
(``_ar_dit`` / ``_it2i`` / ``_i2t`` / ``_t2t``) for callers that opt in
via ``deploy.pipeline`` (or a future ``--pipeline`` CLI). When the bridge
gap is fixed, the default should be switched back to the two-stage T2I.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_MODEL_ARCH = "HunyuanImage3ForCausalMM"
_HF_ARCHS = ("HunyuanImage3ForCausalMM",)
_PROC = "vllm_omni.model_executor.stage_input_processors.hunyuan_image3"

# ---- Shared stage fragments ------------------------------------------------
# AR sampling for image generation tasks (T2I / IT2I). Aligned with the four
# stage_configs/hunyuan_image3_*.yaml that shipped with the model.
_AR_GEN_SAMPLING = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 1024,
    "max_tokens": 4096,
    "stop_token_ids": [127957],  # <|endoftext|>
    "detokenize": False,
}

# AR sampling for text-output tasks (I2T / T2T). Greedy; stops on
# <|endoftext|> or </answer>.
_AR_TEXT_SAMPLING = {
    "temperature": 0.0,
    "top_p": 0.95,
    "top_k": 1024,
    "max_tokens": 2048,
    "stop_token_ids": [127957, 128026],
    "detokenize": True,
}


# ---- DIT_ONLY : single-stage DiT, prompt -> image (AR rewrite disabled) ----
# Registered under the HF-native model_type "hunyuan_image_3_moe" so that
# HF auto-detection picks this by default. This is the only topology that
# reliably produces correct images today, because the AR→DiT bridge is
# incomplete (see docs/design/hunyuan_image3_it2i_gap.md). Equivalent to
# calling the official `generate_image(prompt, bot_task="image")` path.
HUNYUAN_IMAGE3_DIT_ONLY_PIPELINE = PipelineConfig(
    model_type="hunyuan_image_3_moe",
    model_arch=_MODEL_ARCH,
    hf_architectures=_HF_ARCHS,
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="image",
            owns_tokenizer=True,
            requires_multimodal_data=False,
            sampling_constraints={"seed": 42},
        ),
    ),
)


# ---- T2I : AR -> DiT (two-stage, AR→DiT bridge currently broken) -----------
# Registered under a synthetic suffix; select via ``deploy.pipeline:`` or a
# future ``--pipeline`` CLI. Once the bridge is fixed, this should be
# promoted back to the default ``model_type``.
HUNYUAN_IMAGE3_T2I_PIPELINE = PipelineConfig(
    model_type="hunyuan_image_3_moe_ar_dit",
    model_arch=_MODEL_ARCH,
    hf_architectures=_HF_ARCHS,
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="AR",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=False,
            owns_tokenizer=True,
            requires_multimodal_data=False,
            engine_output_type="latent",
            sampling_constraints=_AR_GEN_SAMPLING,
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(0,),
            final_output=True,
            final_output_type="image",
            custom_process_input_func=f"{_PROC}.ar2diffusion",
        ),
    ),
)


# ---- IT2I : AR(multimodal) -> DiT ------------------------------------------
HUNYUAN_IMAGE3_IT2I_PIPELINE = PipelineConfig(
    model_type="hunyuan_image_3_moe_it2i",
    model_arch=_MODEL_ARCH,
    hf_architectures=_HF_ARCHS,
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="AR",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=False,
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="latent",
            sampling_constraints=_AR_GEN_SAMPLING,
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(0,),
            final_output=True,
            final_output_type="image",
            requires_multimodal_data=True,
            custom_process_input_func=f"{_PROC}.ar2diffusion",
        ),
    ),
)


# ---- I2T : single-stage AR, multimodal -> text -----------------------------
HUNYUAN_IMAGE3_I2T_PIPELINE = PipelineConfig(
    model_type="hunyuan_image_3_moe_i2t",
    model_arch=_MODEL_ARCH,
    hf_architectures=_HF_ARCHS,
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="AR",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            sampling_constraints=_AR_TEXT_SAMPLING,
        ),
    ),
)


# ---- T2T : single-stage AR, text only --------------------------------------
HUNYUAN_IMAGE3_T2T_PIPELINE = PipelineConfig(
    model_type="hunyuan_image_3_moe_t2t",
    model_arch=_MODEL_ARCH,
    hf_architectures=_HF_ARCHS,
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="AR",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=False,
            sampling_constraints=_AR_TEXT_SAMPLING,
        ),
    ),
)
