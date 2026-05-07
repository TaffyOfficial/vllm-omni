#!/bin/bash
# HunyuanImage-3.0-Instruct online serving startup script.
#
# Defaults to the 8-GPU two-stage config (4 AR + 4 DiT). Override via env vars:
#
#   STAGE_CONFIG=path/to/yaml      # Custom stage config (e.g. hunyuan_image3_t2i_4gpu.yaml)
#   PORT=8091                      # Server port
#   MODEL=tencent/HunyuanImage-3.0-Instruct
#
# On Ada-class (sm89) GPUs (L20 / L40 / L40S / RTX 6000 Ada), set:
#
#   MOE_BACKEND=triton             # Avoids flashinfer cutlass MoE auto-pick
#                                  # (flashinfer's MoE path targets sm90+ only)
#
# Examples:
#
#   # 4-GPU (2 AR + 2 DiT) on L20X / L40S
#   STAGE_CONFIG=$(python -c "import vllm_omni, os; print(os.path.join(os.path.dirname(vllm_omni.__file__), 'model_executor/stage_configs/hunyuan_image3_t2i_4gpu.yaml'))") \
#   MOE_BACKEND=triton bash run_server.sh
#
#   # 8-GPU MoE default
#   bash run_server.sh
#
#   # 2-GPU FP8 DiT-only (AR runs externally; for development only)
#   STAGE_CONFIG=$(python -c "...moe_dit_2gpu_fp8.yaml...") bash run_server.sh

set -euo pipefail

MODEL="${MODEL:-tencent/HunyuanImage-3.0-Instruct}"
PORT="${PORT:-8091}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-$SCRIPT_DIR/chat_template.jinja}"

# Resolve default stage config from the installed vllm_omni package so the
# script works no matter where the user runs it from.
DEFAULT_STAGE_CONFIG=$(python -c "import vllm_omni, os; print(os.path.join(os.path.dirname(vllm_omni.__file__), 'model_executor/stage_configs/hunyuan_image3_moe.yaml'))")
STAGE_CONFIG="${STAGE_CONFIG:-$DEFAULT_STAGE_CONFIG}"

EXTRA_ARGS=()
if [[ -n "${MOE_BACKEND:-}" ]]; then
    EXTRA_ARGS+=(--moe-backend "$MOE_BACKEND")
fi
if [[ "${ENABLE_PROFILER:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--enable-diffusion-pipeline-profiler --enable-ar-profiler)
fi

echo "Starting HunyuanImage-3.0-Instruct server..."
echo "  Model:         $MODEL"
echo "  Port:          $PORT"
echo "  Stage config:  $STAGE_CONFIG"
echo "  Chat template: $CHAT_TEMPLATE"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    echo "  Extra args:    ${EXTRA_ARGS[*]}"
fi

vllm serve "$MODEL" --omni \
    --port "$PORT" \
    --stage-configs-path "$STAGE_CONFIG" \
    --chat-template "$CHAT_TEMPLATE" \
    "${EXTRA_ARGS[@]}"
