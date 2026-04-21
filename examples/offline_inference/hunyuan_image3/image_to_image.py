# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import os
from pathlib import Path

from PIL import Image

from vllm_omni.entrypoints.omni import Omni

from prompt_utils import build_prompt

"""
HunyuanImage-3.0-Instruct Image-to-Image (IT2I / TI2I) example.

This uses a 2-stage pipeline:
  Stage 0 (AR): reads (image + edit instruction), generates CoT + latent tokens
  Stage 1 (DiT): denoises latents → edited image

System prompt and <think>/<recaption> tags are auto-constructed by
prompt_utils.build_prompt() based on --bot-task.
"""

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STAGE_CONFIG = REPO_ROOT / "vllm_omni" / "model_executor" / "stage_configs" / "hunyuan_image3_it2i.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Edit an image using HunyuanImage-3.0-Instruct (IT2I)."
    )
    parser.add_argument(
        "--model",
        default="tencent/HunyuanImage-3.0-Instruct",
        help="Model name or local path.",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image file (PNG, JPG, etc.).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Edit instruction, e.g. 'Make the petals neon pink'.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="Path to save the generated image.",
    )
    parser.add_argument(
        "--stage-configs-path",
        type=str,
        default=None,
        help="Path to stage config YAML. Defaults to hunyuan_image3_it2i.yaml.",
    )
    parser.add_argument(
        "--bot-task",
        type=str,
        default="it2i_think",
        choices=["it2i_think", "it2i_recaption"],
        help="Prompt mode: it2i_think (CoT+recaption) or it2i_recaption (recaption only).",
    )
    return parser.parse_args()


def load_image(image_path: str) -> Image.Image:
    """Load an image from file path."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    return Image.open(image_path).convert("RGB")


def main(args: argparse.Namespace) -> None:
    stage_configs_path = args.stage_configs_path or str(DEFAULT_STAGE_CONFIG)
    omni = Omni(
        model=args.model,
        stage_configs_path=stage_configs_path,
    )

    # Build IT2I prompt with auto-selected system prompt and mode tags
    prompt = build_prompt(args.prompt, task=args.bot_task)

    input_image = load_image(args.image)

    prompt_dict = {
        "prompt": prompt,
        "modalities": ["image"],
        "multi_modal_data": {"image": input_image},
        "height": input_image.height,
        "width": input_image.width,
    }

    prompts = [prompt_dict]

    for stage_outputs in omni.generate(prompts=prompts, py_generator=True):
        output = stage_outputs.request_output
        if stage_outputs.final_output_type == "image":
            images = output.images if hasattr(output, "images") else []
            if not images and hasattr(output, "multimodal_output"):
                images = output.multimodal_output.get("images", [])

            if images:
                images[0].save(args.output)
                print(f"Saved edited image to {args.output}")
            else:
                print("No image generated.")
        elif stage_outputs.final_output_type == "text":
            # AR stage intermediate text output (CoT reasoning)
            text = output.outputs[0].text if output.outputs else ""
            if text:
                print(f"CoT: {text[:200]}...")


if __name__ == "__main__":
    args = parse_args()
    main(args)
