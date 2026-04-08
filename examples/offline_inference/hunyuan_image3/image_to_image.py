# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import os

from PIL import Image

from vllm_omni.entrypoints.omni import Omni

"""
HunyuanImage-3.0-Instruct Image-to-Image (IT2I / TI2I) example.

This uses a 2-stage pipeline:
  Stage 0 (AR): reads (image + edit instruction), generates CoT + latent tokens
  Stage 1 (DiT): denoises latents → edited image

The unified system prompt enables <think> mode for reasoning before generation.
See upstream README for details on prompt templates and bot_task modes.
"""

DEFAULT_SYSTEM_PROMPT = (
    "You are an advanced multimodal model whose core mission is to "
    "analyze user intent and generate high-quality text and images."
)


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
    return parser.parse_args()


def load_image(image_path: str) -> Image.Image:
    """Load an image from file path."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    return Image.open(image_path).convert("RGB")


def main(args: argparse.Namespace) -> None:
    stage_configs_path = (
        args.stage_configs_path
        or "vllm_omni/model_executor/stage_configs/hunyuan_image3_it2i.yaml"
    )
    omni = Omni(
        model=args.model,
        stage_configs_path=stage_configs_path,
    )

    # Build IT2I prompt: system + <img> placeholder + edit instruction
    prompt = (
        f"<|startoftext|>{DEFAULT_SYSTEM_PROMPT}"
        f"<img><think>{args.prompt}"
    )

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
