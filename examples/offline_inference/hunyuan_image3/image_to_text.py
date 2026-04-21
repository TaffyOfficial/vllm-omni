# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import os
from pathlib import Path

from PIL import Image

from vllm_omni.entrypoints.omni import Omni

from prompt_utils import build_prompt

"""
HunyuanImage-3.0-Instruct Image-to-Text (I2T) example.

System prompt is auto-selected by prompt_utils.build_prompt(task="i2t").
Prompt format (pretrain template):
  <|startoftext|>{system_prompt}<img>{trigger_tag}{user_prompt}
"""

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STAGE_CONFIG = REPO_ROOT / "vllm_omni" / "model_executor" / "stage_configs" / "hunyuan_image3_i2t.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from image using HunyuanImage-3.0-Instruct.")
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
        help="Question about the image, e.g. 'Describe the content of the picture.'",
    )
    parser.add_argument(
        "--stage-configs-path",
        type=str,
        default=None,
        help="Path to stage config YAML. Defaults to auto-resolve hunyuan_image3_i2t.yaml.",
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

    prompt = build_prompt(args.prompt, task="i2t")

    prompt_dict = {
        "prompt": prompt,
        "modalities": ["text"],
    }

    # Add image input if provided
    if args.image:
        if not os.path.exists(args.image):
            raise FileNotFoundError(f"Input image not found: {args.image}")

        input_image = load_image(args.image)
        prompt_dict["multi_modal_data"] = {"image": input_image}

    prompts = [prompt_dict]
    omni_outputs = omni.generate(prompts=prompts)

    first_output = omni_outputs[0]
    request_output = getattr(first_output, "request_output", first_output)
    prompt_text = request_output.prompt
    generated_text = request_output.outputs[0].text
    print(f"Prompt: {prompt_text}")
    print(f"Text: {generated_text}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
