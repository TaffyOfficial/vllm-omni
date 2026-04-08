# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Smoke test for HunyuanImage-3.0 Image-to-Text (I2T) pipeline."""

from collections.abc import Generator
from pathlib import Path

import pytest
import torch

from vllm_omni import Omni

MODEL_NAME = "tencent/HunyuanImage-3.0-Instruct"
REPO_ROOT = Path(__file__).resolve().parents[3]
STAGE_CONFIG_PATH = REPO_ROOT / "vllm_omni" / "model_executor" / "stage_configs" / "hunyuan_image3_i2t.yaml"

pytestmark = [pytest.mark.advanced_model, pytest.mark.diffusion]


@pytest.fixture(scope="module")
def omni() -> Generator[Omni, None, None]:
    engine = Omni(
        model=MODEL_NAME,
        stage_configs_path=str(STAGE_CONFIG_PATH),
        stage_init_timeout=600,
        init_timeout=900,
    )
    try:
        yield engine
    finally:
        engine.close()


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="Need at least 8 CUDA GPUs.")
def test_i2t_generates_text(omni: Omni) -> None:
    """Verify that the I2T pipeline produces non-empty text output."""
    # Use a simple solid-color image as input (no external file dependency)
    from PIL import Image
    input_image = Image.new("RGB", (256, 256), color=(128, 200, 100))

    prompt = (
        "<|startoftext|>You are an assistant that understands images "
        "and outputs text.<img>Describe the content of the picture."
    )
    prompt_dict = {
        "prompt": prompt,
        "modalities": ["text"],
        "multi_modal_data": {"image": input_image},
    }

    outputs = omni.generate(prompts=[prompt_dict])
    assert outputs, "No outputs returned from Omni.generate()"

    first_output = outputs[0]
    request_output = getattr(first_output, "request_output", first_output)
    assert request_output.outputs, "No completion outputs"

    generated_text = request_output.outputs[0].text
    assert isinstance(generated_text, str), f"Expected str, got {type(generated_text)}"
    assert len(generated_text) > 0, "Generated text is empty"
