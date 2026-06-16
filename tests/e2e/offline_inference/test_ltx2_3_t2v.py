# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""L2 mock-weight offline inference coverage for LTX-2.3 text-to-video."""

from pathlib import Path

import numpy as np
import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt

MODEL = str(Path(__file__).parent / "fixtures" / "ltx23_mock_model")
PROMPT = "A tiny paper boat floats down a moonlit canal, cinematic lighting."
NEGATIVE_PROMPT = "blurry, distorted, low quality, watermark"
CUSTOM_PIPELINE_CLASS = "tests.e2e.offline_inference.custom_pipeline.ltx23_mock_pipeline.LTX23MockPipelineForTest"


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize(
    "omni_runner",
    [
        (
            MODEL,
            None,
            {
                "custom_pipeline_args": {"pipeline_class": CUSTOM_PIPELINE_CLASS},
                "enforce_eager": True,
            },
        )
    ],
    indirect=True,
)
def test_ltx2_3_text_to_video_mock_shape_001(omni_runner: OmniRunner) -> None:
    height = 256
    width = 256
    num_frames = 5
    fps = 8
    sampling = OmniDiffusionSamplingParams(
        height=height,
        width=width,
        num_frames=num_frames,
        fps=fps,
        num_inference_steps=2,
        guidance_scale=4.0,
        seed=42,
    )
    prompt = OmniTextPrompt(prompt=PROMPT, negative_prompt=NEGATIVE_PROMPT)

    outputs = omni_runner.generate([prompt], [sampling])

    assert len(outputs) == 1
    output = outputs[0]
    assert output.final_output_type == "image"
    assert output.images and len(output.images) == 1
    assert output.multimodal_output.get("fps") == fps
    assert output.custom_output.get("mock_model") == "ltx2.3"
    assert output.custom_output.get("video_shape") == [num_frames, height, width, 3]

    video_frames = output.images[0]
    assert isinstance(video_frames, list)
    assert len(video_frames) == num_frames
    for frame in video_frames:
        frame_array = np.asarray(frame)
        assert frame_array.shape == (height, width, 3)
        assert frame_array.dtype == np.uint8
