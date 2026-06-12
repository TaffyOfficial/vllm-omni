# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from examples.offline_inference.video_model_defaults import (
    default_image_to_video_class_name,
    default_text_to_video_class_name,
    detect_text_to_video_preset,
    is_ltx2_model,
    is_ltx23_model,
)


def test_ltx23_model_name_detection_accepts_hyphenated_model_ids():
    assert is_ltx23_model("dg845/LTX-2.3-Diffusers")
    assert is_ltx23_model("/models/ltx23-local")
    assert not is_ltx23_model("Lightricks/LTX-2.3")
    assert not is_ltx2_model("Lightricks/LTX-2.3")


def test_ltx23_text_to_video_defaults_select_pipeline_and_valid_smoke_shape():
    assert default_text_to_video_class_name("dg845/LTX-2.3-Diffusers") == "LTX23Pipeline"
    assert default_text_to_video_class_name("Lightricks/LTX-2.3") is None

    preset = detect_text_to_video_preset("dg845/LTX-2.3-Diffusers")

    assert (preset.height, preset.width) == (384, 512)
    assert preset.num_frames == 25
    assert (preset.num_frames - 1) % 8 == 0
    assert preset.num_inference_steps == 20
    assert preset.output == "ltx23_output.mp4"


def test_ltx23_image_to_video_defaults_select_i2v_pipeline():
    assert default_image_to_video_class_name("dg845/LTX-2.3-Diffusers") == "LTX23ImageToVideoPipeline"
    assert default_image_to_video_class_name("Lightricks/LTX-2.3") is None
    assert default_image_to_video_class_name("Lightricks/LTX-2") == "LTX2ImageToVideoPipeline"


def test_text_to_video_sampling_kwargs_pass_generation_frame_rate():
    from examples.offline_inference.text_to_video.text_to_video import build_text_to_video_sampling_kwargs

    generator = torch.Generator(device="cpu")
    args = SimpleNamespace(
        height=384,
        width=512,
        guidance_scale=3.0,
        guidance_scale_high=None,
        num_inference_steps=20,
        num_frames=25,
        fps=12,
        frame_rate=12.5,
    )

    sampling_kwargs = build_text_to_video_sampling_kwargs(args, generator)

    assert sampling_kwargs["fps"] == 12
    assert sampling_kwargs["frame_rate"] == 12.5
    assert sampling_kwargs["generator"] is generator
