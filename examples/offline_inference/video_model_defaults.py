# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VideoGenerationPreset:
    height: int
    width: int
    num_frames: int
    num_inference_steps: int
    guidance_scale: float
    fps: int
    output: str


_MODEL_PRESETS = {
    "wan": VideoGenerationPreset(
        height=720,
        width=1280,
        num_frames=81,
        num_inference_steps=40,
        guidance_scale=4.0,
        fps=24,
        output="wan22_output.mp4",
    ),
    "hunyuan": VideoGenerationPreset(
        height=480,
        width=832,
        num_frames=121,
        num_inference_steps=50,
        guidance_scale=6.0,
        fps=24,
        output="hunyuan_video_15_output.mp4",
    ),
    "ltx2": VideoGenerationPreset(
        height=480,
        width=768,
        num_frames=41,
        num_inference_steps=20,
        guidance_scale=4.0,
        fps=24,
        output="ltx2_output.mp4",
    ),
    "ltx23": VideoGenerationPreset(
        height=384,
        width=512,
        num_frames=25,
        num_inference_steps=20,
        guidance_scale=4.0,
        fps=24,
        output="ltx23_output.mp4",
    ),
}

_LTX23_RAW_CHECKPOINT_MODEL_IDS = {
    "lightricks/ltx-2.3",
}


def _normalize_model_name(value: str | None) -> str:
    return (value or "").lower().replace("-", "").replace("_", "").replace(".", "")


def _is_unsupported_ltx23_raw_checkpoint_model(model: str | None) -> bool:
    return model is not None and model.lower() in _LTX23_RAW_CHECKPOINT_MODEL_IDS


def is_ltx23_model(model: str | None, model_class_name: str | None = None) -> bool:
    if _is_unsupported_ltx23_raw_checkpoint_model(model) and model_class_name is None:
        return False
    model_key = _normalize_model_name(model)
    class_key = _normalize_model_name(model_class_name)
    return "ltx23" in model_key or "ltx23" in class_key


def is_ltx2_model(model: str | None, model_class_name: str | None = None) -> bool:
    if _is_unsupported_ltx23_raw_checkpoint_model(model) and model_class_name is None:
        return False
    model_key = _normalize_model_name(model)
    class_key = _normalize_model_name(model_class_name)
    return is_ltx23_model(model, model_class_name) or "ltx2" in model_key or "ltx2" in class_key


def detect_text_to_video_preset(model: str | None, model_class_name: str | None = None) -> VideoGenerationPreset:
    model_lower = (model or "").lower()
    if is_ltx23_model(model, model_class_name):
        return _MODEL_PRESETS["ltx23"]
    if is_ltx2_model(model, model_class_name):
        return _MODEL_PRESETS["ltx2"]
    if "hunyuan" in model_lower:
        return _MODEL_PRESETS["hunyuan"]
    return _MODEL_PRESETS["wan"]


def default_text_to_video_class_name(model: str | None, model_class_name: str | None = None) -> str | None:
    if model_class_name is not None:
        return model_class_name
    if is_ltx23_model(model):
        return "LTX23Pipeline"
    return None


def default_image_to_video_class_name(model: str | None, model_class_name: str | None = None) -> str | None:
    if model_class_name is not None:
        return model_class_name
    if is_ltx23_model(model):
        return "LTX23ImageToVideoPipeline"
    if is_ltx2_model(model):
        return "LTX2ImageToVideoPipeline"
    return None
