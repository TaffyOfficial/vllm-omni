# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Official-alignment tests for HunyuanImage3 infer_align_image_size."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
    HunyuanImage3ImageProcessor,
    ImageInfo,
    JointImageInfo,
)
from vllm_omni.model_executor.models.hunyuan_image3.hunyuan_image3 import (
    HunyuanImage3Processor,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _gradient_image(width: int = 8, height: int = 4) -> Image.Image:
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :, 0] = np.arange(width, dtype=np.uint8) * 25
    arr[:, :, 1] = np.arange(height, dtype=np.uint8)[:, None] * 60
    arr[:, :, 2] = 17
    return Image.fromarray(arr, mode="RGB")


def test_image_processor_default_center_crop_differs_from_resize():
    src = _gradient_image()

    official_default = HunyuanImage3ImageProcessor._resize_and_crop(src, (4, 4), crop_type="center")
    resize_path = HunyuanImage3ImageProcessor._resize_and_crop(src, (4, 4), crop_type="resize")

    assert official_default.size == (4, 4)
    assert resize_path.size == (4, 4)
    assert not np.array_equal(np.asarray(official_default), np.asarray(resize_path))


def test_image_processor_infer_align_resize_mode_is_direct_resize():
    src = _gradient_image()

    official_resize = HunyuanImage3ImageProcessor._resize_and_crop(src, (4, 4), crop_type="resize")
    center_crop = HunyuanImage3ImageProcessor._resize_and_crop(src, (4, 4), crop_type="center")

    assert official_resize.size == (4, 4)
    expected_resize = src.resize((4, 4), resample=Image.Resampling.LANCZOS)
    assert np.array_equal(np.asarray(official_resize), np.asarray(expected_resize))
    assert not np.array_equal(np.asarray(official_resize), np.asarray(center_crop))


class _FixedTargetResolutionGroup:
    def get_target_size(self, width: int, height: int):
        return 4, 4

    def get_base_size_and_ratio_index(self, width: int, height: int):
        return 1024, 0


def _fake_vae_processor(image: Image.Image):
    arr = np.asarray(image, dtype=np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor.unsqueeze(0)


def _fake_vit_processor(_image: Image.Image):
    return {
        "pixel_values": torch.zeros((1, 1, 3), dtype=torch.float32),
        "pixel_attention_mask": torch.ones((1, 1), dtype=torch.bool),
        "spatial_shapes": torch.tensor([[1, 1]], dtype=torch.long),
    }


@pytest.mark.parametrize(
    ("infer_align_image_size", "expected_image"),
    [
        (False, lambda src: HunyuanImage3ImageProcessor._resize_and_crop(src, (4, 4), crop_type="center")),
        (True, lambda src: HunyuanImage3ImageProcessor._resize_and_crop(src, (4, 4), crop_type="resize")),
    ],
)
def test_ar_process_image_uses_official_crop_mode_and_preserves_original_size(
    infer_align_image_size: bool,
    expected_image,
):
    src = _gradient_image()
    processor = object.__new__(HunyuanImage3Processor)
    processor.infer_align_image_size = infer_align_image_size
    processor.hf_config = SimpleNamespace(
        vit={"num_channels": 3},
        vae_downsample_factor=(1, 1),
        patch_size=1,
    )
    processor.reso_group = _FixedTargetResolutionGroup()
    processor.vae_processor = _fake_vae_processor
    processor.vision_encoder_processor = _fake_vit_processor

    result = processor.process_image(src)

    expected_tensor = _fake_vae_processor(expected_image(src)).squeeze(0).reshape(-1)
    assert torch.equal(result["vae_pixel_values"], expected_tensor)
    assert result["ori_image_width"].tolist() == [src.width]
    assert result["ori_image_height"].tolist() == [src.height]


class _FakeResolutionGroup:
    base_size = 1024

    def __init__(self) -> None:
        self._ratios = {
            0: 1.0,
            1: 0.75,
            2: 2.0,
        }

    def __getitem__(self, idx: int):
        return SimpleNamespace(ratio=self._ratios[idx])

    def get_base_size_and_ratio_index(self, width: int, height: int):
        ratio = height / width
        if abs(ratio - 0.75) < 0.01:
            return self.base_size, 1
        if abs(ratio - 2.0) < 0.01:
            return self.base_size, 2
        return self.base_size, 0


def _fake_processor():
    processor = object.__new__(HunyuanImage3ImageProcessor)
    processor.reso_group = _FakeResolutionGroup()
    return processor


def _joint_cond(
    *,
    ratio_index: int,
    ori_width: int,
    ori_height: int,
) -> JointImageInfo:
    vae = ImageInfo(
        image_type="vae",
        image_width=1024,
        image_height=1024,
        token_width=64,
        token_height=64,
        base_size=1024,
        ratio_index=ratio_index,
        ori_image_width=ori_width,
        ori_image_height=ori_height,
    )
    vit = ImageInfo(
        image_type="siglip2",
        image_width=1024,
        image_height=1024,
        token_width=64,
        token_height=64,
        image_token_length=4096,
        ori_image_width=ori_width,
        ori_image_height=ori_height,
    )
    return JointImageInfo(vae_image_info=vae, vision_image_info=vit)


def test_postprocess_single_matching_bucket_resizes_to_input_ratio_area():
    output = Image.new("RGB", (1024, 1024), color="white")
    cond = _joint_cond(ratio_index=0, ori_width=1200, ori_height=800)

    processed = _fake_processor().postprocess_outputs([output], [[cond]], infer_align_image_size=True)

    assert processed[0].size == (1254, 836)


def test_postprocess_empty_and_mismatched_buckets_keep_outputs_unchanged():
    no_image_output = Image.new("RGB", (1024, 1024), color="white")
    mismatch_output = Image.new("RGB", (1024, 1024), color="white")
    cond = _joint_cond(ratio_index=1, ori_width=1200, ori_height=800)

    processed = _fake_processor().postprocess_outputs(
        [no_image_output, mismatch_output],
        [[], [cond]],
        infer_align_image_size=True,
    )

    assert processed[0].size == (1024, 1024)
    assert processed[1].size == (1024, 1024)


def test_postprocess_multi_image_uses_first_matching_bucket_only():
    output = Image.new("RGB", (1024, 1024), color="white")
    mismatched = _joint_cond(ratio_index=1, ori_width=1600, ori_height=900)
    matched = _joint_cond(ratio_index=0, ori_width=1600, ori_height=900)

    processed = _fake_processor().postprocess_outputs(
        [output],
        [[mismatched, matched]],
        infer_align_image_size=True,
    )

    assert processed[0].size == (1365, 768)


def test_postprocess_returns_outputs_when_batch_has_no_cond_info():
    outputs = [Image.new("RGB", (1024, 1024), color="white")]

    assert _fake_processor().postprocess_outputs(outputs, None, infer_align_image_size=True) is outputs
