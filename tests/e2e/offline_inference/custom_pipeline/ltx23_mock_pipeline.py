# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Lightweight custom pipeline for LTX-2.3 L2 shape tests."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch import nn

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.request import OmniDiffusionRequest


class LTX23MockPipelineForTest(nn.Module):
    """Return deterministic in-memory video frames without loading LTX weights."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__()
        self.od_config = od_config
        self.prefix = prefix
        self.device = get_local_device()

    def load_weights(self, weights) -> set[str]:
        return set()

    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        sp = req.sampling_params
        height = int(sp.height or 256)
        width = int(sp.width or 256)
        num_frames = int(sp.num_frames or 5)
        fps = int(sp.fps or sp.frame_rate or 8)

        frames: list[Image.Image] = []
        for frame_idx in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[..., 0] = (frame_idx * 37) % 256
            frame[..., 1] = np.arange(width, dtype=np.uint16)[None, :] % 256
            frame[..., 2] = np.arange(height, dtype=np.uint16)[:, None] % 256
            frames.append(Image.fromarray(frame, mode="RGB"))

        duration_s = max(float(num_frames) / float(fps), 0.125)
        audio_samples = max(1, int(duration_s * 24000))
        audio = torch.zeros(audio_samples, dtype=torch.float32, device=self.device)

        return DiffusionOutput(
            output={
                "video": [frames],
                "audio": [audio.detach().cpu()],
                "audio_sample_rate": 24000,
                "fps": fps,
                "custom_output": {
                    "mock_model": "ltx2.3",
                    "video_shape": [num_frames, height, width, 3],
                },
            }
        )
