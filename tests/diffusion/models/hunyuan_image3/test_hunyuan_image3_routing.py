# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for HunyuanImage3 AR routing helpers."""

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_unpack_packed_topk_accepts_legacy_four_arg_routing_call():
    from vllm_omni.model_executor.models.hunyuan_image3.hunyuan_image3 import (
        _hunyuan_image3_unpack_packed_topk,
    )

    gating_output = torch.tensor([[0.25, 0.75, 1.0, 0.0]], dtype=torch.float32)

    topk_weights, topk_indices = _hunyuan_image3_unpack_packed_topk(
        torch.empty(1, 1),
        gating_output,
        2,
        False,
    )

    assert torch.equal(topk_weights, torch.tensor([[0.25, 0.75]], dtype=torch.float32))
    assert torch.equal(topk_indices, torch.tensor([[1, 0]], dtype=torch.int32))
