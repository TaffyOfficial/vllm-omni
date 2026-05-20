# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_tokenizer import (
    TokenizerEncodeOutput,
)
from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
    HunyuanImage3Pipeline,
)
from vllm_omni.diffusion.worker.utils import DiffusionRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _make_state(*, prompts=None, num_outputs_per_prompt: int = 1) -> DiffusionRequestState:
    return DiffusionRequestState(
        req_id="req",
        prompts=["prompt"] if prompts is None else prompts,
        sampling=OmniDiffusionSamplingParams(num_outputs_per_prompt=num_outputs_per_prompt),
    )


def test_hunyuan_image3_declares_step_execution_support():
    assert HunyuanImage3Pipeline.supports_step_execution is True


def test_hunyuan_image3_step_execution_rejects_multi_prompt_request():
    pipeline = object.__new__(HunyuanImage3Pipeline)
    state = _make_state(prompts=["a", "b"])

    with pytest.raises(ValueError, match="one prompt per request"):
        pipeline._ensure_step_execution_supported(state)


def test_hunyuan_image3_step_execution_rejects_multi_output_request():
    pipeline = object.__new__(HunyuanImage3Pipeline)
    state = _make_state(num_outputs_per_prompt=2)

    with pytest.raises(ValueError, match="num_outputs_per_prompt"):
        pipeline._ensure_step_execution_supported(state)


def test_hunyuan_image3_step_execution_rejects_sequence_parallel():
    pipeline = object.__new__(HunyuanImage3Pipeline)
    pipeline.od_config = SimpleNamespace(parallel_config=SimpleNamespace(sequence_parallel_size=2))
    state = _make_state()

    with pytest.raises(ValueError, match="sequence parallelism"):
        pipeline._ensure_step_execution_supported(state)


@pytest.mark.parametrize("field_name", ["timesteps", "sigmas"])
def test_hunyuan_image3_step_execution_rejects_custom_schedule(field_name):
    pipeline = object.__new__(HunyuanImage3Pipeline)
    pipeline.od_config = SimpleNamespace(parallel_config=SimpleNamespace(sequence_parallel_size=1))
    state = _make_state()
    setattr(state.sampling, field_name, [1.0])

    with pytest.raises(ValueError, match="custom timesteps or sigmas"):
        pipeline._ensure_step_execution_supported(state)


def test_hunyuan_image3_step_execution_merges_tokenizer_output_and_skips_generation_config():
    pipeline = object.__new__(HunyuanImage3Pipeline)
    states = [_make_state(), _make_state()]
    for idx, state in enumerate(states):
        state.step_index = 0
        state.extra = {
            "cfg_factor": 1,
            "input_ids": torch.tensor([[idx, idx + 10]]),
            "model_kwargs": {
                "position_ids": torch.tensor([[0, 1]]),
                "tokenizer_output": TokenizerEncodeOutput(
                    tokens=torch.tensor([[idx, idx + 10]]),
                    real_pos=torch.tensor([2]),
                ),
                "eos_token_id": [1, 2, 3],
                "max_new_tokens": 4,
            },
        }

    input_ids, model_kwargs, cfg_factor = pipeline._merge_step_model_inputs(states)

    assert cfg_factor == 1
    assert input_ids.tolist() == [[0, 10], [1, 11]]
    assert model_kwargs["position_ids"].tolist() == [[0, 1], [0, 1]]
    assert isinstance(model_kwargs["tokenizer_output"], TokenizerEncodeOutput)
    assert model_kwargs["tokenizer_output"].tokens.tolist() == [[0, 10], [1, 11]]
    assert model_kwargs["tokenizer_output"].real_pos.tolist() == [2, 2]
    assert "eos_token_id" not in model_kwargs
    assert "max_new_tokens" not in model_kwargs


def test_hunyuan_image3_step_execution_pads_different_prompt_lengths():
    pipeline = object.__new__(HunyuanImage3Pipeline)
    pipeline._tkwrapper = SimpleNamespace(pad_token_id=99)
    states = [_make_state(), _make_state()]
    seq_lens = [3, 2]
    for idx, (state, seq_len) in enumerate(zip(states, seq_lens)):
        state.step_index = 0
        state.extra = {
            "cfg_factor": 1,
            "input_ids": torch.arange(seq_len).unsqueeze(0) + idx * 10,
            "model_kwargs": {
                "attention_mask": torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool),
                "position_ids": torch.arange(seq_len).unsqueeze(0),
                "image_mask": torch.ones((1, seq_len), dtype=torch.bool),
                "custom_pos_emb": (
                    torch.ones((1, seq_len, 2)) * (idx + 1),
                    torch.ones((1, seq_len, 2)) * (idx + 3),
                ),
                "tokenizer_output": TokenizerEncodeOutput(
                    tokens=torch.arange(seq_len).unsqueeze(0) + idx * 10,
                    text_mask=torch.ones((1, seq_len)),
                    real_pos=torch.tensor([seq_len]),
                ),
            },
        }

    input_ids, model_kwargs, cfg_factor = pipeline._merge_step_model_inputs(states)

    assert cfg_factor == 1
    assert input_ids.tolist() == [[0, 1, 2], [10, 11, 99]]
    assert model_kwargs["attention_mask"].shape == (2, 1, 3, 3)
    assert model_kwargs["attention_mask"][1, 0].tolist() == [
        [True, True, False],
        [True, True, False],
        [False, False, False],
    ]
    assert model_kwargs["position_ids"].tolist() == [[0, 1, 2], [0, 1, 0]]
    assert model_kwargs["image_mask"].tolist() == [[True, True, True], [True, True, False]]
    assert model_kwargs["query_lens"] == [3, 3]
    assert model_kwargs["seq_lens"] == [3, 3]
    assert model_kwargs["custom_pos_emb"][0].shape == (2, 3, 2)
    assert model_kwargs["custom_pos_emb"][0][1, 2].tolist() == [0, 0]
    assert model_kwargs["tokenizer_output"].tokens.tolist() == [[0, 1, 2], [10, 11, 99]]
    assert model_kwargs["tokenizer_output"].text_mask.tolist() == [[1, 1, 1], [1, 1, 0]]
    assert model_kwargs["tokenizer_output"].real_pos.tolist() == [3, 2]
