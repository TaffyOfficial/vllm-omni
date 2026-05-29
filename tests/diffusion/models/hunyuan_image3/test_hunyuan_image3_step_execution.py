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
        request_id="req",
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
            "guidance_scale": 1.0,
            "input_ids": torch.tensor([[idx, idx + 10]]),
            "model_kwargs": {
                "position_ids": torch.tensor([[0, 1]]),
                "tokenizer_output": TokenizerEncodeOutput(
                    tokens=torch.tensor([[idx, idx + 10]]),
                    real_pos=torch.tensor([2]),
                ),
                "eos_token_id": [1, 2, 3],
                "max_new_tokens": 4,
                "num_inference_steps": 2 + idx,
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
    assert "num_inference_steps" not in model_kwargs


def test_hunyuan_image3_step_execution_pads_different_prompt_lengths():
    pipeline = object.__new__(HunyuanImage3Pipeline)
    pipeline._tkwrapper = SimpleNamespace(pad_token_id=99)
    states = [_make_state(), _make_state()]
    seq_lens = [3, 2]
    for idx, (state, seq_len) in enumerate(zip(states, seq_lens)):
        state.step_index = 0
        state.extra = {
            "cfg_factor": 1,
            "guidance_scale": 1.0,
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


def test_hunyuan_image3_step_execution_merges_none_input_ids_after_first_step():
    pipeline = object.__new__(HunyuanImage3Pipeline)
    states = [_make_state(), _make_state()]
    for state in states:
        state.step_index = 1
        state.extra = {
            "cfg_factor": 1,
            "guidance_scale": 1.0,
            "input_ids": None,
            "model_kwargs": {
                "attention_mask": torch.ones((1, 1, 2, 2), dtype=torch.bool),
                "position_ids": torch.tensor([[0, 1]]),
            },
        }

    input_ids, model_kwargs, cfg_factor = pipeline._merge_step_model_inputs(states)

    assert input_ids is None
    assert cfg_factor == 1
    assert model_kwargs["position_ids"].tolist() == [[0, 1], [0, 1]]


def test_hunyuan_image3_step_execution_rejects_mixed_guidance_scales():
    pipeline = object.__new__(HunyuanImage3Pipeline)
    states = [_make_state(), _make_state()]
    for idx, state in enumerate(states):
        state.step_index = 1
        state.extra = {
            "cfg_factor": 2,
            "guidance_scale": 2.5 + idx,
            "input_ids": torch.tensor([[0, 1], [0, 1]]),
            "model_kwargs": {
                "attention_mask": torch.ones((2, 1, 2, 2), dtype=torch.bool),
                "position_ids": torch.tensor([[0, 1], [0, 1]]),
            },
        }

    with pytest.raises(ValueError, match="mixed guidance scales"):
        pipeline._merge_step_model_inputs(states)


def test_hunyuan_image3_denoise_updates_model_kwargs_until_each_state_is_final(monkeypatch):
    pipeline = object.__new__(HunyuanImage3Pipeline)
    monkeypatch.setattr(HunyuanImage3Pipeline, "device", property(lambda self: torch.device("cpu")), raising=False)
    pipeline._pipeline = SimpleNamespace(_guidance_scale=1.0)
    states = [_make_state(), _make_state()]
    for state, total_steps in zip(states, [2, 4]):
        state.step_index = 1
        state.timesteps = torch.arange(total_steps)
        state.latents = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
        state.extra = {
            "cfg_factor": 1,
            "guidance_scale": 1.0,
            "input_ids": torch.tensor([[0, 1]]),
            "model_kwargs": {
                "attention_mask": torch.ones((1, 1, 2, 2), dtype=torch.bool),
                "position_ids": torch.tensor([[0, 1]]),
            },
        }

    split_calls = []
    monkeypatch.setattr(pipeline, "_restore_prompt_kv_cache", lambda states, cfg_factor: None)
    monkeypatch.setattr(
        pipeline,
        "prepare_inputs_for_generation",
        lambda input_ids, images, timestep, **model_kwargs: {},
    )
    monkeypatch.setattr(
        pipeline,
        "forward_call",
        lambda **kwargs: {"diffusion_prediction": torch.ones((2, 1, 2, 2), dtype=torch.float32)},
    )
    monkeypatch.setattr(
        pipeline,
        "_update_model_kwargs_for_generation",
        lambda model_output, model_kwargs: model_kwargs,
    )
    monkeypatch.setattr(
        pipeline,
        "_split_step_model_inputs",
        lambda states, input_ids, model_kwargs, cfg_factor: split_calls.append(
            ([state.request_id for state in states], input_ids)
        ),
    )
    input_batch = SimpleNamespace(
        states=states,
        latents=torch.zeros((2, 1, 2, 2), dtype=torch.float32),
        timesteps=torch.tensor([1, 1]),
    )

    pred = pipeline.denoise_step(input_batch)

    assert pred.shape == (2, 1, 2, 2)
    assert split_calls == [(["req", "req"], None)]


def test_hunyuan_image3_restore_prompt_kv_cache_pads_variable_full_cache_lengths():
    pipeline = object.__new__(HunyuanImage3Pipeline)
    cache_owner = SimpleNamespace(image_kv_cache_map=None)
    pipeline.model = SimpleNamespace(
        layers=[SimpleNamespace(layer_idx=0, self_attn=SimpleNamespace(image_attn=cache_owner))]
    )
    states = [_make_state(), _make_state()]
    key_a = torch.ones((1, 3, 1, 2))
    value_a = key_a + 10
    key_b = torch.ones((1, 2, 1, 2)) * 2
    value_b = key_b + 10
    for state, key, value in zip(states, [key_a, key_b], [value_a, value_b]):
        state.step_index = 1
        state.extra = {"prompt_kv_cache": {0: [(key, value)]}}

    pipeline._restore_prompt_kv_cache(states, cfg_factor=1)

    merged_key, merged_value = cache_owner.image_kv_cache_map
    assert merged_key.shape == (2, 3, 1, 2)
    assert merged_value.shape == (2, 3, 1, 2)
    assert torch.allclose(merged_key[0], key_a[0])
    assert torch.allclose(merged_key[1, :2], key_b[0])
    assert torch.allclose(merged_key[1, 2], torch.zeros((1, 2)))
    assert torch.allclose(merged_value[1, 2], torch.zeros((1, 2)))


def test_hunyuan_image3_step_scheduler_keeps_latents_float32():
    class FakeScheduler:
        def step(self, noise_pred, timestep, sample, **kwargs):
            del noise_pred, timestep, kwargs
            return (sample.to(dtype=torch.float64) + 1,)

    pipeline = object.__new__(HunyuanImage3Pipeline)
    pipeline._pipeline = SimpleNamespace(prepare_extra_func_kwargs=lambda func, kwargs: {})
    state = _make_state()
    state.latents = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    state.timesteps = torch.tensor([1.0])
    state.extra = {"scheduler": FakeScheduler(), "generator": None}

    pipeline.step_scheduler(state, torch.ones_like(state.latents))

    assert state.step_index == 1
    assert state.latents.dtype == torch.float32
    assert state.latents.tolist() == [[[[1.0, 1.0], [1.0, 1.0]]]]
