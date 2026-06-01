# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.hunyuan_image3 import pipeline_hunyuan_image3 as hy3_pipeline_mod
from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_tokenizer import (
    TokenizerEncodeOutput,
)
from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
    ImageInfo,
    JointImageInfo,
)
from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
    _STEP_AR_KV,
    _STEP_CFG_FACTOR,
    _STEP_GENERATOR,
    _STEP_GUIDANCE_SCALE,
    _STEP_INPUT_IDS,
    _STEP_MODEL_KWARGS,
    _STEP_PROMPT_KV,
    HunyuanImage3Pipeline,
)
from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.utils import DiffusionRequestState


def _pipeline():
    pipeline = object.__new__(HunyuanImage3Pipeline)
    pipeline._tkwrapper = SimpleNamespace(pad_token_id=0)
    return pipeline


def _state(request_id: str, step_index: int) -> DiffusionRequestState:
    state = DiffusionRequestState(
        request_id=request_id,
        sampling=SimpleNamespace(),
        prompts=["prompt"],
    )
    state.step_index = step_index
    state.timesteps = torch.tensor([1.0, 0.5, 0.25, 0.0])
    state.latents = torch.zeros(1, 4, 8, 8)
    state.extra = {
        _STEP_CFG_FACTOR: 1,
        _STEP_AR_KV: None,
        _STEP_INPUT_IDS: None,
        _STEP_GUIDANCE_SCALE: 1.0,
        _STEP_MODEL_KWARGS: {
            "num_image_tokens": 17,
            "ar_kv_reuse_len": 0,
        },
    }
    return state


def _joint_image_info() -> JointImageInfo:
    image_info = ImageInfo(
        image_type="vae",
        image_width=1,
        image_height=1,
        token_width=1,
        token_height=1,
        image_token_length=1,
        base_size=1,
        ratio_index=0,
    )
    vision_info = ImageInfo(
        image_type="vision",
        image_width=1,
        image_height=1,
        token_width=1,
        token_height=1,
        image_token_length=1,
        base_size=1,
        ratio_index=0,
    )
    return JointImageInfo(image_info, vision_info, {})


def test_prepare_encode_uses_bucketed_image_info_size(monkeypatch):
    pipeline = _pipeline()
    monkeypatch.setattr(HunyuanImage3Pipeline, "device", property(lambda self: torch.device("cpu")))
    pipeline.config = SimpleNamespace(vae={"latent_channels": 4})
    pipeline.generation_config = SimpleNamespace()
    pipeline.scheduler = SimpleNamespace()

    class FakeStepPipe:
        def __init__(self):
            self.latent_image_size = None

        def prepare_latents(self, **kwargs):
            self.latent_image_size = kwargs["image_size"]
            return torch.zeros(1, 4, 128, 128, dtype=torch.bfloat16)

        def _maybe_handle_ar_kv_reuse(self, input_ids, model_kwargs, **kwargs):
            return input_ids, 0

    step_pipe = FakeStepPipe()
    pipeline._pipeline = step_pipe
    pipeline._validate_step_request = lambda state: None
    pipeline._extract_step_prompt_inputs = lambda state: (["prompt"], [None], None, None)
    pipeline._extract_ar_kv_from_sampling = lambda sampling: {}
    pipeline._snapshot_injected_ar_kv = lambda: None
    pipeline._prepare_attention_mask_for_generation = lambda input_ids, generation_config, model_kwargs: torch.ones(
        1, 1, 3, 5, dtype=torch.bool
    )
    image_info = ImageInfo(
        image_type="vae",
        image_width=1024,
        image_height=1024,
        token_width=64,
        token_height=64,
        image_token_length=4096,
        base_size=1024,
        ratio_index=0,
    )
    pipeline.prepare_model_inputs = lambda **kwargs: {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "batch_gen_image_info": [image_info],
        "generator": kwargs["generator"],
    }
    monkeypatch.setattr(
        hy3_pipeline_mod,
        "retrieve_timesteps",
        lambda scheduler, num_inference_steps, device, timesteps, sigmas: (torch.arange(2), None),
    )

    state = DiffusionRequestState(
        request_id="bucketed-size",
        sampling=SimpleNamespace(
            height=512,
            width=512,
            num_inference_steps=2,
            guidance_scale=1.0,
            guidance_scale_provided=True,
            guidance_rescale=0.0,
            generator=None,
        ),
        prompts=["prompt"],
    )

    state = pipeline.prepare_encode(state)

    assert step_pipe.latent_image_size == [1024, 1024]


def test_hunyuan_step_group_key_ignores_step_index_for_later_steps():
    pipeline = _pipeline()
    states = [_state("req-0", 1), _state("req-1", 3)]

    groups = pipeline._split_step_groups(states)

    assert len(groups) == 1
    assert [state.request_id for state in groups[0]] == ["req-0", "req-1"]


def test_step_scheduler_preserves_latent_dtype_for_mixed_progress_batches():
    pipeline = _pipeline()
    pipeline._pipeline = SimpleNamespace(prepare_extra_func_kwargs=lambda step, kwargs: {})

    class FakeScheduler:
        def step(self, noise_pred, timestep, latents, **kwargs):
            del timestep, kwargs
            return (latents.float() + noise_pred.float(),)

    state = _state("req", 0)
    state.timesteps = torch.tensor([1.0])
    state.scheduler = FakeScheduler()
    state.latents = torch.zeros(1, 4, 8, 8, dtype=torch.bfloat16)
    state.extra[_STEP_GENERATOR] = None

    pipeline.step_scheduler(state, torch.ones_like(state.latents, dtype=torch.float32))

    assert state.latents.dtype == torch.bfloat16
    assert state.step_index == 1


def test_dummy_warmup_ignores_diffusion_engine_dummy_image():
    pipeline = _pipeline()
    state = _state(DUMMY_DIFFUSION_REQUEST_ID, 0)
    state.prompts = [
        {
            "prompt": "dummy run",
            "additional_information": {
                "batch_cond_image_info": [_joint_image_info()],
            },
        }
    ]

    prompt, _, _, batch_cond_image_info = pipeline._extract_step_prompt_inputs(state)

    assert prompt == ["dummy run"]
    assert batch_cond_image_info is None


def test_real_image_edit_still_fails_fast_in_step_execution():
    pipeline = _pipeline()
    state = _state("real-edit", 0)
    state.prompts = [
        {
            "prompt": "edit",
            "additional_information": {
                "batch_cond_image_info": [_joint_image_info()],
            },
        }
    ]

    with pytest.raises(ValueError, match="does not support image editing"):
        pipeline._extract_step_prompt_inputs(state)


def test_later_step_merge_shifts_spans_without_polluting_request_state():
    pipeline = _pipeline()
    states = [_state("short", 2), _state("long", 4)]
    states[0].extra[_STEP_MODEL_KWARGS].update(
        {
            "attention_mask": torch.ones(1, 1, 3, 5, dtype=torch.bool),
            "full_attn_spans": [[(2, 5)]],
        }
    )
    states[1].extra[_STEP_MODEL_KWARGS].update(
        {
            "attention_mask": torch.ones(1, 1, 3, 7, dtype=torch.bool),
            "full_attn_spans": [[(4, 7)]],
        }
    )
    states[0].extra[_STEP_PROMPT_KV] = [{"lens": torch.tensor([2])}]
    states[1].extra[_STEP_PROMPT_KV] = [{"lens": torch.tensor([4])}]

    row_state_indexes = [0, 1]
    row_branches = [0, 0]
    _, merged = pipeline._merge_step_model_inputs(
        states,
        row_state_indexes,
        row_branches,
        first_step=False,
    )

    assert merged["attention_mask"].shape == (2, 1, 3, 7)
    assert merged["full_attn_spans"] == [[(4, 7)], [(4, 7)]]

    pipeline._split_merged_kwargs_to_states(states, merged, row_state_indexes, row_branches)

    assert states[0].extra[_STEP_MODEL_KWARGS]["attention_mask"].shape == (1, 1, 3, 5)
    assert states[1].extra[_STEP_MODEL_KWARGS]["attention_mask"].shape == (1, 1, 3, 7)
    assert states[0].extra[_STEP_MODEL_KWARGS]["full_attn_spans"] == [[(2, 5)]]
    assert states[1].extra[_STEP_MODEL_KWARGS]["full_attn_spans"] == [[(4, 7)]]


def test_first_step_merge_keeps_tokenizer_output_for_next_step_update():
    pipeline = _pipeline()
    states = [_state("req-0", 0), _state("req-1", 0)]
    for idx, state in enumerate(states):
        state.extra[_STEP_MODEL_KWARGS].update(
            {
                "attention_mask": torch.ones(1, 1, 4, 4, dtype=torch.bool),
                "tokenizer_output": TokenizerEncodeOutput(
                    tokens=torch.full((1, 4), idx),
                    gen_image_mask=torch.tensor([[False, True, True, True]]),
                    gen_image_slices=[slice(1, 4)],
                ),
            }
        )

    _, merged = pipeline._merge_step_model_inputs(
        states,
        row_state_indexes=[0, 1],
        row_branches=[0, 0],
        first_step=True,
    )

    tokenizer_output = merged["tokenizer_output"]
    assert tokenizer_output.tokens.tolist() == [[0, 0, 0, 0], [1, 1, 1, 1]]
    assert tokenizer_output.gen_image_slices == [slice(1, 4), slice(1, 4)]


def test_later_step_merge_allows_request_local_step_counts_and_guidance_values():
    pipeline = _pipeline()
    states = [_state("req-0", 1), _state("req-1", 3)]
    for idx, state in enumerate(states):
        state.extra[_STEP_MODEL_KWARGS].update(
            {
                "attention_mask": torch.ones(1, 1, 2, 4, dtype=torch.bool),
                "full_attn_spans": [[(2, 4)]],
                "guidance_scale": 3.0 + idx,
                "num_inference_steps": 20 + idx,
            }
        )
        state.extra[_STEP_PROMPT_KV] = [{"lens": torch.tensor([2])}]

    _, merged = pipeline._merge_step_model_inputs(
        states,
        row_state_indexes=[0, 1],
        row_branches=[0, 0],
        first_step=False,
    )

    assert "guidance_scale" not in merged
    assert "num_inference_steps" not in merged


def test_merge_ignores_request_local_step_kwargs():
    pipeline = _pipeline()
    states = [_state("req-0", 0), _state("req-1", 0)]
    for step_count, state in zip((20, 30), states):
        state.extra[_STEP_MODEL_KWARGS].update(
            {
                "attention_mask": torch.ones(1, 1, 3, 3, dtype=torch.bool),
                "num_inference_steps": step_count,
                "guidance_scale": 5.0,
            }
        )

    _, merged = pipeline._merge_step_model_inputs(
        states,
        row_state_indexes=[0, 1],
        row_branches=[0, 0],
        first_step=True,
    )

    assert "num_inference_steps" not in merged
    assert "guidance_scale" not in merged


@pytest.mark.parametrize(
    ("request_id", "mutate_state", "error_match"),
    [
        pytest.param(
            "broken-req",
            lambda state: state.extra.pop(_STEP_MODEL_KWARGS),
            "broken-req",
            id="missing-model-kwargs",
        ),
        pytest.param(
            "bad-cfg",
            lambda state: state.extra.__setitem__(_STEP_CFG_FACTOR, 3),
            "bad-cfg",
            id="unsupported-cfg-factor",
        ),
    ],
)
def test_denoise_step_reports_invalid_group_state_with_request_id(request_id, mutate_state, error_match):
    pipeline = _pipeline()
    state = _state(request_id, 0)
    mutate_state(state)

    with pytest.raises(ValueError, match=error_match):
        pipeline.denoise_step(InputBatch.make_batch([state]))


def test_denoise_step_uses_input_batch_group_order_and_splits_back(monkeypatch):
    pipeline = _pipeline()
    monkeypatch.setattr(HunyuanImage3Pipeline, "device", property(lambda self: torch.device("cpu")))
    states = [_state("req-0", 1), _state("req-1", 3)]
    for idx, state in enumerate(states):
        prefix_len = 2 + idx * 2
        state.latents = torch.full((1, 1), float(idx))
        state.extra[_STEP_CFG_FACTOR] = 2
        state.extra[_STEP_GUIDANCE_SCALE] = 1.0
        state.extra[_STEP_INPUT_IDS] = None
        state.extra[_STEP_MODEL_KWARGS].update(
            {
                "attention_mask": torch.ones(2, 1, 2, prefix_len + 2, dtype=torch.bool),
                "full_attn_spans": [[(prefix_len, prefix_len + 2)], [(prefix_len, prefix_len + 2)]],
            }
        )
        state.extra[_STEP_PROMPT_KV] = [
            {
                "key": torch.zeros(2, prefix_len, 1, 1),
                "value": torch.zeros(2, prefix_len, 1, 1),
                "lens": torch.tensor([prefix_len, prefix_len]),
            }
        ]

    captured = {}

    def fake_restore_prompt_kv_cache(states_arg, row_state_indexes, row_branches):
        del states_arg
        captured["row_state_indexes"] = list(row_state_indexes)
        captured["row_branches"] = list(row_branches)

    def fake_prepare_inputs_for_generation(input_ids, images, timestep, **model_kwargs):
        captured["input_ids"] = input_ids
        captured["images"] = images.clone()
        captured["timestep"] = timestep.clone()
        captured["merged_attention_mask_shape"] = tuple(model_kwargs["attention_mask"].shape)
        captured["merged_full_attn_spans"] = model_kwargs["full_attn_spans"]
        return {"model_kwargs": model_kwargs}

    pipeline._restore_prompt_kv_cache = fake_restore_prompt_kv_cache
    pipeline.prepare_inputs_for_generation = fake_prepare_inputs_for_generation
    pipeline.forward_call = lambda **kwargs: {"diffusion_prediction": torch.tensor([[10.0], [20.0], [1.0], [2.0]])}
    pipeline._update_model_kwargs_for_generation = lambda model_output, model_kwargs: model_kwargs
    pipeline.pipeline = SimpleNamespace(cfg_operator=lambda cond, uncond, scale, step: cond + uncond)

    batch = InputBatch.make_batch(states)
    out = pipeline.denoise_step(batch)

    assert captured["row_state_indexes"] == [0, 1, 0, 1]
    assert captured["row_branches"] == [0, 0, 1, 1]
    assert captured["input_ids"] is None
    assert tuple(captured["images"].shape) == (4, 1)
    assert captured["timestep"].tolist() == [0.5, 0.0, 0.5, 0.0]
    assert captured["merged_attention_mask_shape"] == (4, 1, 2, 6)
    assert captured["merged_full_attn_spans"] == [[(4, 6)], [(4, 6)], [(4, 6)], [(4, 6)]]
    torch.testing.assert_close(out, torch.tensor([[11.0], [22.0]]))
    assert states[0].extra[_STEP_MODEL_KWARGS]["attention_mask"].shape == (2, 1, 2, 4)
    assert states[1].extra[_STEP_MODEL_KWARGS]["attention_mask"].shape == (2, 1, 2, 6)
    assert states[0].extra[_STEP_MODEL_KWARGS]["full_attn_spans"] == [[(2, 4)], [(2, 4)]]
    assert states[1].extra[_STEP_MODEL_KWARGS]["full_attn_spans"] == [[(4, 6)], [(4, 6)]]
