# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any

import pytest
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.sampling_params import SamplingParams

from vllm_omni.entrypoints.openai.diffusion_request_utils import (
    DiffusionChatRequestPlan,
    compile_diffusion_chat_request_plan,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SAMPLING_FIELDS = {
    "height": "height",
    "width": "width",
    "seed": "seed",
    "num_inference_steps": "num_inference_steps",
    "guidance_scale": "guidance_scale",
    "cfg_scale": "true_cfg_scale",
    "true_cfg_scale": "true_cfg_scale",
    "layers": "layers",
}
STANDARD_FIELDS = {"temperature", "max_tokens", "seed"}
CONTROL_FIELDS = {"size", "negative_prompt", "lora", "modalities"}


def _request(**kwargs: Any) -> ChatCompletionRequest:
    return ChatCompletionRequest(model="test", messages=[], **kwargs)


def _compile(
    request: ChatCompletionRequest,
    *,
    stage_types: tuple[str, ...] = ("diffusion",),
    defaults: tuple[object, ...] = (),
    comprehension_stage_index: int | None = None,
    declared: frozenset[str] = frozenset(),
    fan_out_declared: bool = False,
) -> DiffusionChatRequestPlan:
    return compile_diffusion_chat_request_plan(
        request=request,
        stage_types=stage_types,
        default_sampling_params_list=defaults,
        comprehension_stage_index=comprehension_stage_index,
        sampling_root_fields=SAMPLING_FIELDS,
        standard_sampling_fields=STANDARD_FIELDS,
        control_root_fields=CONTROL_FIELDS,
        declared_extra_fields=declared,
        apply_declared_to_non_diffusion=fan_out_declared,
    )


@pytest.mark.parametrize(
    "body",
    [
        {"model_option": 1},
        {"extra_body": {"model_option": 1}},
        {"extra_args": {"model_option": 1}},
        {"extra_params": {"model_option": 1}},
        {"extra_body": {"extra_args": {"model_option": 1}}},
        {"extra_body": {"extra_params": {"model_option": 1}}},
    ],
)
def test_all_global_sources_reach_the_same_diffusion_consumer(body: dict[str, object]) -> None:
    declared = frozenset({"model_option"}) if "model_option" in body or "extra_body" in body else frozenset()
    plan = _compile(_request(**body), declared=declared)

    assert plan.clone_sampling_params_list()[0].extra_args["model_option"] == 1


@pytest.mark.parametrize(
    "body, expected_sources",
    [
        (
            {"seed": 1, "extra_args": {"seed": 2}},
            "request.seed, request.extra_args.seed",
        ),
        (
            {"cfg_scale": 1.0, "extra_args": {"true_cfg_scale": 2.0}},
            "request.cfg_scale, request.extra_args.true_cfg_scale",
        ),
        (
            {"model_option": 1, "extra_body": {"model_option": 2}},
            "request.model_option, request.extra_body.model_option",
        ),
    ],
)
def test_global_duplicate_sources_are_rejected(
    body: dict[str, object],
    expected_sources: str,
) -> None:
    with pytest.raises(ValueError, match=expected_sources):
        _compile(_request(**body), declared=frozenset({"model_option"}))


def test_flattened_size_is_compiled_before_pure_diffusion_dispatch() -> None:
    plan = _compile(_request(size="768x512"))

    params = plan.clone_sampling_params_list()[0]
    assert (params.height, params.width) == (512, 768)
    assert plan.controls["height"] == 512
    assert plan.controls["width"] == 768


def test_declared_dimension_keeps_prompt_control_and_moves_sampling_owner() -> None:
    plan = _compile(_request(height=512), declared=frozenset({"height"}))

    params = plan.clone_sampling_params_list()[0]
    assert plan.controls["height"] == 512
    assert params.height is None
    assert params.extra_args["height"] == 512


def test_nested_controls_are_preserved_without_dispatcher_rereads() -> None:
    plan = _compile(
        _request(
            extra_body={
                "modalities": ["image"],
                "negative_prompt": "low quality",
            }
        )
    )

    assert plan.controls["modalities"] == ["image"]
    assert plan.controls["negative_prompt"] == "low quality"


def test_internal_diffusion_state_is_not_a_public_root_field() -> None:
    plan = _compile(
        _request(
            modules={"must": "not become request state"},
            extra_args={"modules": {"pipeline": "owned"}},
        )
    )

    params = plan.clone_sampling_params_list()[0]
    assert params.modules == {}
    assert params.extra_args["modules"] == {"pipeline": "owned"}


def test_flattened_size_reaches_mixed_final_consumers_and_preserves_defaults() -> None:
    ar_default = SamplingParams(max_tokens=4353)
    ar_default.extra_args = {"ar_default": True}
    diffusion_default = OmniDiffusionSamplingParams(extra_args={"diffusion_default": True})

    plan = _compile(
        _request(size="768x512"),
        stage_types=("llm", "diffusion"),
        defaults=(ar_default, diffusion_default),
        comprehension_stage_index=0,
    )

    ar_params, diffusion_params = plan.clone_sampling_params_list()
    assert ar_params.max_tokens == 4353
    assert ar_params.extra_args == {
        "ar_default": True,
        "target_h": 512,
        "target_w": 768,
    }
    assert (diffusion_params.height, diffusion_params.width) == (512, 768)
    assert diffusion_params.extra_args == {"diffusion_default": True}


@pytest.mark.parametrize("fan_out", [False, True])
def test_registry_declaration_controls_non_diffusion_fan_out(fan_out: bool) -> None:
    ar_default = SamplingParams(max_tokens=100)
    diffusion_default = OmniDiffusionSamplingParams()
    plan = _compile(
        _request(max_tokens=32),
        stage_types=("llm", "diffusion"),
        defaults=(ar_default, diffusion_default),
        comprehension_stage_index=0,
        declared=frozenset({"max_tokens"}),
        fan_out_declared=fan_out,
    )

    ar_params, diffusion_params = plan.clone_sampling_params_list()
    assert ar_params.max_tokens == (32 if fan_out else 100)
    assert ("max_tokens" in (ar_params.extra_args or {})) is fan_out
    assert diffusion_params.extra_args["max_tokens"] == 32


def test_fanned_out_root_conflicts_with_the_same_non_diffusion_stage_key() -> None:
    with pytest.raises(ValueError, match="provided more than once"):
        _compile(
            _request(
                max_tokens=32,
                sampling_params_list=[{"extra_args": {"max_tokens": 64}}],
            ),
            stage_types=("llm", "diffusion"),
            defaults=(SamplingParams(), OmniDiffusionSamplingParams()),
            comprehension_stage_index=0,
            declared=frozenset({"max_tokens"}),
            fan_out_declared=True,
        )


def test_request_stage_values_overlay_defaults_once() -> None:
    plan = _compile(
        _request(
            extra_args={"global": 1},
            sampling_params_list=[
                {"temperature": 0.7, "stop": "END"},
                {"guidance_scale": 4.0, "extra_args": {"local": 2}},
            ],
        ),
        stage_types=("llm", "diffusion"),
        defaults=(
            SamplingParams(temperature=0.2, max_tokens=99),
            OmniDiffusionSamplingParams(extra_args={"default": 3}),
        ),
        comprehension_stage_index=0,
    )

    ar_params, diffusion_params = plan.clone_sampling_params_list()
    assert (ar_params.temperature, ar_params.max_tokens) == (0.7, 99)
    assert ar_params.stop == ["END"]
    assert ar_params.output_text_buffer_length == 2
    assert diffusion_params.guidance_scale == 4.0
    assert diffusion_params.extra_args == {"default": 3, "global": 1, "local": 2}


@pytest.mark.parametrize(
    "body",
    [
        {"sampling_params_list": [{"guidance_scale": 4.0}]},
        {"extra_body": {"sampling_params_list": [{"guidance_scale": 4.0}]}},
    ],
)
def test_stage_list_ingress_reaches_the_same_consumer(body: dict[str, object]) -> None:
    params = _compile(_request(**body)).clone_sampling_params_list()[0]

    assert params.guidance_scale == 4.0


def test_root_and_nested_stage_lists_conflict() -> None:
    with pytest.raises(ValueError, match="request.sampling_params_list, request.extra_body.sampling_params_list"):
        _compile(
            _request(
                sampling_params_list=[{}],
                extra_body={"sampling_params_list": [{}]},
            )
        )


def test_global_and_stage_request_values_cannot_overwrite_each_other() -> None:
    with pytest.raises(ValueError) as exc_info:
        _compile(
            _request(
                extra_args={"seed": 1},
                sampling_params_list=[{"extra_args": {"seed": 2}}],
            )
        )

    assert str(exc_info.value) == (
        'Parameter "seed" was provided more than once: '
        "request.extra_args.seed, request.sampling_params_list[0].extra_args.seed."
    )


def test_duplicate_detection_precedes_value_parsing() -> None:
    with pytest.raises(ValueError, match="provided more than once"):
        _compile(
            _request(
                num_inference_steps=[],
                sampling_params_list=[{"num_inference_steps": 2}],
            )
        )


def test_deprecation_is_logged_only_after_a_request_compiles(mocker) -> None:
    from vllm_omni.entrypoints.openai import diffusion_request_utils

    warning_once = mocker.patch.object(diffusion_request_utils.logger, "warning_once")
    _compile(_request(extra_params={"key": 1}))
    warning_once.assert_called_once()

    warning_once.reset_mock()
    with pytest.raises(ValueError, match="provided more than once"):
        _compile(
            _request(
                extra_params={"key": 1},
                sampling_params_list=[{"extra_args": {"key": 2}}],
            )
        )
    warning_once.assert_not_called()


def test_same_key_is_allowed_in_distinct_stages() -> None:
    plan = _compile(
        _request(sampling_params_list=[{"extra_args": {"key": 1}}, {"extra_args": {"key": 2}}]),
        stage_types=("llm", "diffusion"),
    )

    first, second = plan.clone_sampling_params_list()
    assert first.extra_args["key"] == 1
    assert second.extra_args["key"] == 2


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("num_inference_steps", [], "num_inference_steps must be an integer."),
        ("layers", 999, "layers must be between 2 and 10"),
    ],
)
def test_value_parsers_run_before_dispatch(field: str, value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _compile(_request(**{field: value}))


def test_undeclared_container_values_remain_pipeline_owned() -> None:
    raw = {
        "layers": "pipeline-specific",
        "num_inference_steps": {"schedule": "custom"},
    }

    params = _compile(_request(extra_args=raw)).clone_sampling_params_list()[0]

    assert params.extra_args == raw


def test_declared_container_value_uses_the_serving_parser() -> None:
    params = _compile(
        _request(extra_args={"num_inference_steps": "12"}),
        declared=frozenset({"num_inference_steps"}),
    ).clone_sampling_params_list()[0]

    assert params.extra_args["num_inference_steps"] == 12


def test_sampling_params_list_entries_must_be_objects() -> None:
    with pytest.raises(ValueError, match=r"sampling_params_list\[0\] must be a JSON object"):
        _compile(_request(sampling_params_list=[None]))


def test_unknown_direct_stage_field_is_rejected_by_the_target_stage() -> None:
    with pytest.raises(ValueError, match="unsupported parameter"):
        _compile(_request(sampling_params_list=[{"unknown_runtime_field": 1}]))


def test_too_many_stage_entries_are_rejected() -> None:
    with pytest.raises(ValueError, match="pipeline has 1 stages"):
        _compile(_request(sampling_params_list=[{}, {}]))


def test_plan_returns_isolated_stage_parameters() -> None:
    plan = _compile(
        _request(extra_args={"nested": {"value": 1}}),
        defaults=(OmniDiffusionSamplingParams(extra_args={"default": True}),),
    )

    first = plan.clone_sampling_params_list()
    second = plan.clone_sampling_params_list()
    first[0].extra_args["nested"]["value"] = 9

    assert second[0].extra_args["nested"]["value"] == 1
    assert second[0].extra_args["default"] is True
