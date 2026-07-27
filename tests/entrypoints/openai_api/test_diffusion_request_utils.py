# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any

import pytest
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.sampling_params import SamplingParams

from vllm_omni.entrypoints.openai.diffusion_request_utils import (
    DiffusionChatRequestContext,
    DiffusionChatRequestPlan,
    compile_diffusion_chat_request_plan,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

STANDARD_FIELDS = {"temperature", "max_tokens", "seed"}


def _request(**kwargs: Any) -> ChatCompletionRequest:
    return ChatCompletionRequest(model="test", messages=[], **kwargs)


GLOBAL_SOURCE_PATHS = (
    "request",
    "extra_body",
    "extra_args",
    "extra_params",
    "extra_body.extra_args",
    "extra_body.extra_params",
)


def _global_source(path: str, key: str, value: object) -> dict[str, object]:
    body: dict[str, object] = {key: value}
    if path == "request":
        return body
    for container in reversed(path.split(".")):
        body = {container: body}
    return body


def _compile(
    request: ChatCompletionRequest,
    *,
    stage_types: tuple[str, ...] = ("diffusion",),
    defaults: tuple[object, ...] = (),
    comprehension_stage_index: int | None = None,
    declared: frozenset[str] = frozenset(),
    fan_out_declared: bool = False,
    supported_modalities: frozenset[str] = frozenset({"audio", "image", "text"}),
) -> DiffusionChatRequestPlan:
    return compile_diffusion_chat_request_plan(
        request=request,
        context=DiffusionChatRequestContext(
            stage_types=stage_types,
            default_sampling_params_list=defaults,
            comprehension_stage_index=comprehension_stage_index,
            standard_sampling_fields=frozenset(STANDARD_FIELDS),
            declared_extra_fields=declared,
            apply_declared_to_non_diffusion=fan_out_declared,
            supported_modalities=supported_modalities,
        ),
    )


@pytest.mark.parametrize("source", GLOBAL_SOURCE_PATHS)
@pytest.mark.parametrize(
    ("dispatcher", "consumer", "key", "value"),
    [
        pytest.param("pure", "diffusion", "model_option", 1, id="pure-diffusion"),
        pytest.param("mixed", "fan-out", "seed", 32, id="mixed-fan-out"),
        pytest.param("mixed", "defaults", "seed", None, id="mixed-defaults"),
    ],
)
def test_global_source_dispatcher_consumer_matrix(
    source: str,
    dispatcher: str,
    consumer: str,
    key: str,
    value: object,
) -> None:
    body = _global_source(source, key, value)
    if dispatcher == "pure":
        plan = _compile(_request(**body), declared=frozenset({key}))
        assert plan.clone_sampling_params_list()[0].extra_args[key] == value
        return

    plan = _compile(
        _request(**body),
        stage_types=("llm", "diffusion"),
        defaults=(SamplingParams(seed=7), OmniDiffusionSamplingParams(extra_args={"default": True})),
        comprehension_stage_index=0,
        declared=frozenset({key}),
        fan_out_declared=True,
    )
    ar_params, diffusion_params = plan.clone_sampling_params_list()
    if consumer == "fan-out":
        assert ar_params.seed == value
        assert ar_params.extra_args[key] == value
        assert diffusion_params.extra_args == {"default": True, key: value}
    else:
        assert ar_params.seed == 7
        assert key not in (ar_params.extra_args or {})
        assert diffusion_params.extra_args == {"default": True}


@pytest.mark.parametrize(
    "source",
    ("extra_args", "extra_params", "extra_body.extra_args", "extra_body.extra_params"),
)
def test_pipeline_owned_global_none_preserves_stage_default(source: str) -> None:
    plan = _compile(
        _request(**_global_source(source, "solver", None)),
        defaults=(OmniDiffusionSamplingParams(extra_args={"solver": "euler"}),),
    )

    assert plan.clone_sampling_params_list()[0].extra_args["solver"] == "euler"


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


def test_declared_dimension_keeps_prompt_control_and_moves_sampling_owner() -> None:
    plan = _compile(_request(height=512), declared=frozenset({"height"}))

    params = plan.clone_sampling_params_list()[0]
    assert plan.controls["height"] == 512
    assert params.height is None
    assert params.extra_args["height"] == 512


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


def test_stage_replacement_revalidates_sampling_params() -> None:
    with pytest.raises(ValueError, match="Invalid sampling parameters for stage 0"):
        _compile(
            _request(sampling_params_list=[{"top_p": 0}]),
            stage_types=("llm",),
            defaults=(SamplingParams(),),
            comprehension_stage_index=0,
        )


def test_stage_replacement_resets_sampling_params_derived_state() -> None:
    plan = _compile(
        _request(sampling_params_list=[{"stop": [], "stop_token_ids": [2]}]),
        stage_types=("llm",),
        defaults=(SamplingParams(stop=["END"], stop_token_ids=[1]),),
        comprehension_stage_index=0,
    )

    (params,) = plan.clone_sampling_params_list()
    assert params.output_text_buffer_length == 0
    assert params._all_stop_token_ids == {2}


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
