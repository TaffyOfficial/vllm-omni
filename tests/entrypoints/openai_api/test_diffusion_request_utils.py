# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest

from vllm_omni.entrypoints.openai.diffusion_request_utils import (
    compile_diffusion_request_overrides,
    normalize_diffusion_request_extra_args,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_normalize_diffusion_request_extra_args_preserves_unknown_model_keys() -> None:
    extra_args = {"cfg_text_scale": 7.0, "sample_solver": "euler"}

    normalized = normalize_diffusion_request_extra_args(
        provided_root_fields={"seed"},
        extra_args=extra_args,
    )

    assert normalized == extra_args
    assert normalized is not extra_args


def test_normalize_diffusion_request_extra_args_accepts_non_overlapping_legacy_keys(mocker) -> None:
    warning_once = mocker.patch("vllm_omni.entrypoints.openai.diffusion_request_utils.logger.warning_once")

    normalized = normalize_diffusion_request_extra_args(
        extra_args={"cfg_text_scale": 7.0},
        extra_params={"sample_solver": "euler"},
    )

    assert normalized == {"cfg_text_scale": 7.0, "sample_solver": "euler"}
    warning_once.assert_called_once_with(
        "extra_params is deprecated; use extra_args for model-specific diffusion request parameters."
    )


def test_normalize_diffusion_request_extra_args_merges_non_overlapping_nested_compatibility_form() -> None:
    normalized = normalize_diffusion_request_extra_args(
        extra_args={"cfg_text_scale": 7.0},
        nested_extra_args={"sample_solver": "euler"},
    )

    assert normalized == {"cfg_text_scale": 7.0, "sample_solver": "euler"}


def test_normalize_diffusion_request_extra_args_rejects_flattened_nested_conflict() -> None:
    with pytest.raises(ValueError) as exc_info:
        normalize_diffusion_request_extra_args(
            provided_root_fields={"cfg_text_scale"},
            nested_extra_args={"cfg_text_scale": 7.0},
        )

    assert str(exc_info.value) == (
        'Parameter "cfg_text_scale" was provided more than once: '
        "request.cfg_text_scale, request.extra_body.extra_args.cfg_text_scale."
    )


@pytest.mark.parametrize("name", ["extra_args", "extra_params"])
def test_normalize_diffusion_request_extra_args_rejects_non_object_containers(name: str) -> None:
    kwargs = {name: ["not", "an", "object"]}

    with pytest.raises(ValueError, match=rf"^{name} must be a JSON object\.$"):
        normalize_diffusion_request_extra_args(**kwargs)


def test_normalize_diffusion_request_extra_args_rejects_non_string_keys() -> None:
    with pytest.raises(ValueError, match=r"^extra_args must be a JSON object with string keys\.$"):
        normalize_diffusion_request_extra_args(extra_args={1: "not a JSON object key"})


def test_normalize_diffusion_request_extra_args_rejects_root_conflict() -> None:
    with pytest.raises(ValueError) as exc_info:
        normalize_diffusion_request_extra_args(
            provided_root_fields={"flow_shift", "extra_args"},
            extra_args={"flow_shift": 3.0},
        )

    assert str(exc_info.value) == (
        'Parameter "flow_shift" was provided more than once: request.flow_shift, request.extra_args.flow_shift.'
    )


def test_normalize_diffusion_request_extra_args_rejects_root_alias_conflict() -> None:
    with pytest.raises(ValueError) as exc_info:
        normalize_diffusion_request_extra_args(
            provided_root_fields={"cfg_scale", "true_cfg_scale"},
            root_field_aliases={"cfg_scale": "true_cfg_scale"},
        )

    assert str(exc_info.value) == (
        'Parameter "true_cfg_scale" was provided more than once: request.cfg_scale, request.true_cfg_scale.'
    )


def test_normalize_diffusion_request_extra_args_rejects_alias_conflict_without_warning(mocker) -> None:
    warning_once = mocker.patch("vllm_omni.entrypoints.openai.diffusion_request_utils.logger.warning_once")
    with pytest.raises(ValueError) as exc_info:
        normalize_diffusion_request_extra_args(
            extra_args={"sample_solver": "euler"},
            extra_params={"sample_solver": "euler"},
        )

    warning_once.assert_not_called()
    assert str(exc_info.value) == (
        'Parameter "sample_solver" was provided more than once: '
        "request.extra_args.sample_solver, request.extra_params.sample_solver."
    )


def test_normalize_diffusion_request_extra_args_reports_all_conflicts_deterministically() -> None:
    with pytest.raises(ValueError) as exc_info:
        normalize_diffusion_request_extra_args(
            provided_root_fields={"seed", "flow_shift"},
            extra_args={"seed": 1, "flow_shift": 3.0},
            extra_params={"seed": 2},
        )

    assert str(exc_info.value) == (
        'Diffusion request parameters were provided more than once: "flow_shift": '
        'request.flow_shift, request.extra_args.flow_shift; "seed": request.seed, '
        "request.extra_args.seed, request.extra_params.seed."
    )


def test_normalize_diffusion_request_extra_args_does_not_treat_stage_defaults_as_request_conflicts() -> None:
    normalized = normalize_diffusion_request_extra_args(
        provided_root_fields=(),
        extra_args={"flow_shift": 3.0},
    )

    assert normalized == {"flow_shift": 3.0}


def test_normalize_diffusion_request_extra_args_rejects_stage_request_conflict() -> None:
    with pytest.raises(ValueError) as exc_info:
        normalize_diffusion_request_extra_args(
            provided_root_fields={"seed"},
            stage_extra_args={1: {"seed": 111}},
        )

    assert str(exc_info.value) == (
        'Parameter "seed" was provided more than once: request.seed, request.sampling_params_list[1].extra_args.seed.'
    )


def test_normalize_diffusion_request_extra_args_allows_same_key_in_distinct_stages() -> None:
    normalized = normalize_diffusion_request_extra_args(
        stage_extra_args={
            1: {"seed": 111},
            2: {"seed": 222},
        },
    )

    assert normalized == {}


def test_normalize_diffusion_request_extra_args_rejects_non_object_stage_extra_args() -> None:
    with pytest.raises(
        ValueError,
        match=r"^sampling_params_list\[1\]\.extra_args must be a JSON object\.$",
    ):
        normalize_diffusion_request_extra_args(
            stage_extra_args={1: ["not", "an", "object"]},
        )


def test_compile_routes_registry_declared_root_to_extra_args() -> None:
    compiled = compile_diffusion_request_overrides(
        root_values={"max_tokens": 32},
        nested_root_values={},
        sampling_root_fields={},
        declared_extra_fields={"max_tokens"},
        control_root_fields=set(),
    )

    assert compiled.sampling_overrides == {}
    assert compiled.extra_args == {"max_tokens": 32}


def test_compile_uses_sampling_alias_without_model_declaration() -> None:
    compiled = compile_diffusion_request_overrides(
        root_values={"cfg_scale": 3.0},
        nested_root_values={},
        sampling_root_fields={"cfg_scale": "true_cfg_scale"},
        declared_extra_fields=set(),
        control_root_fields=set(),
    )

    assert compiled.sampling_overrides == {"true_cfg_scale": 3.0}
    assert compiled.extra_args == {}


def test_compile_preserves_shared_control_for_model_declared_root() -> None:
    compiled = compile_diffusion_request_overrides(
        root_values={"negative_prompt": "avoid blur"},
        nested_root_values={},
        sampling_root_fields={},
        declared_extra_fields={"negative_prompt"},
        control_root_fields={"negative_prompt"},
    )

    assert compiled.extra_args == {"negative_prompt": "avoid blur"}
    assert compiled.control_overrides == {"negative_prompt": "avoid blur"}


def test_compile_registry_declaration_changes_cfg_scale_conflict_key() -> None:
    with pytest.raises(ValueError) as exc_info:
        compile_diffusion_request_overrides(
            root_values={"cfg_scale": 3.0},
            nested_root_values={},
            sampling_root_fields={"cfg_scale": "true_cfg_scale"},
            declared_extra_fields={"cfg_scale"},
            control_root_fields=set(),
            extra_args={"cfg_scale": 7.0},
        )

    assert str(exc_info.value) == (
        'Parameter "cfg_scale" was provided more than once: request.cfg_scale, request.extra_args.cfg_scale.'
    )


def test_compile_uses_model_aware_aliases_for_stage_conflicts() -> None:
    with pytest.raises(ValueError) as exc_info:
        compile_diffusion_request_overrides(
            root_values={"cfg_scale": 3.0},
            nested_root_values={},
            sampling_root_fields={"cfg_scale": "true_cfg_scale"},
            declared_extra_fields={"cfg_scale"},
            control_root_fields=set(),
            stage_extra_args={1: {"cfg_scale": 7.0}},
        )

    assert str(exc_info.value) == (
        'Parameter "cfg_scale" was provided more than once: '
        "request.cfg_scale, request.sampling_params_list[1].extra_args.cfg_scale."
    )
