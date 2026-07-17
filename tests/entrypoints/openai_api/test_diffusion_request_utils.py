# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import warnings

import pytest

from vllm_omni.entrypoints.openai.diffusion_request_utils import normalize_diffusion_request_extra_args

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_normalize_diffusion_request_extra_args_preserves_unknown_model_keys() -> None:
    extra_args = {"cfg_text_scale": 7.0, "sample_solver": "euler"}

    normalized = normalize_diffusion_request_extra_args(
        provided_root_fields={"seed"},
        extra_args=extra_args,
    )

    assert normalized == extra_args
    assert normalized is not extra_args


def test_normalize_diffusion_request_extra_args_accepts_non_overlapping_legacy_keys() -> None:
    with pytest.warns(DeprecationWarning, match="extra_params is deprecated"):
        normalized = normalize_diffusion_request_extra_args(
            extra_args={"cfg_text_scale": 7.0},
            extra_params={"sample_solver": "euler"},
        )

    assert normalized == {"cfg_text_scale": 7.0, "sample_solver": "euler"}


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


def test_normalize_diffusion_request_extra_args_rejects_alias_conflict_without_warning() -> None:
    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter("always")
        with pytest.raises(ValueError) as exc_info:
            normalize_diffusion_request_extra_args(
                extra_args={"sample_solver": "euler"},
                extra_params={"sample_solver": "euler"},
            )

    assert not warning_records
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
